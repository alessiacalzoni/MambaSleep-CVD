import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
import logging
import torch.utils.checkpoint as checkpoint

# Try to import selective_scan_fn
# try:
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.ops.triton.layer_norm import RMSNorm
from causal_conv1d import causal_conv1d_fn
# except ImportError:
#     selective_scan_fn = None
#     causal_conv1d_fn = None

class EfficientCoupledMambaBlock(nn.Module):
    """
    Efficient Coupled Mamba Block using Mamba Kernel Parallelism.
    
    Logic:
    The summed state H_t = P_t * H_{t-1} + U_t
    where P_t = sum_k A_bar^k_t, U_t = sum_k B_bar^k_t * x_t^k.
    
    To use the Mamba kernel (selective_scan_fn), we treat the flattened state H (size D*N)
    as the "channel" dimension of a standard Mamba scan with d_state=1.
    
    Standard Mamba Scan:
    h_t = exp(delta * A) * h_{t-1} + delta * B * x_t
    
    We map our system to this:
    - Effective State: H_t (flattened to size D*N)
    - Effective Input: U_t (size D*N)
    - Effective A: Identity (since P_t varies, we absorb it into delta)
      Actually, standard scan takes A (fixed).
      We want: H_t = P_t * H_{t-1} + U_t.
      Let's set A_eff = -1 (fixed).
      Let delta_eff = -ln(P_t).
      Then exp(delta_eff * A_eff) = exp(-ln(P_t) * -1) = exp(ln(P_t)) = P_t.
      
      Input term: delta_eff * B_eff * x_eff.
      We want this to equal U_t.
      So B_eff * x_eff = U_t / delta_eff.
      We can pre-compute U_t, set B_eff=1, and pass x_eff = U_t / delta_eff.
      
      Wait, selective_scan_fn takes:
      u: (B, D, L)
      delta: (B, D, L)
      A: (D, N) -> Here N=1. A is (D, 1).
      B: (B, N, L) -> Here N=1. B is (B, 1, L).
      C: (B, N, L) -> Here N=1. C is (B, 1, L).
      D: (D)
      
      We map:
      D (Mamba dim) -> D_model * N_modalities
      N (Mamba state) -> 1
      
      A_eff = -1 * ones(D*N, 1)
      delta_eff = -ln(P_t) (B, D*N, L)
      B_eff = ones(B, 1, L)
      C_eff = ones(B, 1, L) (to output the state directly)
      u_eff = U_t / delta_eff (B, D*N, L)
      D_eff = 0
      
      Output y = C_eff * h + D_eff * u = 1 * H + 0 = H.
      
      So the kernel returns H_t.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, num_modalities=2, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self.num_modalities = num_modalities
        
        # Total flattened dimension for the summed state
        self.d_total = self.d_inner * self.d_state

        # Per-modality components
        self.in_projs = nn.ModuleList([
            nn.Linear(d_model, self.d_inner * 2, bias=False) for _ in range(num_modalities)
        ])

        self.conv1ds = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=True,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
            ) for _ in range(num_modalities)
        ])

        self.x_projs = nn.ModuleList([
            nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False) for _ in range(num_modalities)
        ])
        
        self.dt_projs = nn.ModuleList([
            nn.Linear(self.dt_rank, self.d_inner, bias=True) for _ in range(num_modalities)
        ])

        # A parameter
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_logs = nn.ParameterList([
            nn.Parameter(A_log.clone()) for _ in range(num_modalities)
        ])
        for A_log in self.A_logs:
            A_log._no_weight_decay = True
        
        self.Ds = nn.ParameterList([
            nn.Parameter(torch.ones(self.d_inner)) for _ in range(num_modalities)
        ])

        self.out_projs = nn.ModuleList([
            nn.Linear(self.d_inner, d_model, bias=False) for _ in range(num_modalities)
        ])
        
        self.act = nn.SiLU()

        # Initialize dt bias
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        
        for dt_proj in self.dt_projs:
            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            dt_proj.bias._no_reinit = True

    def _forward_chunk(self, chunk_inputs_stacked, H_prev, conv_states_stacked):
        """
        Internal function to process a single chunk.
        Args:
            chunk_inputs_stacked: (Num_Mods, Batch, Chunk_Len, D_model)
            H_prev: (Batch, D_inner, D_state) - Previous Mamba state
            conv_states_stacked: (Num_Mods, Batch, D_inner, D_conv-1) - Previous Conv states
        Returns:
            chunk_outputs_stacked: (Num_Mods, Batch, Chunk_Len, D_model)
            H_new: (Batch, D_inner, D_state)
            conv_states_new_stacked: (Num_Mods, Batch, D_inner, D_conv-1)
        """
        num_mods, batch_size, current_chunk_len, _ = chunk_inputs_stacked.shape
        device = chunk_inputs_stacked.device
        
        xs, zs = [], []
        A_bars, B_bars = [], []
        
        new_conv_states = []
        
        # 1. Per-Modality Processing (Conv + Projections)
        for i in range(self.num_modalities):
            # Input projection
            xz = self.in_projs[i](chunk_inputs_stacked[i])
            x, z = xz.chunk(2, dim=-1)
            
            # Retrieve specific state
            # causal_conv1d requires stride(1)==1 for initial_states (Channel Last layout) (B, D, W)
            conv_state = conv_states_stacked[i].permute(0, 2, 1).contiguous().permute(0, 2, 1) if conv_states_stacked is not None else None
            
             # Convolution with State Passing
            if causal_conv1d_fn is not None:
                # causal_conv1d_fn expects (B, D, L)
                x_t = x.transpose(1, 2)
                
                # Run conv with state update
                x_t, new_conv_state = causal_conv1d_fn(
                    x=x_t,
                    weight=rearrange(self.conv1ds[i].weight, "d 1 w -> d w"),
                    bias=self.conv1ds[i].bias,
                    activation="silu",
                    initial_states=conv_state, 
                    return_final_states=True
                )
                new_conv_states.append(new_conv_state)
                x = x_t.transpose(1, 2)
            else:
                logger.warning("causal_conv1d_fn is not available. Using fallback implementation.")
                # Fallback implementation
                x_t = x.transpose(1, 2)
                
                # Manual padding with state
                if conv_state is None or torch.all(conv_state == 0): # Assuming 0 init means None/first chunk equivalent in fallback logic if needed
                    # First chunk: standard padding
                    padding = (self.d_conv - 1, 0)
                    x_padded = F.pad(x_t, padding)
                else:
                    # Subsequent chunks: prepend state
                    x_padded = torch.cat([conv_state, x_t], dim=2)
                
                # Update state for next chunk (save last d_conv-1 elements)
                if self.d_conv > 1:
                    new_conv_states.append(x_t[:, :, -(self.d_conv - 1):])
                else:
                    new_conv_states.append(torch.zeros_like(x_t))
                
                # Run Conv (padding=0 since we manually padded above)
                x_conv = F.conv1d(x_padded, self.conv1ds[i].weight, self.conv1ds[i].bias, groups=self.d_inner)
                x = self.act(x_conv).transpose(1, 2)

            xs.append(x)
            zs.append(z)
            
            # Projections (Parameter Computation)
            x_dbl = self.x_projs[i](x) 
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_projs[i](dt)
            dt = F.softplus(dt) 
            
            # Discretize
            A_bar = torch.exp(dt.unsqueeze(-1) * (-torch.exp(self.A_logs[i])))
            A_bars.append(A_bar)
            
            delta_x = x * dt
            B_bar_x = delta_x.unsqueeze(-1) * B.unsqueeze(2)
            B_bars.append(B_bar_x)

        # 2. Summed System for Cooperation
        P = sum(A_bars) / self.num_modalities
        U = sum(B_bars) / self.num_modalities
        
        # Flatten for kernel
        P_flat = rearrange(P, "b l d n -> b l (d n)")
        U_flat = rearrange(U, "b l d n -> b l (d n)")
        
        # 3. Selective Scan with State Passing
        H_prev_in = H_prev

        if selective_scan_fn is not None:
             # PREPEND TRICK
            B_size = batch_size
            D_total = self.d_total
            
            # P_prepend -> exp(-1) so delta=1
            P_val = math.exp(-1) # ~0.3678
            P_prepend = torch.full((B_size, 1, D_total), P_val, device=device, dtype=P_flat.dtype)
            
            # U_prepend -> H_prev
            H_prev_flat = rearrange(H_prev, "b d n -> b (d n)")
            U_prepend = H_prev_flat.unsqueeze(1) # (B, 1, D_total)
            
            # Concatenate
            P_full = torch.cat([P_prepend, P_flat], dim=1) # (B, L+1, D)
            U_full = torch.cat([U_prepend, U_flat], dim=1) # (B, L+1, D)
            del P, U, P_flat, U_flat, P_prepend, U_prepend
            
            # Prepare for kernel (B, D, L)
            P_full = P_full.clamp(min=1e-6)
            delta_eff = -torch.log(P_full)
            del P_full
            
            delta_eff_safe = delta_eff.clamp(min=1e-6)
            del delta_eff
            u_eff = U_full / delta_eff_safe
            del U_full
            
            u_in = u_eff.transpose(1, 2).contiguous()
            del u_eff
            delta_in = delta_eff_safe.transpose(1, 2).contiguous()
            del delta_eff_safe
            
            A_in = -torch.ones(D_total, 1, device=device, dtype=torch.float32)
            B_in = torch.ones(B_size, 1, current_chunk_len + 1, device=device, dtype=torch.float32)
            C_in = torch.ones(B_size, 1, current_chunk_len + 1, device=device, dtype=torch.float32)
            D_in = torch.zeros(D_total, device=device, dtype=torch.float32)
            
            # Run Scan
            H_flat_out = selective_scan_fn(
                u_in, delta_in, A_in, B_in, C_in, D_in,
                z=None, delta_bias=None, delta_softplus=False, return_last_state=False
            )
            del u_in, delta_in, A_in, B_in, C_in, D_in
            
            # H_flat_out: (B, D, L+1) -> (B, L+1, D)
            H_flat_out = H_flat_out.transpose(1, 2)
            
            # Slice output: keep only the chunk
            H_flat_chunk = H_flat_out[:, 1:, :] # (B, L, D)
            
            # Update State for next chunk (Last element)
            # .contiguous() breaks the view chain so H_flat_out can be freed
            H_new = rearrange(H_flat_chunk[:, -1, :], "b (d n) -> b d n", d=self.d_inner, n=self.d_state).contiguous()
            
            # Reshape for recovery
            Hs = rearrange(H_flat_chunk, "b l (d n) -> b l d n", d=self.d_inner, n=self.d_state)
            del H_flat_out, H_flat_chunk
            
            # State for recovery: H_prev_in
            H_shifted_start = H_prev_in.view(batch_size, 1, self.d_inner, self.d_state)
            
        else:
             # Manual Scan Fallback with state passing
            H = H_prev # Start with state from prev chunk
            Hs_list = []
            for t in range(current_chunk_len):
                H = P[:, t] * H + U[:, t]
                Hs_list.append(H)
            Hs = torch.stack(Hs_list, dim=1)
            del Hs_list, P, U, P_flat, U_flat
            
            # Update state
            H_new = H
            
            H_shifted_start = H_prev_in.unsqueeze(1)
        
        # 4. Output Recovery
        H_shifted = torch.cat([H_shifted_start, Hs[:, :-1, :, :]], dim=1)
        del Hs, H_shifted_start
        
        chunk_outs = []
        for i in range(self.num_modalities):
            x = xs[i]
            x_dbl = self.x_projs[i](x)
            _, _, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            
            h_k = A_bars[i] * H_shifted + B_bars[i]
            A_bars[i] = None
            B_bars[i] = None
            
            y_k = torch.sum(h_k * C.unsqueeze(2), dim=-1)
            del h_k
            y_k = y_k + xs[i] * self.Ds[i]
            y_k = y_k * F.silu(zs[i])
            
            out = self.out_projs[i](y_k)
            chunk_outs.append(out)
            
        # Stack outputs
        chunk_outputs_stacked = torch.stack(chunk_outs, dim=0) # (Num_Mods, B, L, D_model)
        new_conv_states_stacked = torch.stack(new_conv_states, dim=0) # (Num_Mods, B, D_in, K)
        
        return chunk_outputs_stacked, H_new, new_conv_states_stacked


    def forward(self, inputs: list[torch.Tensor], chunk_size: int = 512):
        """
        Forward pass with Gradient Checkpointing and Chunked Processing.
        """
        batch_size, seq_len, _ = inputs[0].shape
        device = inputs[0].device
        
        # If chunking is disabled or sequence is short, run full
        if chunk_size <= 0 or seq_len <= chunk_size:
            return self._forward_full(inputs)
             
        # Initialize States
        # Conv state: Stacked tensor (Num_Mods, Batch, D_inner, d_conv-1)
        # Use zeros as initial state if None
        conv_states_stacked = torch.zeros(
            self.num_modalities, batch_size, self.d_inner, self.d_conv - 1, 
            device=device, dtype=inputs[0].dtype
        )
        
        # SSM State: (Batch, D_inner, D_state). Initialized to 0.
        H_prev = torch.zeros(batch_size, self.d_inner, self.d_state, device=device, dtype=inputs[0].dtype)

        # Prepare outputs list
        all_outputs = []
        
        # Chunking Loop
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, seq_len)
            
            # 1. Prepare Chunk Inputs - Stack them for checkpointing
            chunk_inputs_list = [x[:, start:end, :] for x in inputs]
            chunk_inputs_stacked = torch.stack(chunk_inputs_list, dim=0) # (Num_Mods, B, L, D)
            
            # 2. Checkpoint Call
            # We must pass tensors requiring grad to make checkpointing effective for parameters.
            # Inputs require grad usually. H_prev and conv_states strictly don't for the first chunk, but do later.
            if self.training:
                chunk_outputs_stacked, H_new, new_conv_states_stacked = checkpoint.checkpoint(
                    self._forward_chunk, 
                    chunk_inputs_stacked, 
                    H_prev, 
                    conv_states_stacked,
                    use_reentrant=False # Recommended to avoid obscure bugs
                )
            else:
                 chunk_outputs_stacked, H_new, new_conv_states_stacked = self._forward_chunk(chunk_inputs_stacked, H_prev, conv_states_stacked)
            
            # 3. Update States
            H_prev = H_new
            conv_states_stacked = new_conv_states_stacked
            
            # 4. Collect Outputs
            # Unstack
            chunk_outs = [chunk_outputs_stacked[i] for i in range(self.num_modalities)]
            all_outputs.append(chunk_outs)

        # End Chunk Loop
        
        # Aggregate Outputs
        final_outputs = []
        for i in range(self.num_modalities):
            # Collect chunk parts for modality i
            mod_parts = [chunk[i] for chunk in all_outputs]
            final_outputs.append(torch.cat(mod_parts, dim=1))
            
        return final_outputs

    def _forward_full(self, inputs: list[torch.Tensor]):
        batch_size, seq_len, _ = inputs[0].shape
        device = inputs[0].device
        
        # # 1. Align inputs (same as before)
        # lengths = [x.shape[1] for x in inputs]
        # max_len = max(lengths)
        
        # aligned_inputs = []
        # for x in inputs:
        #     if x.shape[1] < max_len:
        #         pad = torch.zeros(batch_size, max_len - x.shape[1], x.shape[2], device=device, dtype=x.dtype)
        #         aligned_inputs.append(torch.cat([x, pad], dim=1))
        #     else:
        #         aligned_inputs.append(x)
        
        # seq_len = max_len
        
        # 2. Process inputs
        xs, zs = [], []
        A_bars, B_bars = [], []
        
        for i in range(self.num_modalities):
            # Input projection
            xz = self.in_projs[i](inputs[i])
            x, z = xz.chunk(2, dim=-1)
            
            # Conv
            if causal_conv1d_fn is not None:
                x = causal_conv1d_fn(
                    x=x.transpose(1, 2),
                    weight=rearrange(self.conv1ds[i].weight, "d 1 w -> d w"),
                    bias=self.conv1ds[i].bias,
                    activation="silu",
                ).transpose(1, 2)
            else:
                logger.warning("causal_conv1d_fn is not available. Using fallback implementation.")
                x = x.transpose(1, 2)
                # Manual padding for causal conv
                x = F.pad(x, (self.d_conv - 1, 0))
                x = self.conv1ds[i](x)
                x = self.act(x).transpose(1, 2)
            
            xs.append(x)
            zs.append(z)
            
            # Projections
            x_dbl = self.x_projs[i](x) # (B, L, dt_rank + 2*d_state)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_projs[i](dt)
            dt = F.softplus(dt) # (B, L, d_inner)
            
            # Discretize
            # A_bar = exp(dt * A)
            # A_bar: (B, L, d_inner, d_state)
            A_bar = torch.exp(dt.unsqueeze(-1) * (-torch.exp(self.A_logs[i])))
            A_bars.append(A_bar)
            
            # B_bar_x = (dt * x).unsqueeze(-1) * B.unsqueeze(2)
            # (B, L, d_inner, d_state)
            delta_x = x * dt
            B_bar_x = delta_x.unsqueeze(-1) * B.unsqueeze(2)
            B_bars.append(B_bar_x)

        # 3. Summed System
        # P_t = sum(A_bars)  (B, L, d_inner, d_state)
        # U_t = sum(B_bars)  (B, L, d_inner, d_state)
        P = sum(A_bars) / self.num_modalities
        U = sum(B_bars) / self.num_modalities
        
        # Flatten for kernel
        # P: (B, L, d_inner*d_state)
        P_flat = rearrange(P, "b l d n -> b l (d n)")
        U_flat = rearrange(U, "b l d n -> b l (d n)")
        
        # 4. Parallel Scan using Kernel Trick
        if selective_scan_fn is not None:
            # Prepare inputs for selective_scan_fn
            # u: (B, D_total, L)
            # delta: (B, D_total, L)
            # A: (D_total, 1) -> -1
            # B: (B, 1, L) -> 1
            # C: (B, 1, L) -> 1
            
            # -------------------------------------------------------------------------
            # THE KERNEL TRICK EXPLAINED
            # -------------------------------------------------------------------------
            # Goal: Solve H_t = P_t * H_{t-1} + U_t using selective_scan_fn.
            # Problem: selective_scan_fn solves h_t = exp(delta * A) * h_{t-1} + delta * B * x_t
            #          where A is FIXED (shape D, N). Our P_t is TIME-VARYING.
            
            # Step 1: Flattening
            # We treat the entire state D_total = (D_inner * D_state) as independent "channels".
            # Kernel View: Channels = D_total, State_Size = 1.
            # Real View:   Channels = D_inner, State_Size = D_state.
            
            # Step 2: Mapping P_t -> delta
            # We want: exp(delta * A) = P_t
            # We set A = -1 (fixed).
            # We set delta = -ln(P_t).
            # Check: exp(-ln(P_t) * -1) = exp(ln(P_t)) = P_t. Correct.
            
            # delta_eff = -ln(P_flat)
            # P is sum of exps, so positive. Avoid log(0).
            P_flat = P_flat.clamp(min=1e-6)
            delta_eff = -torch.log(P_flat)
            
            # Step 3: Mapping U_t -> u
            # We want: delta * B * x = U_t
            # We set B = 1.
            # We set x = u_eff.
            # So we need: delta * u_eff = U_t
            # Therefore: u_eff = U_t / delta.
            
            # u_eff = U_flat / delta_eff
            # Avoid div by 0
            delta_eff_safe = delta_eff.clamp(min=1e-6)
            u_eff = U_flat / delta_eff_safe
            
            # Transpose to (B, D, L) for kernel
            u_in = u_eff.transpose(1, 2).contiguous()
            delta_in = delta_eff_safe.transpose(1, 2).contiguous()
            
            # Step 4: Setting Fixed Parameters
            # A_eff = -1 (Shape: D_total, 1)
            A_in = -torch.ones(self.d_total, 1, device=device, dtype=torch.float32)
            
            # B_in = 1, C_in = 1 (Shape: B, 1, L)
            # We set C=1 so the kernel outputs the state H directly (y = C*h = 1*H = H). I want to extract H
            B_in = torch.ones(batch_size, 1, seq_len, device=device, dtype=torch.float32)
            C_in = torch.ones(batch_size, 1, seq_len, device=device, dtype=torch.float32)
            
            # D_in = 0 (No skip connection inside the scan) because I want to extract H
            D_in = torch.zeros(self.d_total, device=device, dtype=torch.float32)
            
            # Run scan
            # Output y is H (B, D_total, L). Also here, z is none because in calculating H there is not the gating mechanism
            H_flat = selective_scan_fn(
                u_in, delta_in, A_in, B_in, C_in, D_in, 
                z=None, delta_bias=None, delta_softplus=False, return_last_state=False
            )
            
            # Transpose back to (B, L, D_total)
            H_flat = H_flat.transpose(1, 2)
            
            # Reshape to (B, L, d_inner, d_state)
            Hs = rearrange(H_flat, "b l (d n) -> b l d n", d=self.d_inner, n=self.d_state)
            
        else:
            # Fallback to manual scan if kernel not available
            # H_t = P_t * H_{t-1} + U_t
            H = torch.zeros(batch_size, self.d_inner, self.d_state, device=device)
            Hs_list = []
            for t in range(seq_len):
                H = P[:, t] * H + U[:, t]
                Hs_list.append(H)
            Hs = torch.stack(Hs_list, dim=1)

        return self._recover_outputs(inputs, inputs, xs, zs, A_bars, B_bars, Hs, seq_len)

    def _recover_outputs(self, inputs, aligned_inputs, xs, zs, A_bars, B_bars, Hs, seq_len):
        batch_size = inputs[0].shape[0]
        device = inputs[0].device
        outputs = []
        
        H_prev = torch.cat([torch.zeros(batch_size, 1, self.d_inner, self.d_state, device=device), Hs[:, :-1]], dim=1)
        
        for i in range(self.num_modalities):
            # Re-get C
            x = xs[i]
            x_dbl = self.x_projs[i](x)
            _, _, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            
            h_k = A_bars[i] * H_prev + B_bars[i]
            
            y_k = torch.sum(h_k * C.unsqueeze(2), dim=-1)
            y_k = y_k + xs[i] * self.Ds[i]
            y_k = y_k * F.silu(zs[i])
            
            out = self.out_projs[i](y_k)
            outputs.append(out)
            
        return outputs

class EfficientCoupledMambaModel(nn.Module):
    def __init__(
        self, 
        d_model:int =128, 
        d_state:int =16, 
        d_conv:int =4, 
        expand:int =2, 
        n_layers:int =4, 
        num_modalities:int =2, 
        final_pooling:str ="avg",
        norm_type:str ="layer" # "layer" or "rms"
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.n_layers = n_layers
        self.num_modalities = num_modalities
        self.final_pooling = final_pooling
        self.norm_type = norm_type.lower()
        
        # Backbone
        self.coupled_mamba_layers = nn.ModuleList([
            EfficientCoupledMambaBlock(d_model, d_state, d_conv, expand, num_modalities=num_modalities)
            for _ in range(n_layers)
        ])
        
        # Backbone: Separate norms for each layer to fix the shared-norm bug
        NormClass = nn.LayerNorm if self.norm_type == "layer" else RMSNorm
        self.norms = nn.ModuleList([
            nn.ModuleList([NormClass(d_model) for _ in range(num_modalities)])
            for _ in range(n_layers)
        ])

        # Final Norm
        self.final_norm = NormClass(d_model * num_modalities)

    def forward(self, inputs: list[torch.Tensor], mask: torch.Tensor, **kwargs):
        # inputs: [EEG BATCH (B, L, D), ECG BATCH (B, L, D)]
        # mask: (B, L)
        
        # # Project to d_model
        # modalities = [self.proj_mods[i](inputs[i]) for i in range(self.num_modalities)]
        
        for i, (layer, layer_norms) in enumerate(zip(self.coupled_mamba_layers, self.norms)):
            # Pre-Norm: Normalize inputs first
            norm_inputs = [layer_norms[j](inputs[j]) for j in range(self.num_modalities)]
            
            # 1. Block Forward: (B, L, D)xM
            outputs = layer(norm_inputs, **kwargs)

            # 2. Residual Connection: Add outputs to original inputs
            inputs = [inputs[j] + outputs[j] for j in range(self.num_modalities)]
        
        if self.final_pooling == "last":
            # Return the last time step if final_pooling is None
            last_elements = []
            for i, modality in enumerate(inputs):
                # extract the last useful time step
                last_idx = (mask.sum(dim=1)-1).long()
                last_true_element = modality[modality.shape[0]-1, last_idx, :] # (B, D)
                last_elements.append(last_true_element)
            concat = torch.cat(last_elements, dim=-1)
        elif self.final_pooling == "avg": # Used in SleepFM
            # Pooling over the length for each modality to obtain (B,D)xM
            pooled_modalities = []
            for i, modality in enumerate(inputs):
                # Masked Mean Pooling
                sum_pooled = (modality * mask.unsqueeze(-1)).sum(dim=1)
                count = mask.sum(dim=1).clamp(min=1e-9)
                pooled_modalities.append(sum_pooled / count.unsqueeze(-1))
            # Concat the modalities (B, DxM)
            concat = torch.cat(pooled_modalities, dim=-1)
        elif self.final_pooling == "max":
            # Pooling over the length for each modality to obtain (B,D)xM
            pooled_modalities = []
            for i, modality in enumerate(inputs):
                # Masked Max Pooling
                max_pooled = (modality * mask.unsqueeze(-1)).max(dim=1).values
                pooled_modalities.append(max_pooled)
            # Concat the modalities (B, DxM)
            concat = torch.cat(pooled_modalities, dim=-1)
        else:
            raise ValueError(f"Unknown final_pooling: {self.final_pooling}")

        # Apply final norm
        concat = self.final_norm(concat)

        return concat