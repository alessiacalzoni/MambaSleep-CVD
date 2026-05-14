import json, random, numpy as np, pandas as pd
import torch, torch.nn as nn
import sys
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..'))
from scripts.sources.utils.utils import extract_csv

def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
seed_all()

def from_path_to_ids(paths):
    return [int(Path(path).stem.split('-')[1]) for path in paths]
# ── CoxPH loss (from SleepFM) ────────────────────────────────
def cox_ph_loss(hazards, event_times, is_event):
    event_times, idx = event_times.sort(dim=0, descending=True)
    hazards  = hazards.gather(0, idx)
    is_event = is_event.gather(0, idx)
    
    valid_mask = is_event >= 0
    masked_hazards = hazards.masked_fill(~valid_mask, -1e9)
    
    log_cum  = torch.logcumsumexp(masked_hazards.float(), dim=0)
    clean_is_event = is_event.masked_fill(~valid_mask, 0)
    
    losses   = -((hazards - log_cum) * clean_is_event)
    per_label = losses.sum(0) / (clean_is_event.sum(0) + 1e-9)
    return per_label.mean()

# ── C-index calculation ────────────────────────────────────────────────
def c_index_per_label(hazards, times, events):
    """Compute Harrell's C-index per outcome from numpy arrays."""
    cis = {}
    for i, o in enumerate(outcome_vars):
        mask = (times[:, i] > 0) & ~np.isnan(times[:, i]) & (events[:, i] != -1)
        if events[mask, i].sum() < 5:
            cis[o] = None; continue
        cis[o] = concordance_index(times[mask, i], -hazards[mask, i], events[mask, i])
    return cis

def c_index_by_age_group(hazards, times, events, ages, percentiles=[25, 50, 75]):
    """Compute C-index per outcome stratified by age groups.

    Age thresholds are derived from the percentiles of subjects who
    experienced *any* event, following the evaluate_model.py convention.
    """
    # Determine which subjects had at least one event
    has_event = (events == 1).any(axis=1)
    event_ages = ages[has_event]
    # thresholds = np.percentile(event_ages, percentiles).tolist()
    thresholds = [68, 74, 78]

    age_min = float(np.min(ages))
    age_max = float(np.max(ages))
    boundaries = [age_min] + sorted(thresholds) + [age_max + 1]  # +1 so last bin is inclusive

    results = {}  # {group_label: {outcome: c_index | None}}
    for bin_idx in range(len(boundaries) - 1):
        lo, hi = boundaries[bin_idx], boundaries[bin_idx + 1]
        if bin_idx == len(boundaries) - 2:  # last bin inclusive
            mask = (ages >= lo) & (ages <= hi)
        else:
            mask = (ages >= lo) & (ages < hi)
        group_label = f"age_{lo:.0f}_{hi:.0f}"
        n = int(mask.sum())
        
        effective_n = {}
        for i, o in enumerate(outcome_vars):
            if n > 0:
                effective_n[o] = int((events[mask, i] != -1).sum())
            else:
                effective_n[o] = 0
                
        if n == 0:
            results[group_label] = {"n": 0, "effective_n": effective_n, "cis": {o: None for o in outcome_vars}}
            continue
        cis = c_index_per_label(hazards[mask], times[mask], events[mask])
        results[group_label] = {"n": n, "effective_n": effective_n, "cis": cis}
    return results, thresholds

# ── MLP model ─────────────────────────────────────────────────
class DemoMLP(nn.Module):
    def __init__(self, embed_dim, num_classes, dropout=0.3):
        super(DemoMLP, self).__init__()

        self.demo_embedding = nn.Sequential(
            nn.Linear(2, embed_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.disease_heads = nn.Linear(embed_dim // 4, num_classes)

    def forward(self, demo_features):
        demo_embed = self.demo_embedding(demo_features)
        hazards = self.disease_heads(demo_embed)
        return hazards

# ── MLP model MIO ─────────────────────────────────────────────────
class DemoMLP_m(nn.Module):
    def __init__(self, embed_dim, num_classes, dropout=0.3):
        super(DemoMLP_m, self).__init__()

        self.demo_mlp = nn.Sequential(
            nn.LazyLinear(embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.LayerNorm(embed_dim // 4),
            nn.ReLU()
        )
        self.disease_heads = nn.Linear(embed_dim // 4, num_classes)

    def forward(self, demo_features):
        demo_embed = self.demo_mlp(demo_features)
        hazards = self.disease_heads(demo_embed)
        return hazards

# ── Data helpers ──────────────────────────────────────────────
def to_tensors(df):
    X = torch.tensor(df[demo_vars].astype(float).values, dtype=torch.float32)
    E = torch.tensor(df[outcome_vars].astype(float).values,   dtype=torch.float32)
    T = torch.tensor(df[[f"{o}_time" for o in outcome_vars]].astype(float).values, dtype=torch.float32)
    return X, E, T

if __name__ == "__main__":
    outcome_vars = ["angina", "cvd_death", "chf", "chd_death", "mi", "stroke"]
    demo_vars = ['gender', 'age_s']

    parser = argparse.ArgumentParser()
    parser.add_argument("--outcome_csv_file", type=str)
    parser.add_argument("--demo_csv_file", type=str)
    parser.add_argument("--split_path", type=str, )
    
    df = extract_csv(original_csv_file = outcome_csv_file, target_outcome = outcome_vars, composite_outcome = False, threshold_time = None, imputing_method = "mask_events", demographic_csv_file = demo_csv_file, demographic_stratification_var = demo_vars)
    split = json.load(open(split_path, 'r'))

    if isinstance(split[list(split.keys())[0]][0], str):
        for key in split.keys():
            split[key] = from_path_to_ids(split[key])
        json.dump(split, open(split_path.replace(".json", ".json"), 'w'), indent=4)

    train = df[df["id"].isin(split["train"])].copy()
    val   = df[df["id"].isin(split["val"])].copy()
    test  = df[df["id"].isin(split["test"])].copy()

    scaler = StandardScaler()
    train[["age_s"]] = scaler.fit_transform(train[["age_s"]])
    val[["age_s"]]   = scaler.transform(val[["age_s"]])
    test[["age_s"]]  = scaler.transform(test[["age_s"]])
    # train[["bmi_s"]] = scaler.fit_transform(train[["bmi_s"]])
    # val[["bmi_s"]]   = scaler.transform(val[["bmi_s"]])
    # test[["bmi_s"]]  = scaler.transform(test[["bmi_s"]])
    
    X_train, E_train, T_train = to_tensors(train)
    X_val,   E_val,   T_val   = to_tensors(val) 
    X_test,  E_test,  T_test  = to_tensors(test)

    lr = 0.001
    batch_size = 256
    n_epochs = 10
    embed_dim= 128
    num_classes = len(outcome_vars)
    seed_all()

    model = DemoMLP(embed_dim, num_classes)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(X_train.size(0))
        epoch_loss = 0.0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i+batch_size]
            h = model(X_train[idx])
            loss = cox_ph_loss(h, T_train[idx], E_train[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_hazards = model(X_val)
            val_loss = cox_ph_loss(val_hazards, T_val, E_val)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "demo_model.pth")

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {epoch_loss/len(perm):.4f}, Val Loss: {val_loss.item():.4f}")
    
    print(f"Best Val Loss: {best_val_loss:.4f}")
    model.load_state_dict(torch.load("demo_model.pth"))
    model.eval()
    with torch.no_grad():
        test_hazards = model(X_test)
        test_loss = cox_ph_loss(test_hazards, T_test, E_test)
    print(f"Test Loss: {test_loss.item():.4f}")
    test_hazards_np = test_hazards.numpy()
    T_test_np = T_test.numpy()
    E_test_np = E_test.numpy()
    test_ages = df.loc[test.index, "age_s"].values

    test_c_index = c_index_per_label(test_hazards_np, T_test_np, E_test_np)
    print(f"Test C-Index: {test_c_index}")

    # ── Bootstrap helper ──────────────────────────────────────────
    def bootstrap_c_index(hazards, times, events, n_bootstraps=100):
        """Run bootstrapped c_index_per_label and return per-outcome stats."""
        n = len(hazards)
        boot = {o: [] for o in outcome_vars}
        macro_boot = []
        for _ in range(n_bootstraps):
            idx = np.random.choice(n, size=n, replace=True)
            cis = c_index_per_label(hazards[idx], times[idx], events[idx])
            macro_vals = []
            for o in outcome_vars:
                if cis.get(o) is not None:
                    boot[o].append(cis[o])
                    macro_vals.append(cis[o])
            if macro_vals:
                macro_boot.append(np.mean(macro_vals))
        stats = {}
        for o in outcome_vars:
            if boot[o]:
                vals = np.array(boot[o])
                stats[o] = {
                    "c_index": float(np.mean(vals)),
                    "c_index_mean": float(np.mean(vals)),
                    "c_index_std": float(np.std(vals)),
                    "c_index_ci_lower": float(np.percentile(vals, 2.5)),
                    "c_index_ci_upper": float(np.percentile(vals, 97.5)),
                }
            else:
                stats[o] = {"c_index": None, "c_index_mean": None,
                            "c_index_std": None, "c_index_ci_lower": None,
                            "c_index_ci_upper": None}
        
        macro_stats = {}
        if macro_boot:
            vals = np.array(macro_boot)
            macro_stats = {
                "c_index": float(np.mean(vals)),
                "c_index_mean": float(np.mean(vals)),
                "c_index_std": float(np.std(vals)),
                "c_index_ci_lower": float(np.percentile(vals, 2.5)),
                "c_index_ci_upper": float(np.percentile(vals, 97.5)),
            }
        else:
            macro_stats = {"c_index": None, "c_index_mean": None,
                            "c_index_std": None, "c_index_ci_lower": None,
                            "c_index_ci_upper": None}
                            
        return stats, macro_stats

    # ── Global bootstrapped C-Index ───────────────────────────────
    n_bootstraps = 100
    print(f"\nComputing {n_bootstraps} bootstraps for C-Index...")
    global_boot, global_macro = bootstrap_c_index(test_hazards_np, T_test_np, E_test_np, n_bootstraps)

    print(f"Test C-Index ({n_bootstraps} Bootstraps):")
    for o in outcome_vars:
        s = global_boot[o]
        if s["c_index_mean"] is not None:
            print(f"  {o}: {s['c_index_mean']:.4f} ± {s['c_index_std']:.4f} "
                  f"(95% CI: {s['c_index_ci_lower']:.4f} - {s['c_index_ci_upper']:.4f})")
        else:
            print(f"  {o}: N/A")
            
    if global_macro["c_index_mean"] is not None:
        print(f"  MACRO: {global_macro['c_index_mean']:.4f} ± {global_macro['c_index_std']:.4f} "
              f"(95% CI: {global_macro['c_index_ci_lower']:.4f} - {global_macro['c_index_ci_upper']:.4f})")

    # ── Age-group C-Index with bootstrapping ──────────────────────
    print("\nComputing C-Index by Age Group (with bootstrapping)...")
    age_results, age_thresholds = c_index_by_age_group(test_hazards_np, T_test_np, E_test_np, test_ages)
    print(f"Age thresholds: {age_thresholds}")

    age_group_json = {}
    for grp, res in age_results.items():
        n_grp = res["n"]
        effective_n = res.get("effective_n", {})
        print(f"\nGroup {grp} (Total n={n_grp}):")
        if n_grp == 0:
            age_group_json[grp] = {
                "num_samples": 0,
                "effective_samples": effective_n,
                "per_label_metrics": {o: {"c_index": None} for o in outcome_vars},
                "macro_averaged_metrics": {"c_index": None, "c_index_mean": None, "c_index_ci_lower": None, "c_index_ci_upper": None}
            }
            continue

        # Recover mask to get subgroup arrays
        lo, hi = grp.replace("age_", "").split("_")
        lo, hi = float(lo), float(hi)
        if grp == list(age_results.keys())[-1]:  # last bin inclusive
            grp_mask = (test_ages >= lo) & (test_ages <= hi)
        else:
            grp_mask = (test_ages >= lo) & (test_ages < hi)

        grp_boot, grp_macro = bootstrap_c_index(
            test_hazards_np[grp_mask], T_test_np[grp_mask], E_test_np[grp_mask],
            n_bootstraps
        )

        age_group_json[grp] = {
            "num_samples": n_grp,
            "effective_samples": effective_n,
            "per_label_metrics": grp_boot,
            "macro_averaged_metrics": grp_macro
        }

        for o in outcome_vars:
            s = grp_boot[o]
            eff_n = effective_n.get(o, 0)
            if s["c_index_mean"] is not None:
                print(f"  {o} (eff_n={eff_n}): {s['c_index_mean']:.4f} ± {s['c_index_std']:.4f} "
                      f"(95% CI: {s['c_index_ci_lower']:.4f} - {s['c_index_ci_upper']:.4f})")
            else:
                print(f"  {o} (eff_n={eff_n}): N/A")
                
        if grp_macro["c_index_mean"] is not None:
            print(f"  MACRO: {grp_macro['c_index_mean']:.4f} ± {grp_macro['c_index_std']:.4f} "
                  f"(95% CI: {grp_macro['c_index_ci_lower']:.4f} - {grp_macro['c_index_ci_upper']:.4f})")

    # ── Build JSON compatible with degradation_performance_plot.py ─
    output_json = {
        "per_label_metrics": global_boot,
        "macro_averaged_metrics": global_macro,
        "age_group_metrics": age_group_json,
    }

    output_dir = os.getcwd()
    output_json_path = os.path.join(output_dir, "eval_test_demo_only_masked.json")
    with open(output_json_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"\nSaved evaluation JSON to {output_json_path}")
