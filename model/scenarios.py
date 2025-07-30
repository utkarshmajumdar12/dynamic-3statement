import numpy as np
import pandas as pd
from scipy.stats import triang, norm
from tqdm import tqdm

from model.financial_model import run_model, read_assumptions, push_df

def run_single_scenario(growth, margin, tag):
    assump = read_assumptions()
    assump.loc["Sales Growth %"] = growth
    assump.loc["EBIT Margin %"]  = margin
    push_df(assump.reset_index(name="Value"), "Assumptions")
    is_p, bs_p, cf_p = run_model(push_to_sheets=False)
    return pd.Series({
        "Scenario": tag,
        "NPV_FCF": (cf_p["Free Cash Flow"] / (1 + 0.1) ** cf_p.index).sum(),
        "Terminal Cash": cf_p["Ending Cash"].iloc[-1]
    })

def tornado():
    base = read_assumptions()
    print("Base assumptions:", base)
    drivers = {
        "Sales Growth %": np.linspace(base["Sales Growth %"] - 5, base["Sales Growth %"] + 5, 5),
        "EBIT Margin %": np.linspace(base["EBIT Margin %"] - 3, base["EBIT Margin %"] + 3, 5)
    }
    print("Drivers and their test values:", drivers)
    out = []
    for k, vals in drivers.items():
        for v in vals:
            print(f"Running scenario for {k}={v}")
            mod = base.copy()
            mod[k] = v
            push_df(mod.reset_index(name="Value"), "Assumptions")
            is_p, bs_p, cf_p = run_model(False)
            npv = (cf_p["Free Cash Flow"] / (1 + 0.1) ** cf_p.index).sum()
            print(f"NPV_FCF = {npv}")
            out.append({"Driver": k, "Value": v, "NPV_FCF": npv})
    df = pd.DataFrame(out)
    push_df(df, "Assumptions", start_cell="A15")
    return df


def monte_carlo(n_iter=5000):
    base = read_assumptions()
    tri_growth = triang(c=0.5, loc=base["Sales Growth %"] - 5, scale=10)
    norm_margin = norm(base["EBIT Margin %"], 2)
    results = []
    for i in tqdm(range(n_iter), desc="Running sims"):
        g = tri_growth.rvs()
        m = norm_margin.rvs()
        res = run_single_scenario(g, m, f"MC_{i}")
        results.append(res)
    df = pd.DataFrame(results)
    push_df(df, "Assumptions", start_cell="K1")
    return df
