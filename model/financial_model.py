import pandas as pd
import numpy as np
import functools
import requests

from model.constants import FMP_API, TICKER, FMP_KEY, YEARS_FORECAST
from model.sheets_io import pull_df, push_df

@functools.cache
def get_hist_financials():
    """
    Fetch the last YEARS_FORECAST annual Income Statement,
    Balance Sheet, and Cash Flow Statement separately.
    """
    # 1) Income Statement
    url_is = (
        f"{FMP_API}/income-statement-as-reported/"
        f"{TICKER}"
        f"?period=annual&limit={YEARS_FORECAST}&apikey={FMP_KEY}"
    )
    resp_is = requests.get(url_is, timeout=30)
    resp_is.raise_for_status()
    df_is = pd.DataFrame(resp_is.json())

    # 2) Balance Sheet
    url_bs = (
        f"{FMP_API}/balance-sheet-statement-as-reported/"
        f"{TICKER}"
        f"?period=annual&limit={YEARS_FORECAST}&apikey={FMP_KEY}"
    )
    resp_bs = requests.get(url_bs, timeout=30)
    resp_bs.raise_for_status()
    df_bs = pd.DataFrame(resp_bs.json())

    # 3) Cash Flow Statement
    url_cf = (
        f"{FMP_API}/cash-flow-statement-as-reported/"
        f"{TICKER}"
        f"?period=annual&limit={YEARS_FORECAST}&apikey={FMP_KEY}"
    )
    resp_cf = requests.get(url_cf, timeout=30)
    resp_cf.raise_for_status()
    df_cf = pd.DataFrame(resp_cf.json())

    return df_is, df_bs, df_cf


def read_assumptions():
    """
    Reads driver assumptions from the 'Assumptions' sheet tab.
    """
    df = pull_df("Assumptions")
    df.set_index("Metric", inplace=True)
    return df["Value"].astype(float)


def _find_first_field(df: pd.DataFrame, candidates: list[str]) -> str:
    """
    Utility to find the first matching field name in df.columns from candidates (case-insensitive).
    Raises KeyError if none found.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        candidate_lower = candidate.lower()
        if candidate_lower in cols_lower:
            return cols_lower[candidate_lower]
    raise KeyError(
        f"None of the expected fields {candidates} found in DataFrame columns. "
        f"Available columns: {list(df.columns)}"
    )


def project_income_statement(hist_is: pd.DataFrame, assumptions: pd.Series):
    revenue_candidates = [
        "revenue",
        "totalrevenue",
        "revenuefromcontractwithcustomerexcludingassessedtax",
        "revenues"
    ]
    net_income_candidates = [
        "netincome",
        "netincomeloss",
        "netincomeavailabletoparent",
        "netprofit",
        "netearnings"
    ]

    rev_field = _find_first_field(hist_is, revenue_candidates)
    net_income_field = _find_first_field(hist_is, net_income_candidates)

    years = np.arange(1, YEARS_FORECAST + 1)
    growth = assumptions["Sales Growth %"] / 100.0
    margin = assumptions["EBIT Margin %"] / 100.0
    tax_rate = assumptions["Tax Rate %"] / 100.0

    rev0 = hist_is.iloc[0][rev_field]
    revenue = rev0 * (1 + growth) ** years
    ebit = revenue * margin
    ebt = ebit
    taxes = -ebt * tax_rate
    net_income = ebt + taxes

    df = pd.DataFrame({
        "Year": years,
        "Revenue": revenue,
        "EBIT": ebit,
        "EBT": ebt,
        "Taxes": taxes,
        "Net Income": net_income
    }).set_index("Year")
    return df


def project_balance_sheet(hist_bs: pd.DataFrame,
                          is_proj: pd.DataFrame,
                          assumptions: pd.Series):
    cash_candidates = ["cashandcashequivalentsatcarryingvalue", "cashandcashequivalents", "cash"]
    ppe_candidates = ["propertyplantandequipmentnet", "propertyplantandequipment"]
    debt_candidates = [
        "longtermdebtnoncurrent",
        "longtermdebtcurrent",
        "totaldebt",
        "debt",
        "longtermdebt",
        "shorttermdebt"
    ]
    equity_candidates = ["stockholdersequity", "totalstockholdersequity", "totalequity"]

    cash_field = _find_first_field(hist_bs, cash_candidates)
    ppe_field = _find_first_field(hist_bs, ppe_candidates)
    try:
        debt_field = _find_first_field(hist_bs, debt_candidates)
    except KeyError:
        debt_field = None
    try:
        equity_field = _find_first_field(hist_bs, equity_candidates)
    except KeyError:
        equity_field = None

    years = is_proj.index
    df = pd.DataFrame(index=years)

    # Use assumptions based projections rather than raw historical values for dynamic forecasting
    df["Cash"] = assumptions["Cash % of Sales"] / 100.0 * is_proj["Revenue"]
    df["PP&E"] = assumptions["Capex % of Sales"] / 100.0 * is_proj["Revenue"]
    df["Debt"] = assumptions["Target Debt %"] / 100.0 * df["PP&E"]
    df["Equity"] = np.cumsum(is_proj["Net Income"])
    df["Total Assets"] = df["Cash"] + df["PP&E"]
    df["Total Liab+Eq"] = df["Debt"] + df["Equity"]

    return df


def project_cash_flow(is_proj: pd.DataFrame,
                      bs_proj: pd.DataFrame,
                      assumptions: pd.Series):
    years = is_proj.index
    df = pd.DataFrame(index=years)
    df["Net Income"] = is_proj["Net Income"]
    df["CapEx"] = -assumptions["Capex % of Sales"] / 100.0 * is_proj["Revenue"]
    df["Change in Debt"] = bs_proj["Debt"].diff().fillna(bs_proj["Debt"])
    df["Free Cash Flow"] = df.sum(axis=1)
    df["Ending Cash"] = bs_proj["Cash"]
    return df


def run_model(push_to_sheets: bool = True):
    hist_is, hist_bs, hist_cf = get_hist_financials()
    assumptions = read_assumptions()

    is_proj = project_income_statement(hist_is, assumptions)
    bs_proj = project_balance_sheet(hist_bs, is_proj, assumptions)
    cf_proj = project_cash_flow(is_proj, bs_proj, assumptions)

    if push_to_sheets:
        push_df(is_proj.reset_index(), "Income Statement")
        push_df(bs_proj.reset_index(), "Balance Sheet")
        push_df(cf_proj.reset_index(), "Cash Flow Statement")

    return is_proj, bs_proj, cf_proj


if __name__ == "__main__":
    run_model()
