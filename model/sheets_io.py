import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from model.constants import SERVICE_ACCOUNT_FILE, GOOGLE_SHEET_ID

_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

_creds = Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=_SCOPES)
_gc = gspread.authorize(_creds)
_sh = _gc.open_by_key(GOOGLE_SHEET_ID)

def pull_df(tab_name: str, rng: str = None) -> pd.DataFrame:
    ws = _sh.worksheet(tab_name)
    vals = ws.get(rng) if rng else ws.get_all_values()
    df = pd.DataFrame(vals[1:], columns=vals[0])

    # Convert columns that can be numeric to numeric
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            # Ignore conversion errors for non-numeric columns
            pass

    return df

def push_df(df: pd.DataFrame, tab_name: str,
            start_cell: str = "A1", clear: bool = True):
    ws = _sh.worksheet(tab_name)
    if clear:
        ws.clear()
    ws.update([df.columns.values.tolist()] + df.values.tolist(),
              start_cell, value_input_option="USER_ENTERED")
