from pathlib import Path
from io import BytesIO
from typing import Optional, Dict, List, Tuple, Set
import os
import json
import re
import sqlite3
from datetime import datetime
import unicodedata
from collections import defaultdict, Counter  # Counter is new (defaultdict already present)
import random  
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, FileResponse
from pydantic import BaseModel, Field

from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

# --- LLM setup -------------------------------------------------------------
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

# Load .env and override any existing env vars in this process
load_dotenv(find_dotenv(), override=True)

def get_openai_client():
    """
    Return (client, model). client is None if OPENAI_API_KEY isn't available.
    """
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    model = (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
    return (OpenAI(api_key=key), model) if key else (None, model)

# =============================== Configuration ============================
EXCEL_PATH = Path("Realistic_Ticket_Data_Updated.xlsx")
DB_PATH    = Path("kpi_dashboard.db")
TABLE      = "tickets"
TITLE      = "KPI Service (5 KPIs + Dashboard + LLM)"
EVAL_PATH = Path("eval_dataset.jsonl")

# Toggle verbose debug in /chat responses with env DEBUG_CHAT=1|true|yes|on
DEBUG_MODE = os.getenv("DEBUG_CHAT", "0").strip().lower() in {"1", "true", "yes", "on"}

# ---- Project write-up (NLP & fine-tuning) --------------------------------
NLP_NOTES = """
## NLP & Adaptation Notes
We use GPT-4o-mini strictly for structured intent parsing into {action, name, period, by, chart}.
Names are grounded against a dynamic registry built from built-in KPIs and DB-defined KPIs.
No parameter fine-tuning was required; we rely on prompt grounding and a runtime registry
(kpi_definitions, kpi_synonyms) for safe, auditable adaptation without redeploys.
If needed, utterance_logs can seed supervised fine-tuning later.
"""

# ============================== Data Utilities ============================
def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        pd.Index(df.columns)
        .astype(str).str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^0-9a-zA-Z_]+", "", regex=True)
        .str.lower()
    )
    return df

def _first(df: pd.DataFrame, *names: str) -> Optional[str]:
    for n in names:
        if n and n in df.columns:
            return n
    return None

def _ensure_periods(d: pd.DataFrame) -> pd.DataFrame:
    c = "created_on_norm"
    if c not in d.columns:
        d[c] = pd.NaT
    d[c] = pd.to_datetime(d[c], errors="coerce")
    d["week"]     = d[c].dt.to_period("W").astype("string")
    d["biweekly"] = d[c].dt.to_period("2W").astype("string")  # NEW
    d["month"]    = d[c].dt.to_period("M").astype("string")
    d["quarter"]  = d[c].dt.to_period("Q").astype("string")
    return d

def _prepare_sqlite() -> None:
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(
            f"Excel not found: {EXCEL_PATH.resolve()} "
            f"(put your file next to app.py or change EXCEL_PATH)."
        )
    try:
        df = pd.read_excel(EXCEL_PATH, sheet_name=0)
    except Exception:
        df = pd.read_excel(EXCEL_PATH)

    df = _clean_columns(df)

    col_created  = _first(df, "created_on","created_at","ticket_creation_date","ticket_created_on","created","date")
    col_start    = _first(df, "start_of_the_uptime","start_of_uptime","start_time","start")
    col_final    = _first(df, "final_commissioning_date","final_commissioning","resolved_on","closed_on","end_time","end")
    col_status   = _first(df, "status","ticket_status")
    col_priority = _first(df, "priority","ticket_priority")
    col_asset    = _first(df, "asset_name","asset","machine","equipment")
    col_location = _first(df, "location","site","branch")
    col_assignee = _first(df, "assignee_username","assignee","technician")
    col_return   = _first(df, "is_returned_ticket","returned_ticket","is_repeat","repeat_ticket","returned_flag")
    col_sat_in   = _first(df, "sat","satisfaction_score","csat","customer_satisfaction","customer_satisfaction_score")

    for c in [col_created, col_start, col_final]:
        if c:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    df["created_on_norm"] = pd.to_datetime(df[col_created], errors="coerce") if col_created else pd.NaT
    df = _ensure_periods(df)

    if col_start and col_final:
        df["mttr"] = (df[col_final] - df[col_start]).dt.total_seconds() / 60.0
    else:
        df["mttr"] = pd.NA

    def _sat(x):
        try:
            v = float(x)
            if 1 <= v <= 5: return v
            if 0 <= v <= 100: return 1 + 4*(v/100.0)
        except Exception:
            return pd.NA
    df["sat"] = df[col_sat_in].map(_sat) if col_sat_in else pd.NA

    if col_return:
        col = df[col_return].astype(str).str.strip().str.lower()
        mapping = {
            "yes":1,"y":1,"true":1,"1":1,"repeat":1,"returned":1,
            "no":0,"n":0,"false":0,"0":0,"first time":0,"first-time":0,"firsttime":0
        }
        df["returned_flag"] = pd.to_numeric(col.map(mapping), errors="coerce")
    else:
        df["returned_flag"] = pd.NA

    with sqlite3.connect(DB_PATH) as con:
        df.to_sql(TABLE, con, if_exists="replace", index=False)

def _load_df() -> pd.DataFrame:
    if not DB_PATH.exists():
        _prepare_sqlite()
    try:
        with sqlite3.connect(DB_PATH) as con:
            d = pd.read_sql(f'SELECT * FROM "{TABLE}"', con)
    except Exception:
        # DB exists but table missing/corrupt → rebuild from Excel once
        _prepare_sqlite()
        with sqlite3.connect(DB_PATH) as con:
            d = pd.read_sql(f'SELECT * FROM "{TABLE}"', con)

    if "created_on_norm" in d.columns:
        d["created_on_norm"] = pd.to_datetime(d["created_on_norm"], errors="coerce")
    d = _ensure_periods(d)
    return d

# ========================= Learning/Adaptation Layer ======================
REGISTRY = {
    "funcs": {},      # built-in KPI functions (set after KPI_FUNCS defined)
    "defs": {},       # DB-defined KPIs {name -> row}
    "synonyms": {},   # {term -> kpi_name}
    "loaded_at": None,
    "fewshots": []    
}

def _ensure_learning_tables() -> None:
    if not DB_PATH.exists():
        _prepare_sqlite()
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS kpi_definitions(
            name TEXT PRIMARY KEY,
            sql TEXT NOT NULL,
            chart_hint TEXT,
            description TEXT,
            enabled INTEGER DEFAULT 1,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS kpi_synonyms(
            term TEXT PRIMARY KEY,
            kpi_name TEXT NOT NULL,
            weight REAL DEFAULT 1.0
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS utterance_logs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            parsed_json TEXT,
            resolved_kpi TEXT,
            success INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS feedback(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            utterance_id INTEGER,
            label TEXT,
            note TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        con.commit()

def load_registry(force: bool=False) -> None:
    _ensure_learning_tables()
    with sqlite3.connect(DB_PATH) as con:
        defs = pd.read_sql('SELECT * FROM kpi_definitions WHERE enabled=1', con)
        syns = pd.read_sql('SELECT * FROM kpi_synonyms', con)
    REGISTRY["defs"] = {r["name"]: r for _, r in defs.iterrows()} if not defs.empty else {}
    REGISTRY["synonyms"] = {r["term"].lower(): r["kpi_name"] for _, r in syns.iterrows()} if not syns.empty else {}
    REGISTRY["loaded_at"] = datetime.utcnow().isoformat()
    REGISTRY["fewshots"] = []  # NEW: clear dynamic few-shots on reload

def resolve_kpi_name(tokens: List[str]) -> Optional[str]:
    names = set(REGISTRY["defs"].keys()) | set(REGISTRY["funcs"].keys()) | {"dashboard"}

    # 1) exact match
    for t in tokens:
        if t in names:
            return t

    # 2) synonyms table
    for t in tokens:
        if t in REGISTRY["synonyms"]:
            return REGISTRY["synonyms"][t]

    # 3) heuristics
    if "mttr" in tokens:
        return "mttr"
    if any(t in tokens for t in ("ticket", "tickets", "volume", "count", "loads")):
        return "ticket_volume"
    if any(t in tokens for t in ("csat", "satisfaction", "satisfied", "sat")):
        return "satisfaction_avg"
    if "ftfr" in tokens or ("first" in tokens and "time" in tokens and "fix" in tokens):
        return "ftfr"
    if "status" in tokens or "statuses" in tokens:
        return "status_ratio"
    if ("top" in tokens) and any(t in tokens for t in ("location", "locations", "site", "branch")):
        return "top_locations"
    if "dashboard" in tokens:
        return "dashboard"
    return None

def log_utterance(text: str, parsed: dict, resolved_kpi: Optional[str], success: bool) -> None:
    try:
        with sqlite3.connect(DB_PATH) as con:
            con.execute("""INSERT INTO utterance_logs(text, parsed_json, resolved_kpi, success)
                           VALUES (?, ?, ?, ?)""",
                        (text, json.dumps(parsed or {}), resolved_kpi, int(bool(success))))
            con.commit()
    except Exception:
        pass

# ============================== Chart Utilities ===========================
# ============================== Chart Utilities ===========================
def _png(fig) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

# ---- axis helpers --------------------------------------------------------
import math

_PERIOD_RE_MONTH   = re.compile(r"^\d{4}-\d{2}$")        # 2024-01
_PERIOD_RE_QUARTER = re.compile(r"^\d{4}Q[1-4]$")        # 2024Q1
_PERIOD_RE_RANGE   = re.compile(r"^\d{4}-\d{2}-\d{2}/\d{4}-\d{2}-\d{2}$")  # week/biweekly range

def _short_label(s: str) -> str:
    s = str(s)
    if _PERIOD_RE_MONTH.match(s):
        dt = pd.to_datetime(s + "-01", errors="coerce")
        if pd.notna(dt): return dt.strftime("%b %Y")     # Jan 2024
    if _PERIOD_RE_QUARTER.match(s):
        y, q = s.split("Q")
        return f"Q{q} {y}"                               # Q1 2024
    if _PERIOD_RE_RANGE.match(s) and "/" in s:
        start = s.split("/")[0]
        dt = pd.to_datetime(start, errors="coerce")
        if pd.notna(dt): return dt.strftime("W%U %Y")    # W05 2024
    return s

def _smart_xtick_positions(n: int, max_ticks: int = 12) -> List[int]:
    if n <= max_ticks:
        return list(range(n))
    step = math.ceil(n / max_ticks)
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return idx

def _apply_smart_xticks(ax, labels: List[str], max_ticks: int = 12, rotation: int = 30):
    n = len(labels)
    idx = _smart_xtick_positions(n, max_ticks)
    lab = [_short_label(labels[i]) for i in idx]
    ax.set_xticks(idx)
    ax.set_xticklabels(lab, rotation=rotation, ha="right")

def _sorted_table(index_list: List, values_list: List):
    """
    Sort table rows by the first component, recognizing:
    - 'YYYY-MM' (month)
    - 'YYYYQ#'  (quarter)
    - 'YYYY-MM-DD/YYYY-MM-DD' (week/biweekly range)
    Falls back to string sort for non-period labels.
    """
    items = list(zip(index_list, values_list))

    def _key(label):
        head = label[0] if isinstance(label, (list, tuple)) and label else label
        s = str(head)

        if _PERIOD_RE_MONTH.match(s):
            dt = pd.to_datetime(s + "-01", errors="coerce")
            return (0, dt if pd.notna(dt) else s, str(label))

        if _PERIOD_RE_QUARTER.match(s):
            y, q = s.split("Q")
            return (0, int(y) * 4 + int(q), str(label))

        if _PERIOD_RE_RANGE.match(s) and "/" in s:
            start = s.split("/")[0]
            dt = pd.to_datetime(start, errors="coerce")
            return (0, dt if pd.notna(dt) else start, str(label))

        return (1, str(label))

    items.sort(key=lambda iv: _key(iv[0]))
    idx_sorted  = [i for i, _ in items]
    vals_sorted = [v for _, v in items]
    return idx_sorted, vals_sorted

# ---- plotting ------------------------------------------------------------
def plot_series(labels, values, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(values))
    ax.plot(x, values, marker="o", linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _apply_smart_xticks(ax, list(labels), max_ticks=12, rotation=30)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _png(fig)

def plot_bar(labels, values, title, xlabel="Category", ylabel="Value"):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(values))
    ax.bar(x, values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _apply_smart_xticks(ax, list(labels), max_ticks=16, rotation=30)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _png(fig)

def plot_pie(labels, values, title):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(values, labels=[_short_label(l) for l in labels], autopct="%1.0f%%")
    ax.set_title(title)
    fig.tight_layout()
    return _png(fig)

# ============================== KPI Helpers ===============================
def _err(msg="Data not available.") -> Dict:
    return {"type":"error","message":msg}

def _period_col(period: Optional[str]) -> Optional[str]:
    if not period: return None
    return {"week":"week","biweekly":"biweekly","month":"month","quarter":"quarter"}.get(period.lower())

def _by_col(by: Optional[str], dref: pd.DataFrame) -> Optional[str]:
    if not by: return None
    by = by.lower()
    mapping = {
        "priority": "priority" if "priority" in dref.columns else ("ticket_priority" if "ticket_priority" in dref.columns else None),
        "asset": next((c for c in ["asset_name","asset","machine","equipment"] if c in dref.columns), None),
        "location": next((c for c in ["location","site","branch"] if c in dref.columns), None),
        "assignee": next((c for c in ["assignee_username","assignee","technician"] if c in dref.columns), None),
        "status": next((c for c in ["status","ticket_status"] if c in dref.columns), None),
    }
    return mapping.get(by, None)

# ============================== Core KPIs (5) =============================
def kpi_mttr(d: pd.DataFrame, period: Optional[str]=None, by: Optional[str]=None) -> Dict:
    mttr_col = "mttr" if "mttr" in d.columns else None
    if not mttr_col: return _err()
    groupers: List[str] = []
    pcol = _period_col(period); bcol = _by_col(by, d)
    if pcol: groupers.append(pcol)
    if bcol: groupers.append(bcol)
    if groupers:
        s = d.groupby(groupers)[mttr_col].mean(numeric_only=True).dropna()
        if s.empty: return _err()
        idx = s.index if isinstance(s.index, pd.MultiIndex) else s.index.tolist()
        idx = [i if isinstance(i,str) else list(i) for i in (idx.tolist() if hasattr(idx,"tolist") else idx)]
        return {"type":"table","title":"MTTR (min)","index":idx,"values":[float(v) for v in s.values]}
    vals = pd.to_numeric(d[mttr_col], errors="coerce").dropna()
    if vals.empty: return _err()
    return {"type":"metric","title":"MTTR (min)","value":float(vals.mean())}

def kpi_top_locations(d: pd.DataFrame, period: Optional[str]=None, by: Optional[str]=None, top_n: int = 10) -> Dict:
    col = _by_col("location", d)
    if not col:
        return _err()
    s = d[col].astype(str)
    s = s[~s.str.lower().isin({"", "nan", "none"})]
    if s.empty: return _err()
    counts = s.value_counts().head(top_n)
    if counts.empty: return _err()
    labels = counts.index.tolist()
    values = [int(v) for v in counts.values]
    return {"type": "bar", "title": f"Top {len(labels)} Locations (by volume)", "labels": labels, "values": values}

def kpi_top_locations_table(d: pd.DataFrame, period: Optional[str]=None, by: Optional[str]=None, top_n: int = 10) -> Dict:
    col = _by_col("location", d)
    if not col:
        return _err()
    s = d[col].astype(str)
    s = s[~s.str.lower().isin({"", "nan", "none"})]
    if s.empty: 
        return _err()
    counts = s.value_counts().head(top_n)
    if counts.empty: 
        return _err()
    labels = counts.index.tolist()
    values = [int(v) for v in counts.values]
    # table-shaped payload so the frontend can render a grid
    return {
        "type": "table",
        "title": f"Top {len(labels)} Locations (by volume)",
        "index": labels,      # row labels
        "values": values      # row values (ints)
    }

def kpi_ticket_volume(d: pd.DataFrame, period: Optional[str]=None, by: Optional[str]=None) -> Dict:
    groupers: List[str] = []
    pcol = _period_col(period); bcol = _by_col(by, d)
    if pcol: groupers.append(pcol)
    if bcol: groupers.append(bcol)
    if groupers:
        s = d.groupby(groupers).size()
        if s.empty: return _err()
        idx = s.index if isinstance(s.index, pd.MultiIndex) else s.index.tolist()
        idx = [i if isinstance(i, str) else list(i) for i in (idx.tolist() if hasattr(idx, "tolist") else idx)]
        vals = [int(v) for v in s.values]

        # NEW: keep table order aligned with chart axis
        idx, vals = _sorted_table(idx, vals)

        return {"type":"table","title":"Ticket Volume","index":idx,"values":vals}
    return {"type":"metric","title":"Ticket Volume","value":int(len(d))}

def kpi_satisfaction_avg(d: pd.DataFrame, period: Optional[str]=None, by: Optional[str]=None) -> Dict:
    sat = next((c for c in ["sat","satisfaction_score","csat","customer_satisfaction"] if c in d.columns), None)
    if not sat: return _err()
    groupers: List[str] = []
    pcol = _period_col(period); bcol = _by_col(by, d)
    if pcol: groupers.append(pcol)
    if bcol: groupers.append(bcol)
    series = pd.to_numeric(d[sat], errors="coerce")
    if groupers:
        s = series.groupby([d[g] for g in groupers]).mean().dropna()
        if s.empty: return _err()
        idx = s.index if isinstance(s.index, pd.MultiIndex) else s.index.tolist()
        idx = [i if isinstance(i,str) else list(i) for i in (idx.tolist() if hasattr(idx,"tolist") else idx)]
        return {"type":"table","title":"Avg Satisfaction (1-5)","index":idx,"values":[float(v) for v in s.values]}
    vals = series.dropna()
    if vals.empty: return _err()
    return {"type":"metric","title":"Avg Satisfaction (1-5)","value":float(vals.mean())}

def kpi_status_ratio(d: pd.DataFrame, period: Optional[str]=None, by: Optional[str]=None) -> Dict:
    col = _by_col("status", d)
    if not col: return _err()
    groupers: List[str] = []
    pcol = _period_col(period); bcol = _by_col(by, d)
    if pcol: groupers.append(pcol)
    if bcol and bcol != col: groupers.append(bcol)
    if groupers:
        s = d.groupby(groupers + [col]).size()
        if s.empty: return _err()
        pct = s.groupby(level=list(range(len(groupers)))).apply(lambda x: 100*x/x.sum()).round(1)
        rows = []
        for idx, val in pct.items():
            if not isinstance(idx, tuple): idx = (idx,)
            grp = idx[:-1]; status_val = idx[-1]
            rows.append({"group": list(grp) if len(grp)>1 else (grp[0] if grp else None),
                         "status": str(status_val), "pct": float(val)})
        return {"type":"ratio","title":"Status Ratio (%)","rows":rows}
    vc = d[col].value_counts(normalize=True)*100
    if vc.empty: return _err()
    return {"type":"bar","title":"Status Ratio (%)","labels":vc.index.astype(str).tolist(),
            "values":[float(round(v,1)) for v in vc.values]}

def kpi_ftfr(d: pd.DataFrame, period: Optional[str]=None, by: Optional[str]=None) -> Dict:
    if "returned_flag" not in d.columns: return _err()
    v = pd.to_numeric(d["returned_flag"], errors="coerce")
    groupers: List[str] = []
    pcol = _period_col(period); bcol = _by_col(by, d)
    if pcol: groupers.append(pcol)
    if bcol: groupers.append(bcol)
    if groupers:
        s = (1.0 - v.groupby([d[g] for g in groupers]).mean()).dropna()
        if s.empty: return _err()
        idx = s.index if isinstance(s.index, pd.MultiIndex) else s.index.tolist()
        idx = [i if isinstance(i,str) else list(i) for i in (idx.tolist() if hasattr(idx,"tolist") else idx)]
        return {"type":"table","title":"First Time Fix Rate","index":idx,"values":[float(v) for v in s.values]}
    v = v.dropna()
    if v.empty: return _err()
    return {"type":"metric","title":"FTFR","value":float(1.0 - v.mean())}

KPI_FUNCS = {
    "mttr": kpi_mttr,
    "ticket_volume": kpi_ticket_volume,
    "satisfaction_avg": kpi_satisfaction_avg,
    "status_ratio": kpi_status_ratio,
    "ftfr": kpi_ftfr,
    "top_locations": kpi_top_locations,
    "top_locations_table": kpi_top_locations_table,  # <-- NEW
}

# Allowed chart types per KPI (used to adjust/deny unsuitable requests)
ALLOWED_CHARTS: Dict[str, Set[str]] = {
    "mttr": {"bar", "line"},
    "satisfaction_avg": {"bar", "line"},
    "ticket_volume": {"bar", "line", "pie"},
    "status_ratio": {"pie", "bar"},
    "ftfr": {"bar", "line"},
    "top_locations": {"bar", "pie"},
    "top_locations_table": {"bar", "pie", "line"}, 
}

# set built-ins into registry and load DB-defined on import
REGISTRY["funcs"] = KPI_FUNCS.copy()
load_registry(force=True)

# ---- DB-defined KPI runner ----------------------------------------------
def kpi_from_sql(name: str) -> Dict:
    row = REGISTRY["defs"][name]
    sql = row.get("sql", "")
    if not sql.strip():
        return _err()
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql(sql, con)
    cols_lower = [c.lower() for c in df.columns]
    if "label" not in cols_lower or "value" not in cols_lower:
        return _err()
    label_col = df.columns[cols_lower.index("label")]
    value_col = df.columns[cols_lower.index("value")]
    labels = df[label_col].astype(str).tolist()
    values = pd.to_numeric(df[value_col], errors="coerce").fillna(0).astype(float).tolist()
    if not labels or not values:
        return _err()
    title = row.get("description") or name
    payload = {"type":"bar","title":title,"labels":labels,"values":values}
    if row.get("chart_hint") in ("bar","line","pie"):
        payload["_chart_hint"] = row["chart_hint"]
    return payload

# ============================ Chart Selection Logic =======================
def _preferred_chart_for(name: str, payload: Dict, period: Optional[str]) -> str:
    name = name.lower()
    ptype = payload.get("type")
    if name == "status_ratio":
        return "pie"
    if name in ("mttr", "satisfaction_avg") and (ptype in ("table", "bar")):
        return "line" if period else "bar"
    if name in ("top_locations", "ticket_volume", "ftfr"):
        return "bar"
    if payload.get("_chart_hint") in ("bar","line","pie"):
        return payload["_chart_hint"]
    return "bar"

def _validate_chart_for(name: str, requested: Optional[str], payload: Dict, period: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    Ensure requested chart is suitable; if not, pick a safe alternative and return a 'notice'.
    """
    allowed = ALLOWED_CHARTS.get(name.lower(), {"bar","line","pie"})
    if requested and requested in allowed:
        return requested, None
    # choose a sane fallback
    fallback = _preferred_chart_for(name, payload, period)
    if fallback not in allowed:
        # last resort
        fallback = "bar" if "bar" in allowed else next(iter(allowed))
    if requested and requested not in allowed:
        nice = requested.upper()
        nname = (payload.get("title") or name).split("(")[0].strip()
        return fallback, f"{nice} isn’t suitable for {nname}; showing {fallback.upper()} instead."
    return fallback, None

# ================================ FastAPI ================================
class ChatIn(BaseModel):
    message: str

# Admin payloads
class KpiDefIn(BaseModel):
    name: str = Field(..., pattern=r"^[a-z0-9_]+$")
    sql: str
    chart_hint: Optional[str] = Field(None, pattern="^(bar|line|pie)$")
    description: Optional[str] = None
    enabled: Optional[int] = 1

class SynIn(BaseModel):
    term: str
    kpi_name: str
    weight: Optional[float] = 1.0

class FeedbackIn(BaseModel):
    utterance_id: int
    label: str
    note: Optional[str] = None

app = FastAPI(title=TITLE)
@app.get("/eval/stats")
def eval_stats():
    # Summarize the eval dataset and show which KPIs are missing labels
    try:
        items = _load_eval_items(EVAL_PATH)
    except Exception as e:
        return JSONResponse(
            {"error": str(e), "file": str(EVAL_PATH.resolve())},
            status_code=400
        )

    by_name = defaultdict(int)
    for it in items:
        nm = ((it.get("gold") or {}).get("name") or "").lower()
        by_name[nm] += 1

    all_names = sorted(set(list(KPI_FUNCS.keys()) + list(REGISTRY["defs"].keys())))
    missing = [n for n in all_names if by_name.get(n, 0) == 0]

    return {
        "n_items": len(items),
        "by_name": dict(by_name),
        "missing": missing,
        "file": str(EVAL_PATH.resolve())
    }
# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)
# ---- Serve the frontend from /web ----
app.mount("/web", StaticFiles(directory="web", html=True), name="web")

@app.get("/")
def root():
    return RedirectResponse(url="/web/index.html")

# ---------- Health ----------
@app.get("/health")
def health():
    try:
        _ = _load_df()
        client, model = get_openai_client()
        return {"ok": True, "llm": bool(client), "model": model if client else None, "registry_loaded_at": REGISTRY["loaded_at"]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ---------- NLP notes ----------
@app.get("/nlp-notes")
def nlp_notes():
    return Response(content=NLP_NOTES, media_type="text/markdown")

# ---------- LLM sanity check ----------
@app.get("/test-llm")
def test_llm():
    client, model = get_openai_client()
    if not client:
        return {"ok": False, "error": "OPENAI_API_KEY not loaded"}
    try:
        r = client.responses.create(model=model, input="Reply OK")
        return {"ok": True, "reply": r.output_text, "model": model}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ---------- KPI JSON ----------
@app.get("/kpi/{name}")
def get_kpi(name: str,
            period: str = Query(None, pattern="^(week|biweekly|month|quarter)$"),
            by: str = Query(None, pattern="^(priority|asset|location|assignee|status)$")):
    name = name.lower()
    d = _load_df()
    if name in KPI_FUNCS:
        payload = KPI_FUNCS[name](d, period=period, by=by)
        return JSONResponse({"kpi": name, "period": period, "by": by, "payload": payload})
    if name in REGISTRY["defs"]:
        payload = kpi_from_sql(name)
        return JSONResponse({"kpi": name, "period": None, "by": None, "payload": payload})
    raise HTTPException(404, f"Unknown KPI '{name}'. See /categories or add via /admin/kpis.")

# --- axis label mapping for single charts -------------------------------
def _axis_labels_for(name: str, chart: Optional[str], payload: Dict) -> tuple[Optional[str], Optional[str]]:
    """
    Returns (x_label, y_label) that match the dashboard convention.
    For pie charts we return (None, None) because axes aren't shown.
    """
    n = (name or "").lower()
    if chart == "pie":
        return (None, None)

    if n in ("mttr", "mttr_min"):
        return ("Date", "Rate (min)")
    if n in ("ticket_volume", "tickets"):
        return ("Date", "Number of tickets")
    if n in ("satisfaction_avg", "avg_satisfaction"):
        return ("Date", "Score (1–5)")
    if n in ("ftfr", "first_time_fix_rate"):
        return ("Date", "Rate")
    if n.startswith("top_locations"):
        return ("City", "Number of tickets")
    if n in ("status_ratio", "status_mix"):
        return ("Status", "%")

    # sensible default
    return ("Index", "Value")

# ---------- KPI Chart (PNG) ----------
def render_chart_for_payload(name: str, payload: Dict, period: Optional[str], chart: Optional[str]):
    # Validate chart type first
    chart, _ = _validate_chart_for(name, chart, payload, period)

    title = payload.get("title", name.upper())
    ptype = payload.get("type")
    xlab, ylab = _axis_labels_for(name, chart, payload)

    if ptype == "metric":
        # single bar; use KPI-specific ylabel when we have it
        return plot_bar([title], [payload["value"]], title, xlabel="", ylabel=(ylab or "Value"))

    if ptype == "bar":
        labels, values = payload["labels"], payload["values"]
        if chart == "pie":
            return plot_pie(labels, values, title)
        if chart == "line":
            return plot_series(labels, values, title, (xlab or "Index"), (ylab or "Value"))
        return plot_bar(labels, values, title, xlabel=(xlab or "Category"), ylabel=(ylab or "Value"))

    if ptype == "table":
        labels = [lab if isinstance(lab, str) else " · ".join(map(str, lab)) for lab in payload["index"]]

        # Respect explicit chart first
        if chart == "pie" and len(labels) <= 12:
            return plot_pie(labels, payload["values"], title)
        if chart == "line":
            return plot_series(labels, payload["values"], title, (xlab or "Index"), (ylab or "Value"))
        if chart == "bar":
            return plot_bar(labels, payload["values"], title, xlabel=(xlab or "Category"), ylabel=(ylab or "Value"))

        # No explicit chart → choose by period
        if period is not None:
            return plot_series(labels, payload["values"], title, (xlab or "Index"), (ylab or "Value"))
        return plot_bar(labels, payload["values"], title, xlabel=(xlab or "Category"), ylabel=(ylab or "Value"))

    if ptype == "ratio":
        rows = payload.get("rows", [])
        if not rows:
            raise HTTPException(400, "No data to chart.")
        agg = defaultdict(list)
        for r in rows:
            agg[r["status"]].append(r["pct"])
        labels = list(agg.keys())
        values = [sum(v) / len(v) for v in agg.values()]
        if chart == "pie":
            return plot_pie(labels, values, title)
        # bar fallback → Y should be %
        return plot_bar(labels, values, title, xlabel=(xlab or "Status"), ylabel="%")

    raise HTTPException(400, "Payload not chartable.")

@app.get("/chart/{name}")
def chart(
    name: str,
    period: str = Query(None, pattern="^(week|biweekly|month|quarter)$"),
    by: str = Query(None, pattern="^(priority|asset|location|assignee|status)$"),
    chart: Optional[str] = None,
):

    name = name.lower()
    d = _load_df()
    if name in KPI_FUNCS:
        payload = KPI_FUNCS[name](d, period=period, by=by)
    elif name in REGISTRY["defs"]:
        payload = kpi_from_sql(name)
    else:
        raise HTTPException(404, f"Unknown KPI '{name}'.")
        # Normalize/sanitize chart input; only {bar,line,pie} reach the renderer
    if chart:
        chart = str(chart).strip().lower()
        if chart in {"bar chart", "bar graph"}:
            chart = "bar"
        elif chart in {"line chart", "line graph"}:
            chart = "line"
        elif chart in {"pie chart"}:
            chart = "pie"
        if chart not in {"bar", "line", "pie"}:
            chart = None
    if not chart:
        chart = _preferred_chart_for(name, payload, period)
    png = render_chart_for_payload(name, payload, period, chart)
    return Response(content=png, media_type="image/png")

# ---------- Dashboard (JSON + per-KPI chart URLs) ----------
@app.get("/dashboard")
def dashboard(request: Request,
              period: str = Query(None, pattern="^(week|biweekly|month|quarter)$"),
              by: str = Query(None, pattern="^(priority|asset|location|assignee|status)$")):
    d = _load_df()
    out = {name: func(d, period=period, by=by) for name, func in KPI_FUNCS.items()}
    for n in REGISTRY["defs"].keys():
        out[n] = kpi_from_sql(n)

    base = str(request.base_url).rstrip("/")
    charts: List[Dict[str, str]] = []
    for name, payload in out.items():
        style = _preferred_chart_for(name, payload, period)
        qs = []
        if name in KPI_FUNCS:
            if period: qs.append(f"period={period}")
            if by:     qs.append(f"by={by}")
        qs.append(f"chart={style}")
        qs = "&".join(qs)
        charts.append({
            "kpi": name,
            "title": payload.get("title", name.upper()),
            "url": f"{base}/chart/{name}?{qs}"
        })

    return JSONResponse({
        "period": period,
        "by": by,
        "kpis": out,
        "kpi_names": list(KPI_FUNCS.keys()) + list(REGISTRY["defs"].keys()),
        "charts": charts
    })

# ---------- Dashboard Chart (combined PNG grid) ----------
@app.get("/dashboard/chart")
def dashboard_chart(period: str = Query(None, pattern="^(week|biweekly|month|quarter)$"),
                    by: str = Query(None, pattern="^(priority|asset|location|assignee|status)$")):
    d = _load_df()
    payloads = {name: func(d, period=period, by=by) for name, func in KPI_FUNCS.items()}
    for n in REGISTRY["defs"].keys():
        payloads[n] = kpi_from_sql(n)

    # more breathing room + automatic spacing
    fig, axs = plt.subplots(3, 2, figsize=(16, 16), constrained_layout=True)
    axes = axs.flatten()
    idx_ax = 0

    def _labels_from_payload(payload: Dict) -> List[str]:
        t = payload.get("type")
        if t == "bar":
            return [str(x) for x in payload.get("labels", [])]
        if t == "table":
            return [lab if isinstance(lab, str) else " · ".join(map(str, lab))
                    for lab in payload.get("index", [])]
        if t == "ratio":
            rows = payload.get("rows", [])
            agg = defaultdict(list)
            for r in rows:
                agg[str(r["status"])].append(float(r["pct"]))
            return list(agg.keys())
        return []

    def _set_axis_labels(ax, name: str):
        n = name.lower()
        if n == "mttr":
            ax.set_xlabel("Date"); ax.set_ylabel("Rate (min)")
        elif n == "ticket_volume":
            ax.set_xlabel("Date"); ax.set_ylabel("Number of tickets")
        elif n == "satisfaction_avg":
            ax.set_xlabel("Date"); ax.set_ylabel("Score (1–5)")
        elif n == "ftfr":
            ax.set_xlabel("Date"); ax.set_ylabel("Rate")
        elif n in ("top_locations", "top_locations_table"):
            ax.set_xlabel("City"); ax.set_ylabel("Number of tickets")
        elif n == "status_ratio":
            # pies don’t use axes; bar fallback should have %
            ax.set_ylabel("%")

    for name, payload in payloads.items():
        if idx_ax >= len(axes):
            break
        ax = axes[idx_ax]; idx_ax += 1

        title = payload.get("title", name.upper())
        ptype = payload.get("type")
        style = _preferred_chart_for(name, payload, period)
        _set_axis_labels(ax, name)

        # unified labels (used by bar/line helpers)
        labels = _labels_from_payload(payload)

        if ptype == "metric":
            ax.bar([title], [payload["value"]])
            ax.set_title(title)

        elif ptype in ("bar", "table"):
            # normalize values
            if ptype == "bar":
                values = payload["values"]
            else:
                values = payload.get("values", [])
            if style == "pie" and len(labels) <= 12:
                ax.pie(values, labels=[_short_label(l) for l in labels], autopct="%1.0f%%")
                ax.set_title(title)
            elif style == "line":
                x = list(range(len(values)))
                ax.plot(x, values, marker="o", linewidth=1.6)
                ax.set_title(title)
                # smart, abbreviated tick labels for dates/periods
                _apply_smart_xticks(ax, labels, max_ticks=10, rotation=30)
                ax.grid(axis="y", alpha=0.3)
            else:
                x = np.arange(len(values))
                ax.bar(x, values)
                ax.set_title(title)
                _apply_smart_xticks(ax, labels, max_ticks=10, rotation=30)
                ax.grid(axis="y", alpha=0.3)

        elif ptype == "ratio":
            rows = payload.get("rows", [])
            if not rows:
                ax.text(0.5, 0.5, "No data", ha="center", va="center"); ax.set_title(title); continue
            agg = defaultdict(list)
            for r in rows:
                agg[str(r["status"])].append(float(r["pct"]))
            labels = list(agg.keys())
            values = [sum(v)/len(v) for v in agg.values()]
            if style == "pie":
                ax.pie(values, labels=labels, autopct="%1.0f%%"); ax.set_title(title)
            else:
                x = np.arange(len(values))
                ax.bar(x, values)
                ax.set_title(title)
                _apply_smart_xticks(ax, labels, max_ticks=8, rotation=0)
                ax.set_ylabel("%")
                ax.grid(axis="y", alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No chart", ha="center", va="center"); ax.set_title(title)

    # hide any unused axes
    while idx_ax < len(axes):
        axes[idx_ax].axis("off"); idx_ax += 1

    png = _png(fig)
    return Response(content=png, media_type="image/png")

# ---------- Categories ----------
@app.get("/categories")
def categories():
    d = _load_df()
    def uniques(colnames):
        for c in colnames:
            if c in d.columns:
                return sorted(list(pd.Series(d[c]).dropna().astype(str).unique()))
        return []
    return {
        "priority": uniques(["priority","ticket_priority"]),
        "asset":    uniques(["asset_name","asset","machine","equipment"]),
        "location": uniques(["location","site","branch"]),
        "assignee": uniques(["assignee_username","assignee","technician"]),
        "status":   uniques(["status","ticket_status"]),
        "periods":  ["week","biweekly","month","quarter"],
        "kpis":     list(KPI_FUNCS.keys()) + list(REGISTRY["defs"].keys())
    }

# ---------- Export KPIs as CSV ----------
@app.get("/export/kpis.csv")
def export_kpis_csv(period: str = Query(None, pattern="^(week|biweekly|month|quarter)$"),
                    by: str = Query(None, pattern="^(priority|asset|location|assignee|status)$")):
    d = _load_df()
    rows = []
    for name, func in KPI_FUNCS.items():
        payload = func(d, period=period, by=by)
        t = payload.get("type")
        if t == "metric":
            rows.append({"kpi": name, "group": "", "value": payload["value"]})
        elif t == "bar":
            for lab, val in zip(payload["labels"], payload["values"]):
                rows.append({"kpi": name, "group": str(lab), "value": val})
        elif t == "table":
            for lab, val in zip(payload["index"], payload["values"]):
                lab_str = lab if isinstance(lab, str) else " · ".join(map(str, lab))
                rows.append({"kpi": name, "group": lab_str, "value": val})
        elif t == "ratio":
            for r in payload.get("rows", []):
                grp = r.get("group")
                grp_str = grp if isinstance(grp, str) else (" · ".join(map(str, grp)) if grp else "")
                rows.append({"kpi": name, "group": f"{grp_str} :: {r.get('status')}", "value": r.get("pct")})
    for n in REGISTRY["defs"].keys():
        payload = kpi_from_sql(n)
        if payload.get("type") == "bar":
            for lab, val in zip(payload["labels"], payload["values"]):
                rows.append({"kpi": n, "group": str(lab), "value": val})
    if not rows:
        raise HTTPException(400, "No KPI data available to export.")
    out_df = pd.DataFrame(rows)
    csv_path = Path("kpis_export.csv")
    out_df.to_csv(csv_path, index=False)
    return FileResponse(str(csv_path), filename="kpis_export.csv", media_type="text/csv")

# ---------- Download the SQLite DB ----------
@app.get("/download/db")
def download_db():
    if not DB_PATH.exists():
        _prepare_sqlite()
    return FileResponse(str(DB_PATH), filename="kpi_dashboard.db", media_type="application/octet-stream")

# --------- Lightweight intent & name handling for snappy UX ---------------
INTENTS = {
    "help": [
        r"^\s*help(?:\s+me)?\b",
        r"\b(help|how\s+to|what\s+can\s+you\s+do|commands?)\b",
        r"\bplease\s+provide\s+data\b",
        r"\bwhat\s+(?:data|kpis|metrics)\s+(?:are|is)\s+available\b",
        r"\bwhat\s+data\s+do\s+you\s+have\b",
        r"\bavailable\s+data\b",
        r"\blist\s+(?:data|kpis|metrics)\b",
        r"\bwhat\s+can\s+i\s+see\b",
    ],
    "by_help": [
        r"^\s*by\s*$",
    ],
    "greeting": [
        r"^\s*(hi|hello|hey|howdy|good\s*(?:morning|afternoon|evening))\b",
    ],
    "goodbye": [
        r"^\s*(?:b+ye+|bye|goodbye|see\s*(?:ya|you)|ciao|later)\b",
    ],
}

def _normalize(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00A0", " ")
    s = s.replace("\u200B", "").replace("\u200C", "").replace("\u200D", "")
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _simple_lower(s: str) -> str:
    return _normalize(s).lower()

def predict_intent(msg: str):
    m = _simple_lower(msg)

    # very short commands first
    if m in {"hi","hello","hey","howdy","good morning","good afternoon","good evening"}:
        return "greeting"
    if m in {"bye","goodbye","see ya","see you","ciao","later"}:
        return "goodbye"
    if m == "by":
        return "by_help"
    if m in {"help","help me","help me please","how to","what can you do","commands","command","?","please provide data"} or m.startswith("help "):
        return "help"

    # regex fallback
    for p in INTENTS["help"]:
        if re.search(p, m):
            return "help"
    for p in INTENTS["by_help"]:
        if re.search(p, m):
            return "by_help"
    for p in INTENTS["greeting"]:
        if re.search(p, m):
            return "greeting"
    for p in INTENTS["goodbye"]:
        if re.search(p, m):
            return "goodbye"
    return None

NAME_BADWORDS = {"mttr","ticket","tickets","status","volume","dashboard","help","bye","goodbye","month","week","biweekly","quarter","by","asset","priority","location","assignee","status","line","bar","pie"}

def extract_name(msg: str) -> Optional[str]:
    """
    Only greet when the user explicitly introduces their name.
    """
    raw = _normalize(msg)
    m = re.search(
        r"\b(?:my\s+name\s+is|i\s*am|i'?m|this\s+is)\s+([A-Za-z][A-Za-z\-\s]{1,40})\b",
        raw,
        flags=re.I,
    )
    if not m:
        return None

    cand = re.sub(r"[^\w\-\s]", "", m.group(1)).strip()
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]{1,40}", cand)
    if not tokens:
        return None
    if any(t.lower() in NAME_BADWORDS for t in tokens):
        return None

    return " ".join(t.capitalize() for t in tokens)

# ---------- LLM parser used by /chat ----------
def llm_understand(user_message: str) -> dict:
    """
    Intent parser that can use dynamic few-shots (REGISTRY['fewshots']).
    If USE_LLM_DURING_EVAL=0, falls back to deterministic parsing.
    """
    client, model = get_openai_client()
    use_llm = bool(client) and os.getenv("USE_LLM_DURING_EVAL", "1").strip().lower() not in {"0","false","no"}

    # Deterministic fallback (no LLM or disabled during eval)
    if not use_llm:
        p, b, c = extract_period_by_chart(user_message)
        tokens = re.findall(r"[a-z0-9_]+", (user_message or "").lower())
        name_guess = resolve_kpi_name(tokens)
        action = "show" if name_guess else "help"
        return {"action": action, "name": name_guess, "period": p, "by": b, "chart": c}

    valid_names = sorted(set(list(KPI_FUNCS.keys()) + list(REGISTRY["defs"].keys()) + ["dashboard"]))

    system_msg = (
        "You are an intent parser for a KPI dashboard. "
        "Return ONLY a JSON object with keys: "
        "{action,name,period,by,chart}. "
        "action ∈ ['dashboard','get','show','analyze','help','greeting']; "
        f"name ∈ {valid_names}. "
        "If a field is not present, set it to null. Do not add extra keys."
    )

    static_fewshots = [
        ("hi",                           {"action":"greeting","name":None,"period":None,"by":None,"chart":None}),
        ("help me please",               {"action":"help","name":None,"period":None,"by":None,"chart":None}),
        ("dashboard",                    {"action":"dashboard","name":"dashboard","period":None,"by":None,"chart":None}),
        ("mttr month by asset line",     {"action":"show","name":"mttr","period":"month","by":"asset","chart":"line"}),
        ("ticket volume by location",    {"action":"show","name":"ticket_volume","period":None,"by":"location","chart":None}),
        ("status mix by month",          {"action":"show","name":"status_ratio","period":"month","by":None,"chart":None}),
        ("top locations",                {"action":"show","name":"top_locations","period":None,"by":None,"chart":None}),
    ]

    dynamic = REGISTRY.get("fewshots") or []  # NEW: learned from K examples
    fewshots = static_fewshots + dynamic

    try:
        msgs = [{"role":"system","content":system_msg}]
        for u, j in fewshots:
            msgs.append({"role":"user","content":u})
            msgs.append({"role":"assistant","content":json.dumps(j)})
        msgs.append({"role":"user","content":user_message})

        r = client.responses.create(model=model, input=msgs, temperature=0, max_output_tokens=200)
        raw = (r.output_text or "").strip()
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            raw = raw[start:end+1]
        return json.loads(raw)
    except Exception as e:
        # Robust deterministic fallback
        p, b, c = extract_period_by_chart(user_message)
        tokens = re.findall(r"[a-z0-9_]+", (user_message or "").lower())
        name_guess = resolve_kpi_name(tokens)
        action = "show" if name_guess else "help"
        return {"action": action, "name": name_guess, "period": p, "by": b, "chart": c}

def llm_chat(user_message: str) -> str:
    """
    Fallback: keep it short and on-task; avoid trivia like name origins.
    """
    m = _simple_lower(user_message)
    if re.search(r"\b(data|kpi|metric)s?\b", m):
        return "Tell me what KPI and grouping you want. For example: 'mttr month by asset line', 'ticket volume by location', or 'dashboard'."
    return "I can help you explore your KPIs. Try: 'dashboard', 'mttr month by asset line', 'ticket volume by location'."

# ---------- Lightweight param extractor (period/by/chart) ----------
BY_FIELDS = ["priority","asset","location","assignee","status"]

def infer_by_field(text: str) -> Optional[str]:
    m = (text or "").lower()
    for fld in BY_FIELDS:
        if re.search(rf"\b{fld}\b", m):
            return fld
    return None

TOKEN_CHART   = {"bar", "line", "pie"}

def extract_period_by_chart(text: str):
    m = (text or "").lower()
    # period
    period = None
    if re.search(r"\bbi[-\s]?weekly\b", m) or re.search(r"\bfortnight\b", m) or re.search(r"\b(?:2|two)\s*weeks?\b", m):
        period = "biweekly"
    elif re.search(r"\bweek\b", m):
        period = "week"
    elif re.search(r"\bmonth\b", m):
        period = "month"
    elif re.search(r"\bquarter\b", m):
        period = "quarter"

    # by
    by = None
    m_by = re.search(r"\bby\s+(priority|asset|location|assignee|status)\b", m)
    if m_by:
        by = m_by.group(1)
    else:
        by = infer_by_field(m)

    # chart
    chart = None
    if re.search(r"\bline\b", m):
        chart = "line"
    elif re.search(r"\bpie(\s+chart)?\b", m):
        chart = "pie"
    elif re.search(r"\bbar(\s+(graph|chart))?\b", m):
        chart = "bar"

    return period, by, chart

def _catalog_from_df(d: pd.DataFrame) -> dict:
    def uniques(cols):
        for c in cols:
            if c in d.columns:
                return sorted(list(pd.Series(d[c]).dropna().astype(str).unique()))
        return []
    return {
        "kpis": sorted(list(KPI_FUNCS.keys()) + list(REGISTRY["defs"].keys())),
        "periods": ["week","biweekly","month","quarter"],
        "by_fields": ["priority", "asset", "location", "assignee", "status"],
        "by_values": {
            "priority": uniques(["priority","ticket_priority"]),
            "asset":    uniques(["asset_name","asset","machine","equipment"]),
            "location": uniques(["location","site","branch"]),
            "assignee": uniques(["assignee_username","assignee","technician"]),
            "status":   uniques(["status","ticket_status"]),
        }
    }

def _preview_values(vals: List[str], n=10) -> str:
    vals = vals or []
    head = vals[:n]
    more = max(0, len(vals) - len(head))
    return ", ".join(head) + (f" … (+{more} more)" if more else "")

def make_available_payload(d: pd.DataFrame):
    cat = _catalog_from_df(d)
    reply = (
        "Available data:\n"
        f"- KPIs: {', '.join(cat['kpis'])}\n"
        f"- Periods: {', '.join(cat['periods'])}\n"
        f"- Group by: {', '.join(cat['by_fields'])}\n"
        f"- Priority: {_preview_values(cat['by_values']['priority'])}\n"
        f"- Asset: {_preview_values(cat['by_values']['asset'])}\n"
        f"- Location: {_preview_values(cat['by_values']['location'])}\n"
        f"- Assignee: {_preview_values(cat['by_values']['assignee'])}\n"
        f"- Status: {_preview_values(cat['by_values']['status'])}"
    )
    return {
        "intent": "help",  # keep it in NON-CHART path for the frontend
        "reply": reply,
        "kpis": cat["kpis"],
        "periods": cat["periods"],
        "by": cat["by_fields"],
        "by_values": cat["by_values"],
    }

def make_greeting_payload(d: pd.DataFrame, name: Optional[str]=None):
    cat = _catalog_from_df(d)
    hello = f"Hi {name}!" if name else "Hi!"
    return {
        "intent": "greeting",
        "reply": f"{hello} How can I help you?\nExamples: 'dashboard', 'mttr month by asset line', 'ticket volume by location'",
        "quick": {
            "kpis": cat["kpis"][:6],
            "periods": cat["periods"],
            "by_fields": cat["by_fields"]
        }
    }

def make_by_help_payload(d: pd.DataFrame):
    cat = _catalog_from_df(d)
    return {
        "intent": "by_help",
        "reply": "You can group by one of: priority, asset, location, assignee, status.\nExample: 'mttr month by asset' or 'ticket volume by location bar'.",
        "by_fields": cat["by_fields"],
        "by_values": cat["by_values"]
    }

def make_help_payload(d: pd.DataFrame):
    examples = [
        "dashboard",
        "mttr month by asset line",
        "ticket volume by location bar",
        "satisfaction_avg quarter line",
        "status mix by month",
        "top locations",
    ]
    cat = _catalog_from_df(d)
    return {
        "intent": "help",
        "reply": "Here are some things I can do. Ask for a KPI with period/by/chart.",
        "examples": examples,
        "kpis": cat["kpis"],
        "periods": cat["periods"],
        "by": cat["by_fields"],
        "charts": ["bar", "line", "pie"],
    }

def make_goodbye_payload():
    return {"intent":"goodbye","reply":"Bye"}

@app.post("/chat")
def chat(inp: ChatIn, request: Request):
    d = _load_df()
    load_registry()
    base = str(request.base_url).rstrip("/")
    user_msg = (inp.message or "").strip()

    # 1) Local intents (instant answers)
    local = predict_intent(user_msg)
    if local == "greeting":
        return JSONResponse({"intent": "greeting", "payload": make_greeting_payload(d)})
    if local == "by_help":
        return JSONResponse({"intent": "by_help", "payload": make_by_help_payload(d)})
    if local == "help":
        # If user asked "what data is available", list it
        import re
        if re.search(r"(what\s+data.*(available|have)|available\s+data|list\s+(data|kpis|metrics)|what\s+can\s+i\s+see)", user_msg, flags=re.I):
            return JSONResponse({"intent": "help", "payload": make_available_payload(d)})
        return JSONResponse({"intent": "help", "payload": make_help_payload(d)})
    if local == "goodbye":
        return JSONResponse({"intent": "goodbye", "payload": make_goodbye_payload()})

    # 2) LLM parser (structured)
    parsed = llm_understand(user_msg) or {}
    action = (parsed.get("action") or "").lower()
    name_kpi = (parsed.get("name") or "").lower() or None
    period = parsed.get("period") or None
    by     = parsed.get("by") or None
    chart  = parsed.get("chart") or None

    # 2a) Dashboard
    if action == "dashboard":
        return JSONResponse({
            "intent": "dashboard",
            "payload": {"reply": "Here’s the dashboard view.", "url": f"{base}/dashboard"},
            "parsed": {"period": period, "by": by, "chart": chart}
        })

    # 2b) KPI action
    if action in {"get","show","analyze"} and name_kpi:
        d_payload = KPI_FUNCS[name_kpi](d, period=period, by=by) if name_kpi in KPI_FUNCS else kpi_from_sql(name_kpi)
        chosen_chart, notice = _validate_chart_for(name_kpi, chart, d_payload, period)
        return JSONResponse({
            "intent": name_kpi,
            "payload": d_payload,
            "period": period,
            "by": by,
            "chart": chosen_chart,
            "notice": notice,
            "parsed": {"period": period, "by": by, "chart": chart}
        })

    # 2c) Deterministic fallback (no LLM)
    import re
    tokens = re.findall(r"[a-z0-9_]+", user_msg.lower())
    kpi_guess = resolve_kpi_name(tokens)
    if kpi_guess:
        p, b, c = extract_period_by_chart(user_msg)
        d_payload = KPI_FUNCS[kpi_guess](d, period=p, by=b) if kpi_guess in KPI_FUNCS else kpi_from_sql(kpi_guess)
        chosen_chart, notice = _validate_chart_for(kpi_guess, c, d_payload, p)
        return JSONResponse({
            "intent": kpi_guess,
            "payload": d_payload,
            "period": p,
            "by": b,
            "chart": chosen_chart,
            "notice": notice,
            "parsed": {"period": p, "by": b, "chart": c}
        })

    # 3) Last resort: greet only if the user explicitly gave a name; else small talk
    name = extract_name(user_msg)
    if name:
        return JSONResponse({"intent": "greeting", "payload": make_greeting_payload(d, name=name)})

    reply = llm_chat(user_msg)
    return JSONResponse({"intent": "chat", "payload": {"reply": reply}})

# ------ Admin: upsert KPI definition ----------
@app.post("/admin/kpis")
def upsert_kpi(defn: KpiDefIn):
    _ensure_learning_tables()
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""INSERT INTO kpi_definitions(name, sql, chart_hint, description, enabled, updated_at)
                       VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                       ON CONFLICT(name) DO UPDATE SET
                         sql=excluded.sql,
                         chart_hint=excluded.chart_hint,
                         description=excluded.description,
                         enabled=excluded.enabled,
                         updated_at=CURRENT_TIMESTAMP""",
                    (defn.name.lower(), defn.sql, defn.chart_hint, defn.description, int(defn.enabled or 1)))
        con.commit()
    load_registry(force=True)
    return {"ok": True, "kpi": defn.name.lower()}

# ---------- Admin: upsert synonym ----------
@app.post("/admin/synonyms")
def upsert_synonym(s: SynIn):
    _ensure_learning_tables()
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""INSERT INTO kpi_synonyms(term, kpi_name, weight)
                       VALUES (?, ?, ?)
                       ON CONFLICT(term) DO UPDATE SET
                         kpi_name=excluded.kpi_name,
                         weight=excluded.weight""",
                    (s.term.lower(), s.kpi_name.lower(), float(s.weight or 1.0)))
        con.commit()
    load_registry(force=True)
    return {"ok": True, "term": s.term.lower(), "kpi_name": s.kpi_name.lower()}

# ---------- Admin: feedback ----------
@app.post("/admin/feedback")
def add_feedback(fb: FeedbackIn):
    _ensure_learning_tables()
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""INSERT INTO feedback(utterance_id, label, note)
                       VALUES (?, ?, ?)""", (fb.utterance_id, fb.label, fb.note))
        con.commit()
    return {"ok": True}

# ============================== Paper-style Evaluation  ==============================
# Dataset format (JSONL per line):
# {"text":"mttr month by asset line","gold":{"name":"mttr","period":"month","by":"asset","chart":"line"}}

_EVAL_FIELDS = ("name", "period", "by", "chart")
_WORD_RE_EVAL = re.compile(r"[a-z0-9_]+")

def _eval_tokens(s: str) -> List[str]:
    return _WORD_RE_EVAL.findall((s or "").lower())

def _vec(cnt: Dict[str,int]) -> Dict[str,float]:
    if not cnt: return {}
    norm = math.sqrt(sum(v*v for v in cnt.values()))
    return {k: (v / norm) for k, v in cnt.items()} if norm > 0 else dict(cnt)

def _cosine(v1: Dict[str,float], v2: Dict[str,float]) -> float:
    if not v1 or not v2: return 0.0
    if len(v1) > len(v2): v1, v2 = v2, v1
    return float(sum(v * v2.get(k, 0.0) for k, v in v1.items()))

def _pairwise_sim(texts: List[str]) -> np.ndarray:
    vecs = [_vec(Counter(_eval_tokens(t))) for t in texts]
    n = len(texts)
    S = np.zeros((n, n), dtype=float)
    for i in range(n):
        S[i, i] = 1.0
        for j in range(i+1, n):
            s = _cosine(vecs[i], vecs[j])
            S[i, j] = s
            S[j, i] = s
    return S

def _rps_rank(texts: List[str]) -> List[int]:
    """
    Ratio-Penalty Selection ranking indices for given texts.
    Score(s | X) = sum_y sim(s,y) / (1 + sum_x∈X sim(s,x))
    """
    n = len(texts)
    if n == 0: return []
    S = _pairwise_sim(texts)
    cov = S.sum(axis=1)  # coverage
    chosen: List[int] = []
    remaining: Set[int] = set(range(n))
    while remaining:
        if not chosen:
            i = int(np.argmax(cov))
            chosen.append(i); remaining.remove(i); continue
        penalty = S[list(remaining)][:, chosen].sum(axis=1)
        gains = cov[list(remaining)] / (1.0 + penalty)
        pool = list(remaining)
        i = pool[int(np.argmax(gains))]
        chosen.append(i); remaining.remove(i)
    return chosen

def _load_eval_items(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found at '{path.resolve()}'. "
            "Create eval_dataset.jsonl with lines like: "
            '{"text":"mttr month by asset line","gold":{"name":"mttr","period":"month","by":"asset","chart":"line"}}'
        )
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                if "text" in obj and "gold" in obj and isinstance(obj["gold"], dict):
                    out.append(obj)
            except Exception:
                continue
    if not out:
        raise ValueError("Evaluation file is empty or malformed.")
    return out

def _stratified_split(items: List[dict], test_size: float=0.2, seed: int=42) -> Tuple[List[dict], List[dict]]:
    by_name = defaultdict(list)
    for it in items:
        nm = ((it.get("gold") or {}).get("name") or "").lower()
        by_name[nm].append(it)
    rng = random.Random(seed)
    train, test = [], []
    for nm, lst in by_name.items():
        rng.shuffle(lst)
        n_test = max(1, int(round(test_size * len(lst))))
        test.extend(lst[:n_test])
        train.extend(lst[n_test:])
    rng.shuffle(train); rng.shuffle(test)
    return train, test

def _apply_training_from_k(train_items: List[dict], k: int, selector: str, seed: int) -> Dict[str, object]:
    rng = random.Random(seed)
    texts = [it["text"] for it in train_items]
    if selector == "rps":
        order = _rps_rank(texts)
        keep = order[:min(k, len(order))]
    else:
        idxs = list(range(len(train_items))); rng.shuffle(idxs)
        keep = idxs[:min(k, len(idxs))]
    picked = [train_items[i] for i in keep]

    # Build synonyms + few-shots from the picked K examples
    syn_added = 0
    fewshots_pairs: List[Tuple[str, dict]] = []
    for it in picked:
        text = it["text"]
        gold = it.get("gold") or {}
        name = (gold.get("name") or "").lower().strip()
        if not name: continue
        # add simple token->name synonyms (do not overwrite admin synonyms)
        for t in _eval_tokens(text):
            if len(t) >= 3 and t not in {"month","week","biweekly","quarter","by","line","bar","pie"}:
                if t not in REGISTRY["synonyms"]:
                    REGISTRY["synonyms"][t] = name
                    syn_added += 1
        # add few-shot (user text, normalized json)
        fewshots_pairs.append((text, {
            "action": "show",
            "name": name or None,
            "period": (gold.get("period") or None),
            "by": (gold.get("by") or None),
            "chart": (gold.get("chart") or None),
        }))
    REGISTRY["fewshots"] = fewshots_pairs
    return {"picked": len(picked), "synonyms_added": syn_added, "fewshots_added": len(fewshots_pairs)}

def _safe_equal(a: Optional[str], b: Optional[str]) -> bool:
    if a is None or b is None: return False
    return str(a).strip().lower() == str(b).strip().lower()

def _evaluate_items(items: List[dict]) -> Dict[str, object]:
    tp = fp = fn = 0
    per_field = {f: {"tp":0,"fp":0,"fn":0} for f in _EVAL_FIELDS}
    samples = []

    for it in items:
        text = it.get("text","")
        gold = (it.get("gold") or {})
        gold_norm = {k: (gold.get(k).lower() if isinstance(gold.get(k), str) else gold.get(k)) for k in _EVAL_FIELDS}

        parsed = llm_understand(text) or {}
        pred = {
            "name": (parsed.get("name") or None),
            "period": (parsed.get("period") or None),
            "by": (parsed.get("by") or None),
            "chart": (parsed.get("chart") or None),
        }
        pred_norm = {k: (pred.get(k).lower() if isinstance(pred.get(k), str) else pred.get(k)) for k in _EVAL_FIELDS}
        samples.append({"text": text, "gold": gold_norm, "pred": pred_norm})

        for f in _EVAL_FIELDS:
            g = gold_norm.get(f); p = pred_norm.get(f)
            if g is not None and p is not None:
                if _safe_equal(g, p):
                    tp += 1; per_field[f]["tp"] += 1
                else:
                    fp += 1; fn += 1
                    per_field[f]["fp"] += 1; per_field[f]["fn"] += 1
            elif g is None and p is not None:
                fp += 1; per_field[f]["fp"] += 1
            elif g is not None and p is None:
                fn += 1; per_field[f]["fn"] += 1

    def _prf(tpi, fpi, fni):
        prec = tpi / (tpi + fpi) if (tpi + fpi) > 0 else 0.0
        rec  = tpi / (tpi + fni) if (tpi + fni) > 0 else 0.0
        f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    micro_p, micro_r, micro_f1 = _prf(tp, fp, fn)
    fields = {}
    for f, c in per_field.items():
        p, r, f1 = _prf(c["tp"], c["fp"], c["fn"])
        fields[f] = {"precision": round(p,4), "recall": round(r,4), "f1": round(f1,4), **c}

    return {
        "n_items": len(items),
        "micro": {"precision": round(micro_p,4), "recall": round(micro_r,4), "f1": round(micro_f1,4),
                  "tp": tp, "fp": fp, "fn": fn},
        "per_field": fields,
        "samples": samples[:25]
    }

@app.get("/eval/nlu/single")
def eval_nlu_single(k: int = Query(20, ge=1),
                    selector: str = Query("rps", pattern="^(random|rps)$"),
                    seed: int = 42,
                    test_size: float = Query(0.2, ge=0.05, le=0.5),
                    eval_path: Optional[str] = None):
    """
    Train/test split (stratified by 'name') + train with K labeled examples (random or RPS) +
    evaluate on the held-out test set. Returns micro-F1 across (name, period, by, chart).
    """
    items = _load_eval_items(Path(eval_path) if eval_path else EVAL_PATH)
    train_items, test_items = _stratified_split(items, test_size=test_size, seed=seed)

    # reset learned adapters before training
    load_registry(force=True)

    train_info = _apply_training_from_k(train_items, k=k, selector=selector, seed=seed)
    test_results = _evaluate_items(test_items)

    return JSONResponse({
        "selector": selector,
        "k": k,
        "seed": seed,
        "test_size": test_size,
        "train_counts": {"total": len(train_items), "picked": train_info["picked"]},
        "adapters": {"synonyms_added": train_info["synonyms_added"], "fewshots_added": train_info["fewshots_added"]},
        "test_results": test_results,
        "eval_file": str((Path(eval_path) if eval_path else EVAL_PATH).resolve())
    })

@app.get("/eval/nlu/curve")
def eval_nlu_curve(ks: str = Query("10,20,40,80"),
                   selector: str = Query("rps", pattern="^(random|rps)$"),
                   seed: int = 42,
                   test_size: float = Query(0.2, ge=0.05, le=0.5),
                   eval_path: Optional[str] = None):
    """
    Evaluate micro-F1 for multiple K values (learning curve). Example: ks=10,20,40,80
    """
    items = _load_eval_items(Path(eval_path) if eval_path else EVAL_PATH)
    train_items, test_items = _stratified_split(items, test_size=test_size, seed=seed)
    rng = random.Random(seed)
    texts = [it["text"] for it in train_items]
    rps_order = _rps_rank(texts) if selector == "rps" else None

    def run_for_k(k: int) -> Dict[str, object]:
        load_registry(force=True)  # reset adapters
        if selector == "rps":
            keep_idx = rps_order[:min(k, len(train_items))]
        else:
            idxs = list(range(len(train_items))); rng.shuffle(idxs)
            keep_idx = idxs[:min(k, len(train_items))]
        picked = [train_items[i] for i in keep_idx]
        # train adapters
        REGISTRY["fewshots"] = []
        syn_added = 0
        for it in picked:
            gold = it.get("gold") or {}
            nm = (gold.get("name") or "").lower().strip()
            if not nm: continue
            for t in _eval_tokens(it["text"]):
                if len(t) >= 3 and t not in {"month","week","biweekly","quarter","by","line","bar","pie"}:
                    if t not in REGISTRY["synonyms"]:
                        REGISTRY["synonyms"][t] = nm
                        syn_added += 1
            REGISTRY["fewshots"].append((it["text"], {
                "action":"show",
                "name": nm or None,
                "period": (gold.get("period") or None),
                "by": (gold.get("by") or None),
                "chart": (gold.get("chart") or None),
            }))
        res = _evaluate_items(test_items)
        return {"k": k, "micro_f1": res["micro"]["f1"], "precision": res["micro"]["precision"],
                "recall": res["micro"]["recall"], "per_field": res["per_field"],
                "synonyms_added": syn_added, "fewshots_added": len(REGISTRY["fewshots"])}

    k_list = [int(x) for x in ks.split(",") if str(x).strip().isdigit()]
    k_list = [k for k in k_list if k >= 1]
    points = [run_for_k(k) for k in k_list]

    return JSONResponse({
        "selector": selector,
        "seed": seed,
        "test_size": test_size,
        "points": points,
        "eval_file": str((Path(eval_path) if eval_path else EVAL_PATH).resolve())
    })

@app.get("/debug/routes")
def _debug_routes():
    return {
        "app_file": __file__,
        "paths": [r.path for r in app.routes],
        "has_eval_single": "/eval/nlu/single" in [r.path for r in app.routes],
        "has_eval_curve": "/eval/nlu/curve" in [r.path for r in app.routes],
    }

