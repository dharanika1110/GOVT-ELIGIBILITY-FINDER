"""
Data workbench - analyse ANY tabular file (NFHS, census, health, education, survey ...).
No Streamlit code here so it can be tested on its own.
"""
from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

import engine as E

SEED = 42
MISSING_TOKENS = {"", "*", "-", "--", "—", "na", "n/a", "nan", "null", "none", "…", "..", "nil"}


# --------------------------------------------------------------------------- #
# Reading any format
# --------------------------------------------------------------------------- #
def list_sheets(raw: bytes, name: str) -> list[str]:
    if name.lower().endswith((".xlsx", ".xlsm", ".xls")):
        try:
            return pd.ExcelFile(io.BytesIO(raw)).sheet_names
        except Exception:
            return []
    return []


def read_any(raw: bytes, name: str, sheet=0) -> pd.DataFrame:
    """Read CSV / TSV / TXT / Excel / JSON into a DataFrame (delimiter and encoding auto-detected)."""
    n = name.lower()
    if n.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(io.BytesIO(raw), sheet_name=sheet)
    if n.endswith(".json"):
        try:
            return pd.read_json(io.BytesIO(raw))
        except ValueError:
            return pd.json_normalize(__import__("json").loads(raw.decode("utf-8-sig")))
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
        except pd.errors.ParserError:
            return pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding=enc, on_bad_lines="skip")
    raise ValueError("Could not read this file. Supported: CSV, TSV, TXT, XLSX, XLS, JSON.")


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def _to_num(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, (int, float, np.number)):
        return float(v)
    s = str(v).strip()
    if s.lower() in MISSING_TOKENS:
        return np.nan
    s = re.sub(r"^\((.*)\)$", r"\1", s)            # NFHS style "(45.2)" = small sample
    s = re.sub(r"[,%₹$\s]", "", s)
    try:
        return float(s)
    except ValueError:
        return np.nan


def coerce_numeric_like(df: pd.DataFrame, thresh: float = 0.8) -> tuple[pd.DataFrame, list[str]]:
    """Turn text columns such as '1,234', '45%', '(12.3)', '*' into numbers when >= thresh of values parse."""
    df, converted = df.copy(), []
    for c in df.columns:
        col = df[c]
        if pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col) \
                or pd.api.types.is_datetime64_any_dtype(col):
            continue
        filled = col.notna() & (col.astype(str).str.strip().str.lower().isin(MISSING_TOKENS) == False)  # noqa: E712
        if filled.sum() == 0:
            continue
        conv = col.map(_to_num)
        if conv[filled].notna().mean() >= thresh:
            df[c] = conv
            converted.append(str(c))
    return df, converted


def tidy_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() or f"col_{i}" for i, c in enumerate(df.columns)]
    # make duplicate column names unique
    seen, cols = {}, []
    for c in df.columns:
        seen[c] = seen.get(c, 0) + 1
        cols.append(c if seen[c] == 1 else f"{c}_{seen[c]}")
    df.columns = cols
    return df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)


def _id_like(df: pd.DataFrame, c: str) -> bool:
    col = df[c]
    if not pd.api.types.is_numeric_dtype(col) or len(df) < 5:
        return False
    named = bool(re.search(r"(^|[^a-z])(id|code|sl\.?\s?no|s\.?no|serial|index|pincode|pin)([^a-z]|$)", c.lower()))
    unique_int = (col.notna().all() and (col % 1 == 0).all() and col.nunique() == len(df)
                  and (col.is_monotonic_increasing or col.is_monotonic_decreasing))
    return named or unique_int


def detect(df: pd.DataFrame) -> dict:
    """Find numeric analysis columns, grouping columns, a name column and a state column."""
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
           and not _id_like(df, c) and df[c].nunique() > 1]
    nonnum = [c for c in df.columns if c not in num and not pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in nonnum if 1 < df[c].nunique() <= 60]
    states = set(E.STATES_ALL)
    state_col = None
    for c in nonnum:
        vals = df[c].dropna().astype(str).map(E.canon_state)
        if len(vals) and ((re.search(r"state|st_nm|\but\b", c.lower()) is not None) or vals.isin(states).mean() >= .6):
            if vals.isin(states).mean() >= .4:
                state_col = c
                break
    name_col = None
    for c in nonnum:
        if re.search(r"district|name|city|town|area|block|region|village|ward", c.lower()) and c != state_col:
            name_col = c
            break
    if name_col is None:
        for c in nonnum:
            if c != state_col and df[c].nunique() >= 0.8 * len(df):
                name_col = c
                break
    return {"numeric": num, "categorical": cat, "state": state_col, "name": name_col}


def profile(df: pd.DataFrame) -> dict:
    miss = df.isna().mean().sort_values(ascending=False)
    return {"rows": len(df), "cols": df.shape[1], "missing_cells": float(df.isna().mean().mean()),
            "duplicates": int(df.duplicated().sum()), "missing": miss,
            "dtypes": df.dtypes.astype(str).value_counts().to_dict()}


def clean(df: pd.DataFrame, drop_duplicates=True, impute="median", max_missing=0.6) -> tuple[pd.DataFrame, list[str]]:
    """Returns (clean df, log of what was done)."""
    log, df = [], df.copy()
    if drop_duplicates:
        n = int(df.duplicated().sum())
        if n:
            df = df.drop_duplicates().reset_index(drop=True)
            log.append(f"Removed {n} duplicate row(s)")
    too_empty = [c for c in df.columns if df[c].isna().mean() > max_missing]
    if too_empty:
        df = df.drop(columns=too_empty)
        log.append(f"Dropped {len(too_empty)} column(s) with > {max_missing:.0%} missing: " + ", ".join(too_empty[:6]))
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if impute in ("median", "mean"):
        k = 0
        for c in numeric:
            m = int(df[c].isna().sum())
            if m:
                df[c] = df[c].fillna(df[c].median() if impute == "median" else df[c].mean())
                k += m
        if k:
            log.append(f"Filled {k} missing numeric value(s) with the column {impute}")
    elif impute == "drop":
        n0 = len(df)
        df = df.dropna(subset=numeric).reset_index(drop=True)
        if len(df) < n0:
            log.append(f"Dropped {n0 - len(df)} row(s) with missing numbers")
    return df, log


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #
def top_correlations(df: pd.DataFrame, cols: list[str], n: int = 8) -> pd.DataFrame:
    c = df[cols].corr()
    pairs = [(a, b, c.loc[a, b]) for i, a in enumerate(cols) for b in cols[i + 1:] if pd.notna(c.loc[a, b])]
    out = pd.DataFrame(pairs, columns=["Variable A", "Variable B", "Correlation"])
    return out.reindex(out["Correlation"].abs().sort_values(ascending=False).index).head(n).round(3)


def auto_cluster(df: pd.DataFrame, cols: list[str], k: int | None = None) -> dict:
    d = df.dropna(subset=cols).reset_index(drop=True)
    X = StandardScaler().fit_transform(d[cols])
    kmax = max(2, min(8, len(d) - 1))
    sweep = []
    for kk in range(2, kmax + 1):
        km = KMeans(n_clusters=kk, n_init=10, random_state=SEED).fit(X)
        sweep.append({"k": kk, "Inertia": km.inertia_, "Silhouette": silhouette_score(X, km.labels_, sample_size=min(len(X), 3000), random_state=SEED)})
    sweep = pd.DataFrame(sweep)
    best_k = int(sweep.loc[sweep["Silhouette"].idxmax(), "k"])
    k = k or best_k
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
    sizes = pd.Series(km.labels_).value_counts()
    rename = {old: f"Cluster {i + 1}" for i, old in enumerate(sizes.index)}      # 1 = largest
    d["Cluster"] = pd.Series(km.labels_).map(rename).values
    pc = PCA(n_components=2, random_state=SEED).fit_transform(X) if len(cols) >= 2 else np.c_[X[:, 0], np.zeros(len(d))]
    d["PC1"], d["PC2"] = pc[:, 0], pc[:, 1]
    prof = d.groupby("Cluster")[cols].mean().sort_index()
    prof["Rows"] = d.groupby("Cluster").size()
    z = pd.DataFrame(X, columns=cols).groupby(d["Cluster"].values).mean()
    desc = {}
    for cl, row in z.iterrows():
        top = row.reindex(row.abs().sort_values(ascending=False).index).head(2)
        desc[cl] = ", ".join(f"{'high' if v > 0 else 'low'} {c}" for c, v in top.items())
    return {"data": d, "profile": prof, "z": z, "describe": desc, "sweep": sweep, "best_k": best_k,
            "k": k, "silhouette": float(silhouette_score(X, km.labels_, sample_size=min(len(X), 3000), random_state=SEED))}


def drivers(df: pd.DataFrame, target: str, cols: list[str]) -> dict:
    feats = [c for c in cols if c != target]
    d = df.dropna(subset=feats + [target])
    if len(d) > 20000:
        d = d.sample(20000, random_state=SEED)
    X, y = d[feats], d[target]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=SEED).fit(Xtr, ytr)
    pred = rf.predict(Xte)
    folds = min(5, max(2, len(d) // 10))
    cv = cross_val_score(RandomForestRegressor(n_estimators=150, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
                         X, y, cv=KFold(folds, shuffle=True, random_state=SEED), scoring="r2")
    perm = permutation_importance(rf, Xte, yte, n_repeats=10, random_state=SEED, n_jobs=-1)
    imp = pd.DataFrame({"Indicator": feats, "Importance": rf.feature_importances_,
                        "Permutation": perm.importances_mean}).sort_values("Importance", ascending=False)
    return {"r2": r2_score(yte, pred), "rmse": float(np.sqrt(mean_squared_error(yte, pred))), "cv_r2": cv.mean(),
            "cv_std": cv.std(), "importance": imp, "actual": yte.values, "pred": pred, "n": len(d), "folds": folds}


def priority_index(df: pd.DataFrame, cols: list[str], lower_is_better: list[str]) -> pd.DataFrame:
    """Composite need score 0-100 (100 = most in need). Each indicator is z-scored; indicators where
    LOWER is better are flipped; the mean 'goodness' is converted to a need percentile."""
    d = df.dropna(subset=cols).copy()
    z = (d[cols] - d[cols].mean()) / d[cols].std(ddof=0).replace(0, 1)
    for c in lower_is_better:
        if c in z:
            z[c] = -z[c]
    d["Priority score"] = ((1 - z.mean(axis=1).rank(pct=True)) * 100).round(1)
    weakest = z.apply(lambda r: ", ".join(r.sort_values().head(2).index), axis=1)
    d["Weakest indicators"] = weakest
    return d.sort_values("Priority score", ascending=False).reset_index(drop=True)


def scheme_gap(df: pd.DataFrame, state_col: str, metric: str, schemes: pd.DataFrame, higher_is_better: bool) -> pd.DataFrame:
    """State average of `metric` next to the number of State-level schemes available in that state."""
    d = df[[state_col, metric]].copy()
    d["State"] = d[state_col].astype(str).map(E.canon_state)
    g = d.groupby("State")[metric].mean().rename("Average").reset_index()
    cnt = schemes[schemes["level"] == "State"].groupby("state").size().rename("State schemes").reset_index()
    cnt["state"] = cnt["state"].map(E.canon_state)
    out = g.merge(cnt, left_on="State", right_on="state", how="left").drop(columns="state")
    out["State schemes"] = out["State schemes"].fillna(0).astype(int)
    need = out["Average"] < out["Average"].median() if higher_is_better else out["Average"] > out["Average"].median()
    few = out["State schemes"] < out["State schemes"].median()
    out["Possible gap"] = np.where(need & few, "⚠️ high need, fewer schemes", np.where(need, "high need", ""))
    out["Central schemes (all states)"] = int((schemes["level"] == "Central").sum())
    return out.sort_values("Average", ascending=higher_is_better).reset_index(drop=True)


def insights(df: pd.DataFrame, cols: list[str], name_col: str | None, state_col: str | None) -> list[str]:
    out = [f"The table has **{len(df):,} rows** and **{len(cols)} numeric indicators**."]
    label = df[name_col] if name_col else pd.Series(df.index.astype(str), index=df.index)
    for c in cols[:3]:
        hi, lo = df[c].idxmax(), df[c].idxmin()
        out.append(f"**{c}** ranges from {df[c].min():.1f} ({label[lo]}) to {df[c].max():.1f} ({label[hi]}); "
                   f"average {df[c].mean():.1f}.")
    if len(cols) >= 2:
        tc = top_correlations(df, cols, 1)
        if len(tc):
            r = tc.iloc[0]
            kind = "positively" if r["Correlation"] > 0 else "negatively"
            out.append(f"Strongest relationship: **{r['Variable A']}** and **{r['Variable B']}** are {kind} "
                       f"correlated (r = {r['Correlation']:.2f}).")
    cv = (df[cols].std() / df[cols].mean().abs().replace(0, np.nan)).dropna().sort_values(ascending=False)
    if len(cv):
        out.append(f"Most uneven indicator across rows: **{cv.index[0]}** (coefficient of variation {cv.iloc[0]:.2f}).")
    z = ((df[cols] - df[cols].mean()) / df[cols].std(ddof=0).replace(0, 1)).abs()
    n_out = int((z > 3).any(axis=1).sum())
    if n_out:
        out.append(f"**{n_out}** row(s) have at least one extreme value (|z| > 3) — worth checking.")
    if state_col:
        out.append(f"A state column (**{state_col}**) was detected, so the scheme-gap view is available.")
    return out
