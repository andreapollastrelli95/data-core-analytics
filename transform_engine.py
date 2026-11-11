"""Trasformazioni strutturali: join, accodamento (concat) e aggregazioni group-by."""
from __future__ import annotations

import pandas as pd

from cleaner import DataError, OpResult, fmt_int, is_numeric, names, pick
from codegen import lit

JOIN_TYPES = {
    "Left join (tutte le righe del principale)": "left",
    "Right join (tutte le righe del secondario)": "right",
    "Inner join (solo corrispondenze)": "inner",
    "Outer join (tutte le righe)": "outer",
}

AGG_FUNCS = {
    "Somma": "sum", "Media": "mean", "Mediana": "median", "Minimo": "min",
    "Massimo": "max", "Conteggio (non nulli)": "count", "Valori distinti": "nunique",
    "Deviazione std": "std", "Primo": "first", "Ultimo": "last",
}
NUMERIC_ONLY = {"sum", "mean", "median", "std"}
MAX_JOIN_ROWS = 20_000_000


def merge(left: pd.DataFrame, right: pd.DataFrame, how: str, left_on: list, right_on: list,
          right_code: str = "") -> OpResult:
    if not left_on or not right_on:
        raise DataError("Seleziona almeno una colonna chiave per ciascun dataset.")
    if len(left_on) != len(right_on):
        raise DataError("Il numero di chiavi deve essere uguale nei due dataset "
                        f"({len(left_on)} nel principale, {len(right_on)} nel secondario).")
    for lk, rk in zip(left_on, right_on):
        ln, rn = is_numeric(left[lk]), is_numeric(right[rk])
        if ln != rn:
            raise DataError(f"Le chiavi '{lk}' e '{rk}' hanno tipi incompatibili "
                            f"({left[lk].dtype} e {right[rk].dtype}).\n\n"
                            "Converti prima una delle due con «Tipi di dato».")

    # Stima delle righe risultanti per evitare esplosioni many-to-many.
    lc = left.groupby(left_on, dropna=False).size()
    rc = right.groupby(right_on, dropna=False).size()
    rc.index.names = lc.index.names
    estimate = int((lc * rc).dropna().sum())
    if estimate > MAX_JOIN_ROWS:
        raise DataError(f"La join produrrebbe circa {fmt_int(estimate)} righe (relazione "
                        "molti-a-molti sulle chiavi scelte). Verifica le chiavi.")
    try:
        out = left.merge(right, how=how, left_on=left_on, right_on=right_on,
                         suffixes=("", "_dx"), indicator=False)
    except (ValueError, TypeError, KeyError) as exc:
        raise DataError(f"Join non riuscita: {exc}") from exc
    out = out.reset_index(drop=True)
    msg = (f"Join {how.upper()} su {names(left_on)}: {fmt_int(len(left))} → "
           f"{fmt_int(len(out))} righe, {out.shape[1]} colonne")
    code = (f"{right_code}\n" if right_code else "") + (
        f"df = df.merge(df_dx, how={how!r}, left_on={lit(list(left_on))}, "
        f"right_on={lit(list(right_on))}, suffixes=('', '_dx')).reset_index(drop=True)")
    return OpResult(out, msg, code=code)


def concat(top: pd.DataFrame, bottom: pd.DataFrame, add_source: bool = False,
           right_code: str = "") -> OpResult:
    only_top = [c for c in top.columns if c not in bottom.columns]
    only_bottom = [c for c in bottom.columns if c not in top.columns]
    a, b = top, bottom
    if add_source:
        a = top.assign(origine="principale")
        b = bottom.assign(origine="secondario")
    out = pd.concat([a, b], ignore_index=True, sort=False)
    msg = (f"Accodate {fmt_int(len(bottom))} righe dal dataset secondario: "
           f"{fmt_int(len(top))} → {fmt_int(len(out))} righe")
    if only_top or only_bottom:
        msg += f" ({len(only_top) + len(only_bottom)} colonne presenti in un solo dataset, " \
               "riempite con null)"
    if add_source:
        body = ('df = pd.concat([df.assign(origine="principale"), '
                'df_dx.assign(origine="secondario")], ignore_index=True)')
    else:
        body = "df = pd.concat([df, df_dx], ignore_index=True)"
    return OpResult(out, msg, code=(f"{right_code}\n" if right_code else "") + body)


def group_by(df: pd.DataFrame, keys: list, metrics: list[tuple], count_rows: bool = True,
             sort_desc_by: str | None = None) -> OpResult:
    """metrics: lista di (colonna, funzione pandas)."""
    if not keys:
        raise DataError("Seleziona almeno una colonna di raggruppamento.")
    if not metrics and not count_rows:
        raise DataError("Aggiungi almeno una metrica oppure attiva il conteggio delle righe.")
    for col, func in metrics:
        if col in keys:
            raise DataError(f"La colonna '{col}' è già una chiave di raggruppamento.")
        if func in NUMERIC_ONLY and not is_numeric(df[col]):
            label = next(k for k, v in AGG_FUNCS.items() if v == func)
            raise DataError(f"«{label}» richiede una colonna numerica: '{col}' è di tipo "
                            f"{df[col].dtype}.")

    named = {}
    for col, func in metrics:
        name = f"{col}_{func}"
        named[name] = (col, func)
    if count_rows:
        named["n_righe"] = (keys[0], "size")
    try:
        out = df.groupby(keys, dropna=False, observed=True, sort=True).agg(**named).reset_index()
    except (TypeError, ValueError) as exc:
        raise DataError(f"Aggregazione non riuscita: {exc}") from exc
    if sort_desc_by and sort_desc_by in out.columns:
        out = out.sort_values(sort_desc_by, ascending=False, ignore_index=True)

    msg = (f"Group by su {names(keys)}: {fmt_int(len(df))} righe → {fmt_int(len(out))} gruppi, "
           + pick(len(named), "{n} metrica", "{n} metriche"))
    agg_code = ", ".join(f"{k}={lit(v)}" if k.isidentifier() else f"**{{{k!r}: {lit(v)}}}"
                         for k, v in named.items())
    code = (f"df = df.groupby({lit(list(keys))}, dropna=False, observed=True)"
            f".agg({agg_code}).reset_index()")
    if sort_desc_by and sort_desc_by in out.columns:
        code += f"\ndf = df.sort_values({sort_desc_by!r}, ascending=False, ignore_index=True)"
    return OpResult(out, msg, code=code)
