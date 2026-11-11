"""Report di sintesi: HTML autonomo (con grafici incorporati) ed Excel multi-foglio."""
from __future__ import annotations

import base64
import html
import os
from datetime import datetime

import numpy as np
import pandas as pd

import stats_engine
from cleaner import DataError, EXCEL_MAX_ROWS, excel_safe, fmt_bytes, fmt_int, fmt_pct
from plotting import figure_png


def _gather(df: pd.DataFrame, summary: dict) -> dict:
    cols = pd.DataFrame([{
        "colonna": str(c["name"]), "tipo": c["label"], "dtype": c["dtype"],
        "non nulli": c["non_null"], "nulli": c["nulls"], "% nulli": round(c["pct"], 2),
        "valori unici": c["unique"],
    } for c in summary["columns"]])
    num = stats_engine.describe_numeric(df)
    cat = stats_engine.describe_categorical(df)
    corr = None
    try:
        corr, _ = stats_engine.correlation(df, "pearson")
    except DataError:
        pass
    return {"cols": cols, "num": num, "cat": cat, "corr": corr}


def _ml_frames(ml) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    if ml is None:
        return None
    m = pd.DataFrame([{"metrica": k, "valore": v} for k, v in ml.metrics.items()])
    info = pd.DataFrame([
        {"metrica": "modello", "valore": ml.algorithm_label},
        {"metrica": "target", "valore": ml.target},
        {"metrica": "righe train / test", "valore": f"{ml.n_train} / {ml.n_test}"},
    ])
    return pd.concat([info, m], ignore_index=True), ml.importance.head(30)


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
CSS = """
:root{--bg:#f6f7f9;--card:#fff;--fg:#1f2328;--muted:#5b6470;--line:#e3e6ea;--accent:#2563eb}
@media (prefers-color-scheme:dark){:root{--bg:#16181c;--card:#1f2228;--fg:#e6e8eb;--muted:#9aa3ad;--line:#30343b;--accent:#60a5fa}}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 "Segoe UI",system-ui,sans-serif;background:var(--bg);color:var(--fg)}
main{max-width:1180px;margin:0 auto;padding:32px 20px 64px}
h1{margin:0 0 4px;font-size:26px}h2{margin:36px 0 12px;font-size:18px;border-bottom:1px solid var(--line);padding-bottom:6px}
.sub{color:var(--muted);margin:0 0 24px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.kpi b{display:block;font-size:22px;font-variant-numeric:tabular-nums}.kpi span{color:var(--muted);font-size:12px}
.tbl{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{background:var(--bg);position:sticky;top:0;font-weight:600}
th:first-child,td:first-child{text-align:left}tr:last-child td{border-bottom:0}
img{max-width:100%;background:#fff;border:1px solid var(--line);border-radius:10px}
ol.log{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 12px 12px 40px;font-family:Consolas,monospace;font-size:12.5px}
.warn{color:#b45309}
"""


def _table(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df is None or df.empty:
        return "<p class='sub'>Nessun dato.</p>"
    shown = df.head(max_rows)

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:,.4g}"
        return html.escape(str(v))

    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in shown.columns)
    body = "".join("<tr>" + "".join(f"<td>{fmt(v)}</td>" for v in row) + "</tr>"
                   for row in shown.itertuples(index=False, name=None))
    more = f"<p class='sub'>Mostrate {max_rows} righe su {fmt_int(len(df))}.</p>" if len(df) > max_rows else ""
    return f"<div class='tbl'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>{more}"


def _img(fig) -> str:
    data = base64.b64encode(figure_png(fig)).decode()
    return f"<img alt='grafico' src='data:image/png;base64,{data}'>"


def html_report(path: str, df: pd.DataFrame, summary: dict, source: str, log: list[str],
                ml=None) -> None:
    g = _gather(df, summary)
    kpis = [("Righe", fmt_int(summary["rows"])), ("Colonne", fmt_int(summary["cols"])),
            ("Memoria", fmt_bytes(summary["memory"])), ("Celle nulle", fmt_pct(summary["null_pct"])),
            ("Righe duplicate", "—" if summary["duplicates"] is None else fmt_int(summary["duplicates"]))]
    parts = [
        f"<h1>Report di sintesi</h1><p class='sub'>Sorgente: {html.escape(source)} · "
        f"generato il {datetime.now():%d/%m/%Y %H:%M} con Data Core Analytics</p>",
        "<div class='kpis'>" + "".join(f"<div class='kpi'><b>{v}</b><span>{k}</span></div>"
                                       for k, v in kpis) + "</div>",
        "<h2>Struttura delle colonne</h2>", _table(g["cols"]),
        "<h2>Statistiche descrittive (numeriche)</h2>", _table(g["num"].round(4)),
        "<h2>Variabili categoriche e testuali</h2>", _table(g["cat"]),
    ]
    if g["corr"] is not None:
        parts += ["<h2>Correlazioni (Pearson)</h2>",
                  _img(stats_engine.correlation_figure(g["corr"], "pearson", dark=False))]
    if summary["total_nulls"]:
        parts += ["<h2>Valori mancanti</h2>",
                  _img(stats_engine.chart_figure(df, "missing", dark=False))]
    mlf = _ml_frames(ml)
    if mlf:
        from ml_engine import importance_figure

        parts += ["<h2>Modello di Machine Learning</h2>", _table(mlf[0]),
                  _img(importance_figure(ml, dark=False))]
    parts += ["<h2>Anteprima dei dati</h2>", _table(df.head(20)),
              "<h2>Log delle operazioni</h2>",
              "<ol class='log'>" + "".join(f"<li>{html.escape(e)}</li>" for e in log) + "</ol>"
              if log else "<p class='sub'>Nessuna operazione registrata.</p>"]
    doc = (f"<!doctype html><html lang='it'><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>Report — {html.escape(os.path.basename(source))}</title><style>{CSS}</style>"
           f"</head><body><main>{''.join(parts)}</main></body></html>")
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(doc)
    except OSError as exc:
        raise DataError(f"Impossibile scrivere il report:\n{exc}") from exc


# --------------------------------------------------------------------------- #
# Excel multi-foglio
# --------------------------------------------------------------------------- #
def excel_report(path: str, df: pd.DataFrame, summary: dict, source: str, log: list[str],
                 ml=None) -> str:
    g = _gather(df, summary)
    overview = pd.DataFrame([
        ("Sorgente", source), ("Generato il", f"{datetime.now():%d/%m/%Y %H:%M}"),
        ("Righe", summary["rows"]), ("Colonne", summary["cols"]),
        ("Memoria", fmt_bytes(summary["memory"])), ("Valori nulli", summary["total_nulls"]),
        ("% celle nulle", round(summary["null_pct"], 2)),
        ("Righe duplicate", summary["duplicates"]),
    ], columns=["voce", "valore"])
    note = ""
    data = df
    if len(df) > EXCEL_MAX_ROWS - 1:
        data = df.head(EXCEL_MAX_ROWS - 1)
        note = f" (foglio Dati troncato a {fmt_int(EXCEL_MAX_ROWS - 1)} righe)"
    sheets = [("Riepilogo", overview), ("Colonne", g["cols"]), ("Statistiche", g["num"]),
              ("Categoriche", g["cat"])]
    if g["corr"] is not None:
        sheets.append(("Correlazioni", g["corr"].round(4).reset_index(names="variabile")))
    mlf = _ml_frames(ml)
    if mlf:
        sheets += [("ML metriche", mlf[0]), ("ML importanza", mlf[1])]
    sheets += [("Log", pd.DataFrame({"operazione": log or ["(nessuna)"]})),
               ("Dati", excel_safe(data.iloc[:, :16384]))]
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            for name, frame in sheets:
                frame.to_excel(xw, sheet_name=name, index=False)
                ws = xw.sheets[name]
                ws.freeze_panes = "A2"
                for i, col in enumerate(frame.columns[:60], 1):
                    sample = frame[col].head(200).astype(str).str.len()
                    width = min(60, max(len(str(col)), int(sample.max() if len(sample) else 0)) + 2)
                    ws.column_dimensions[ws.cell(1, i).column_letter].width = width
    except PermissionError as exc:
        raise DataError("Impossibile scrivere il report: il file potrebbe essere aperto in Excel.") from exc
    except OSError as exc:
        raise DataError(f"Impossibile scrivere il report:\n{exc}") from exc
    return note
