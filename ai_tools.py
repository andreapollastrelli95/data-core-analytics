"""Strumenti che l'assistente IA può chiamare: ogni strumento esegue un motore dell'app.

Il modello non descrive più che cosa fare: sceglie uno strumento, l'app lo esegue sui dati
caricati e restituisce al modello il risultato calcolato. Tutti gli strumenti sono di sola
lettura (il dataset non viene mai modificato) e restituiscono, oltre al risultato, il codice
pandas/SQL equivalente, che finisce nel registro delle operazioni.

Il modulo non dipende dall'interfaccia grafica né dalla rete.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import cleaner
import sql_engine
import stats_engine
import transform_engine
import ts_engine
from cleaner import DataError
from codegen import lit

MAX_ROWS_TO_MODEL = 40        # righe di una tabella inviate al modello (il resto resta in locale)
MAX_COLS_TO_MODEL = 14
MAX_TEXT_TO_MODEL = 4000


@dataclass
class ToolContext:
    df: pd.DataFrame
    df2: pd.DataFrame | None = None       # dataset secondario (tabella SQL «df2»)
    summary: dict | None = None


@dataclass
class ToolResult:
    payload: dict                         # ciò che viene restituito al modello (JSON)
    description: str = ""                 # riga per il registro
    code: str = ""                        # codice Python equivalente
    imports: tuple = ()
    table: pd.DataFrame | None = None     # risultato completo (per report e revisore)
    error: str = ""
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Serializzazione
# --------------------------------------------------------------------------- #
def clean_value(v):
    """Converte un valore pandas/numpy in un tipo JSON, arrotondando i decimali."""
    if v is None or v is pd.NaT:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        if f == 0:
            return 0.0
        digits = 6 - int(math.floor(math.log10(abs(f)))) - 1
        return round(f, max(digits, 2))
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        ts = pd.Timestamp(v)
        return ts.strftime("%Y-%m-%d") if ts == ts.normalize() else ts.isoformat(sep=" ")
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    text = str(v)
    return text if len(text) <= 120 else text[:117] + "…"


def table_payload(df: pd.DataFrame, max_rows: int = MAX_ROWS_TO_MODEL) -> dict:
    cols = list(df.columns)[:MAX_COLS_TO_MODEL]
    shown = df[cols].head(max_rows)
    out = {"colonne": [str(c) for c in cols],
           "righe": [[clean_value(v) for v in row] for row in shown.itertuples(index=False, name=None)],
           "righe_totali": int(len(df))}
    if len(df) > max_rows:
        out["nota"] = f"Mostrate le prime {max_rows} righe su {len(df)}."
    if df.shape[1] > len(cols):
        out["nota_colonne"] = f"Mostrate {len(cols)} colonne su {df.shape[1]}."
    return out


def resolve(df: pd.DataFrame, name):
    """Nome di colonna indicato dal modello → colonna reale del DataFrame."""
    if name is None or name == "":
        return None
    for c in df.columns:
        if str(c) == str(name):
            return c
    low = str(name).strip().lower()
    matches = [c for c in df.columns if str(c).strip().lower() == low]
    if len(matches) == 1:
        return matches[0]
    raise DataError(f"La colonna '{name}' non esiste. Colonne disponibili: "
                    + ", ".join(str(c) for c in df.columns[:60]))


def resolve_many(df: pd.DataFrame, names) -> list:
    if not names:
        return []
    if isinstance(names, str):
        names = [names]
    return [resolve(df, n) for n in names]


# --------------------------------------------------------------------------- #
# Strumenti
# --------------------------------------------------------------------------- #
def tool_profile(ctx: ToolContext) -> ToolResult:
    df = ctx.df
    s = ctx.summary or cleaner.summarize(df)
    cols = pd.DataFrame([{"colonna": str(c["name"]), "tipo": c["label"], "nulli": c["nulls"],
                          "% nulli": round(c["pct"], 2), "distinti": c["unique"]}
                         for c in s["columns"]])
    payload = {"righe": s["rows"], "colonne_totali": s["cols"], "celle_nulle": s["total_nulls"],
               "percentuale_celle_nulle": round(s["null_pct"], 2),
               "righe_duplicate": s["duplicates"], "struttura": table_payload(cols, 80)}
    code = ("print(df.shape)\nprint(df.dtypes)\nprint(df.isna().sum())\n"
            "print('Righe duplicate:', df.duplicated().sum())")
    return ToolResult(payload, "Profilo del dataset (righe, tipi, nulli, duplicati)", code, table=cols)


def tool_sql(ctx: ToolContext, query: str = "") -> ToolResult:
    tables = {"df": ctx.df}
    if ctx.df2 is not None:
        tables["df2"] = ctx.df2
    result = sql_engine.run_query(query, tables, sandbox=True)
    payload = {"risultato": table_payload(result)}
    code = sql_engine.query_code(query, list(tables), target="risultato") + "\nprint(risultato)"
    short_q = " ".join(query.split())
    return ToolResult(payload, f"Query SQL: {short_q[:110]}{'…' if len(short_q) > 110 else ''}", code,
                      ("import duckdb",), table=result, meta={"query": query})


def tool_aggregate(ctx: ToolContext, per=None, metriche=None, ordina_per: str = "",
                   limite: int = 0) -> ToolResult:
    df = ctx.df
    keys = resolve_many(df, per)
    metrics = []
    for m in metriche or []:
        func = str(m.get("funzione", "sum")).lower()
        if func not in transform_engine.AGG_FUNCS.values():
            raise DataError(f"Funzione '{func}' non valida. Usa una tra: "
                            + ", ".join(transform_engine.AGG_FUNCS.values()))
        metrics.append((resolve(df, m.get("colonna")), func))
    res = transform_engine.group_by(df, keys, metrics, count_rows=True,
                                    sort_desc_by=ordina_per or None)
    out = res.df
    code = res.code.replace("df = df.groupby", "risultato = df.groupby").replace(
        "df = df.sort_values", "risultato = risultato.sort_values") + "\nprint(risultato)"
    shown = out.head(int(limite)) if limite and int(limite) > 0 else out
    payload = {"risultato": table_payload(shown), "gruppi_totali": int(len(out))}
    return ToolResult(payload, res.message, code, table=out,
                      meta={"keys": keys, "metrics": metrics})


def tool_stats(ctx: ToolContext, colonne=None) -> ToolResult:
    df = ctx.df
    cols = resolve_many(df, colonne)
    sub = df[cols] if cols else df
    num = stats_engine.describe_numeric(sub)
    cat = stats_engine.describe_categorical(sub)
    payload = {}
    if not num.empty:
        keep = ["colonna", "conteggio", "nulli", "media", "dev_std", "min", "p25", "mediana",
                "p75", "max", "skewness"]
        payload["numeriche"] = table_payload(num[keep])
    if not cat.empty:
        payload["categoriche"] = table_payload(cat.drop(columns=["memoria_KB"]))
    table = num if not num.empty else cat
    return ToolResult(payload, "Statistiche descrittive"
                      + (f" di {cleaner.names(cols)}" if cols else ""),
                      stats_engine.DESCRIBE_CODE, table=table)


def tool_correlation(ctx: ToolContext, metodo: str = "pearson", colonne=None) -> ToolResult:
    metodo = (metodo or "pearson").lower()
    if metodo not in ("pearson", "spearman", "kendall"):
        raise DataError("Metodo non valido: usa pearson, spearman o kendall.")
    cols = resolve_many(ctx.df, colonne)
    _corr, pairs = stats_engine.correlation(ctx.df, metodo, cols or None)
    return ToolResult({"coppie": table_payload(pairs, 25)},
                      f"Correlazioni ({metodo}) tra le variabili numeriche",
                      stats_engine.correlation_code(metodo), table=pairs)


TESTS = {"normalita": "normality", "ttest": "ttest", "anova": "anova", "chi2": "chi2",
         "correlazione": "corr"}


def tool_test(ctx: ToolContext, test: str = "", variabile1: str = "", variabile2: str = "",
              alpha: float = 0.05) -> ToolResult:
    key = TESTS.get(str(test).lower())
    if key is None:
        raise DataError("Test non valido: usa " + ", ".join(TESTS) + ".")
    v1 = resolve(ctx.df, variabile1)
    v2 = resolve(ctx.df, variabile2) if variabile2 else None
    alpha = float(alpha or 0.05)
    text = stats_engine.run_test(ctx.df, key, v1, v2, alpha)
    return ToolResult({"esito": text[:MAX_TEXT_TO_MODEL]}, f"Test d'ipotesi «{test}» su "
                      + cleaner.names([c for c in (v1, v2) if c is not None]),
                      stats_engine.test_code(key, v1, v2, alpha), ("from scipy import stats",),
                      meta={"text": text})


def tool_outliers(ctx: ToolContext, colonne=None, metodo: str = "iqr", soglia: float = 0) -> ToolResult:
    metodo = (metodo or "iqr").lower()
    if metodo not in ("iqr", "zscore"):
        raise DataError("Metodo non valido: usa iqr oppure zscore.")
    soglia = float(soglia) if soglia else (1.5 if metodo == "iqr" else 3.0)
    cols = resolve_many(ctx.df, colonne)
    table, mask = stats_engine.detect_outliers(ctx.df, cols, metodo, soglia)
    payload = {"per_colonna": table_payload(table), "righe_con_almeno_un_outlier": int(mask.sum())}
    code = (f"# Outlier con metodo {metodo} (soglia {soglia})\n"
            "num = df.select_dtypes('number')\n"
            + ("q1, q3 = num.quantile(.25), num.quantile(.75)\n"
               f"fuori = (num < q1 - {soglia} * (q3 - q1)) | (num > q3 + {soglia} * (q3 - q1))\n"
               if metodo == "iqr" else
               f"fuori = ((num - num.mean()) / num.std()).abs() > {soglia}\n")
            + "print(fuori.sum())")
    return ToolResult(payload, f"Ricerca outlier ({metodo}, soglia {soglia})", code, table=table)


FREQ_CODES = {"oraria": "h", "giornaliera": "D", "settimanale": "W", "mensile": "MS",
              "trimestrale": "QS", "annuale": "YS"}


def _series(ctx: ToolContext, colonna_data, colonna_valore, frequenza, aggregazione):
    freq = FREQ_CODES.get(str(frequenza or "mensile").lower())
    if freq is None:
        raise DataError("Frequenza non valida: usa " + ", ".join(FREQ_CODES) + ".")
    agg = str(aggregazione or "sum").lower()
    if agg not in ts_engine.AGGS.values():
        raise DataError("Aggregazione non valida: usa " + ", ".join(ts_engine.AGGS.values()) + ".")
    date_col = resolve(ctx.df, colonna_data)
    value_col = resolve(ctx.df, colonna_valore) if colonna_valore else None
    series, info = ts_engine.build_series(ctx.df, date_col, value_col, freq, agg)
    return series, info, date_col, value_col, freq, agg


def tool_series(ctx: ToolContext, colonna_data: str = "", colonna_valore: str = "",
                frequenza: str = "mensile", aggregazione: str = "sum") -> ToolResult:
    series, info, date_col, value_col, freq, agg = _series(ctx, colonna_data, colonna_valore,
                                                           frequenza, aggregazione)
    table = pd.DataFrame({"periodo": series.index, str(series.name): series.to_numpy()})
    payload = {"serie": table_payload(table, 60), "periodi": info["periods"],
               "periodi_senza_dati": info["gaps"], "date_non_valide_scartate": info["dropped_dates"]}
    code = ts_engine.series_code(date_col, value_col, freq, agg) + "\nprint(serie)"
    return ToolResult(payload, f"Serie temporale {frequenza} di "
                      f"{value_col if value_col is not None else 'conteggio righe'} ({agg})",
                      code, table=table)


METHOD_CODES = {"media_mobile": "ma", "exponential_smoothing": "ses", "holt": "holt",
                "holt_winters": "hw"}


def tool_forecast(ctx: ToolContext, colonna_data: str = "", colonna_valore: str = "",
                  frequenza: str = "mensile", aggregazione: str = "sum", metodo: str = "holt",
                  orizzonte: int = 3) -> ToolResult:
    method = METHOD_CODES.get(str(metodo or "holt").lower())
    if method is None:
        raise DataError("Metodo non valido: usa " + ", ".join(METHOD_CODES) + ".")
    series, _info, date_col, value_col, freq, agg = _series(ctx, colonna_data, colonna_valore,
                                                            frequenza, aggregazione)
    horizon = max(1, min(int(orizzonte or 3), 60))
    season = ts_engine.SEASON.get(freq, 12)
    res = ts_engine.forecast(series, method, horizon, window=3, season=season)
    fc = res["forecast"]
    payload = {"previsione": table_payload(fc, 60),
               "errore_su_dati_tenuti_da_parte": {k: clean_value(v) for k, v in res["metrics"].items()},
               "periodi_storici": int(len(series)),
               "ultimo_valore_osservato": clean_value(series.iloc[-1])}
    code = (ts_engine.series_code(date_col, value_col, freq, agg) + "\n"
            + ts_engine.forecast_code(method, horizon, 3, season))
    imports = () if method == "ma" else (
        "from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing",)
    return ToolResult(payload, f"Previsione {metodo} a {horizon} periodi di "
                      f"{value_col if value_col is not None else 'conteggio righe'}", code, imports,
                      table=fc, meta={"metrics": res["metrics"]})


# --------------------------------------------------------------------------- #
# Registro degli strumenti (nome, descrizione, schema JSON dei parametri, funzione)
# --------------------------------------------------------------------------- #
def _schema(props: dict | None = None, required: list | None = None) -> dict:
    return {"type": "object", "properties": props or {}, "required": required or []}


_STR = {"type": "string"}
_COLS = {"type": "array", "items": {"type": "string"}}

TOOLS = [
    {"name": "profilo_dataset", "fn": tool_profile,
     "description": "Struttura e qualità del dataset: righe, colonne, tipo e valori nulli per "
                    "colonna, righe duplicate. Usalo per primo se non conosci i dati.",
     "parameters": _schema()},
    {"name": "esegui_sql", "fn": tool_sql,
     "description": "Esegue una query SQL di sola lettura (dialetto DuckDB) sul dataset. La tabella "
                    "principale si chiama df; se è caricato un dataset secondario si chiama df2. "
                    "Metti tra doppi apici i nomi di colonna con spazi o simboli. Restituisce al "
                    "massimo 40 righe: aggrega o usa LIMIT.",
     "parameters": _schema({"query": {"type": "string", "description": "Istruzione SELECT."}},
                           ["query"])},
    {"name": "aggrega", "fn": tool_aggregate,
     "description": "Raggruppa per una o più colonne e calcola metriche (sum, mean, median, min, "
                    "max, count, nunique, std). Aggiunge sempre n_righe. I nomi delle metriche nel "
                    "risultato sono colonna_funzione (es. Fatturato_sum).",
     "parameters": _schema({
         "per": {**_COLS, "description": "Colonne di raggruppamento."},
         "metriche": {"type": "array", "items": _schema(
             {"colonna": _STR, "funzione": _STR}, ["colonna", "funzione"])},
         "ordina_per": {"type": "string", "description": "Colonna del risultato per l'ordine "
                                                         "decrescente, es. Fatturato_sum."},
         "limite": {"type": "integer", "description": "Numero massimo di gruppi da restituire."}},
         ["per"])},
    {"name": "statistiche", "fn": tool_stats,
     "description": "Statistiche descrittive (media, mediana, percentili, deviazione standard, "
                    "valori più frequenti) di tutte le colonne o di quelle indicate.",
     "parameters": _schema({"colonne": _COLS})},
    {"name": "correlazioni", "fn": tool_correlation,
     "description": "Correlazioni tra variabili numeriche con p-value, ordinate per forza.",
     "parameters": _schema({"metodo": {"type": "string", "enum": ["pearson", "spearman", "kendall"]},
                            "colonne": _COLS})},
    {"name": "test_ipotesi", "fn": tool_test,
     "description": "Test statistico: normalita (1 variabile numerica), ttest (numerica + gruppo "
                    "a 2 livelli), anova (numerica + gruppo), chi2 (2 categoriche), correlazione "
                    "(2 numeriche).",
     "parameters": _schema({"test": {"type": "string", "enum": list(TESTS)}, "variabile1": _STR,
                            "variabile2": _STR, "alpha": {"type": "number"}},
                           ["test", "variabile1"])},
    {"name": "outlier", "fn": tool_outliers,
     "description": "Conta i valori anomali nelle colonne numeriche con metodo iqr o zscore.",
     "parameters": _schema({"colonne": _COLS, "metodo": {"type": "string", "enum": ["iqr", "zscore"]},
                            "soglia": {"type": "number"}})},
    {"name": "serie_temporale", "fn": tool_series,
     "description": "Aggrega una colonna numerica nel tempo (frequenza oraria, giornaliera, "
                    "settimanale, mensile, trimestrale, annuale). Senza colonna_valore conta le righe.",
     "parameters": _schema({"colonna_data": _STR, "colonna_valore": _STR,
                            "frequenza": {"type": "string", "enum": list(FREQ_CODES)},
                            "aggregazione": {"type": "string",
                                             "enum": list(ts_engine.AGGS.values())}},
                           ["colonna_data"])},
    {"name": "previsione", "fn": tool_forecast,
     "description": "Previsione di una serie temporale con validazione su dati tenuti da parte "
                    "(MAE, RMSE, MAPE). Metodi: media_mobile, exponential_smoothing, holt, "
                    "holt_winters.",
     "parameters": _schema({"colonna_data": _STR, "colonna_valore": _STR,
                            "frequenza": {"type": "string", "enum": list(FREQ_CODES)},
                            "aggregazione": {"type": "string",
                                             "enum": list(ts_engine.AGGS.values())},
                            "metodo": {"type": "string", "enum": list(METHOD_CODES)},
                            "orizzonte": {"type": "integer"}}, ["colonna_data", "colonna_valore"])},
]
_BY_NAME = {t["name"]: t for t in TOOLS}


def declarations() -> list[dict]:
    """Elenco degli strumenti nel formato da passare al modello (senza le funzioni Python)."""
    return [{k: t[k] for k in ("name", "description", "parameters")} for t in TOOLS]


def run_tool(name: str, args: dict | None, ctx: ToolContext) -> ToolResult:
    """Esegue uno strumento; gli errori sui dati tornano al modello invece di interrompere."""
    spec = _BY_NAME.get(name)
    if spec is None:
        msg = f"Strumento '{name}' inesistente. Strumenti disponibili: " + ", ".join(_BY_NAME)
        return ToolResult({"errore": msg}, f"Strumento sconosciuto: {name}", error=msg)
    allowed = set(spec["parameters"]["properties"])
    kwargs = {k: v for k, v in (args or {}).items() if k in allowed}
    try:
        return spec["fn"](ctx, **kwargs)
    except DataError as exc:
        msg = str(exc)
    except Exception as exc:  # noqa: BLE001 - l'errore torna al modello, l'app non si ferma
        msg = f"{type(exc).__name__}: {exc}"
    return ToolResult({"errore": msg[:MAX_TEXT_TO_MODEL]}, f"{name}: errore", error=msg)


def call_label(name: str, args: dict | None) -> str:
    """Rappresentazione leggibile di una chiamata, per il registro."""
    parts = []
    for k, v in (args or {}).items():
        text = lit(v) if not isinstance(v, str) else repr(" ".join(v.split()))
        parts.append(f"{k}={text if len(text) <= 160 else text[:157] + '…'}")
    return f"{name}({', '.join(parts)})"
