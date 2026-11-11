"""Analisi temporale: riconoscimento date, ricampionamento, scomposizione e previsione."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from cleaner import DataError, OpResult, fmt_int, is_numeric, names
from codegen import lit
from plotting import ACCENT, FORECAST, PALETTE, new_figure, short, themed

FREQS = {"Orario": "h", "Giornaliero": "D", "Settimanale": "W", "Mensile": "MS",
         "Trimestrale": "QS", "Annuale": "YS"}
SEASON = {"h": 24, "D": 7, "W": 52, "MS": 12, "QS": 4, "YS": 1}
AGGS = {"Somma": "sum", "Media": "mean", "Mediana": "median", "Minimo": "min",
        "Massimo": "max", "Conteggio": "count"}
METHODS = {
    "Media mobile": "ma",
    "Exponential Smoothing semplice": "ses",
    "Holt (trend lineare)": "holt",
    "Holt-Winters (trend + stagionalità)": "hw",
}
COUNT_ROWS = "(conteggio righe)"


# --------------------------------------------------------------------------- #
# Riconoscimento e preparazione
# --------------------------------------------------------------------------- #
def _parse_dates(s: pd.Series) -> pd.Series:
    if ptypes.is_datetime64_any_dtype(s.dtype):
        return s.dt.tz_localize(None) if isinstance(s.dtype, pd.DatetimeTZDtype) else s
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(s.astype("string"), format="mixed", dayfirst=True, errors="coerce")


def datetime_candidates(df: pd.DataFrame) -> list:
    found = []
    for c in df.columns:
        s = df[c]
        if ptypes.is_datetime64_any_dtype(s.dtype):
            found.append(c)
            continue
        if not (ptypes.is_object_dtype(s.dtype) or ptypes.is_string_dtype(s.dtype)):
            continue
        sample = s.dropna().astype(str).head(200)
        if len(sample) < 3:
            continue
        if pd.to_numeric(sample, errors="coerce").notna().mean() > 0.5:
            continue  # numeri scritti come testo, non date
        if _parse_dates(sample).notna().mean() >= 0.8:
            found.append(c)
    return found


def build_series(df: pd.DataFrame, date_col, value_col, freq: str, agg: str) -> tuple[pd.Series, dict]:
    if date_col is None:
        raise DataError("Seleziona la colonna temporale.")
    dates = _parse_dates(df[date_col])
    valid = dates.notna()
    if valid.sum() < 3:
        raise DataError(f"La colonna '{date_col}' non contiene date riconoscibili.\n\n"
                        "Convertila prima con «Tipi di dato» indicando il formato.")
    if value_col in (None, COUNT_ROWS):
        values = pd.Series(1.0, index=df.index)
        agg = "sum"
    else:
        if not is_numeric(df[value_col]):
            raise DataError(f"La colonna valore '{value_col}' deve essere numerica.")
        values = df[value_col].astype(float)
    data = pd.Series(values[valid].to_numpy(), index=dates[valid].to_numpy()).sort_index()
    try:
        series = data.resample(freq).agg(agg)
    except (ValueError, TypeError) as exc:
        raise DataError(f"Ricampionamento non riuscito: {exc}") from exc
    if agg in ("sum", "count"):
        empty = data.resample(freq).size() == 0
    else:
        empty = series.isna()
    gaps = int(empty.sum())
    if agg not in ("sum", "count"):
        series = series.interpolate(limit_direction="both")
    series = series.asfreq(freq)
    series.name = value_col if value_col not in (None, COUNT_ROWS) else "n_righe"
    if len(series) < 3:
        raise DataError("Dopo l'aggregazione restano meno di 3 periodi: scegli una frequenza "
                        "più fine.")
    info = {"periods": len(series), "gaps": gaps, "start": series.index[0],
            "end": series.index[-1], "dropped_dates": int((~valid).sum())}
    return series, info


def series_code(date_col, value_col, freq: str, agg: str) -> str:
    v = "_valori = pd.Series(1.0, index=df.index)" if value_col in (None, COUNT_ROWS) else \
        f"_valori = df[{lit(value_col)}]"
    a = "sum" if value_col in (None, COUNT_ROWS) else agg
    lines = [f'_date = pd.to_datetime(df[{lit(date_col)}], format="mixed", dayfirst=True, errors="coerce")',
             v, "serie = pd.Series(_valori.to_numpy(), index=_date).dropna().sort_index()",
             f"serie = serie[serie.index.notna()].resample({freq!r}).{a}()"]
    if a not in ("sum", "count"):
        lines.append("serie = serie.interpolate(limit_direction='both')")
    return "\n".join(lines)


def resample_dataset(df: pd.DataFrame, date_col, value_cols: list, freq: str, agg: str) -> OpResult:
    if not value_cols:
        raise DataError("Seleziona almeno una colonna valore numerica.")
    dates = _parse_dates(df[date_col])
    bad = [c for c in value_cols if not is_numeric(df[c])]
    if bad:
        raise DataError(f"Colonne non numeriche: {names(bad)}")
    data = df[value_cols].astype(float).set_index(dates).loc[lambda d: d.index.notna()].sort_index()
    out = data.resample(freq).agg(agg).reset_index().rename(columns={"index": date_col})
    out.columns = [date_col, *value_cols]
    label = next(k for k, v in FREQS.items() if v == freq).lower()
    msg = (f"Serie aggregata ({label}, {agg}) su '{date_col}': {fmt_int(len(df))} righe → "
           f"{fmt_int(len(out))} periodi")
    code = (f'_d = pd.to_datetime(df[{lit(date_col)}], format="mixed", dayfirst=True, errors="coerce")\n'
            f"df = df[{lit(list(value_cols))}].set_index(_d).sort_index().resample({freq!r})"
            f".{agg}().rename_axis({lit(date_col)}).reset_index()")
    return OpResult(out, msg, code=code)


# --------------------------------------------------------------------------- #
# Scomposizione
# --------------------------------------------------------------------------- #
def decompose(series: pd.Series, model: str, period: int):
    from statsmodels.tsa.seasonal import seasonal_decompose

    if period < 2:
        raise DataError("Il periodo stagionale deve essere almeno 2.")
    if len(series) < 2 * period:
        raise DataError(f"Per la scomposizione servono almeno 2 cicli completi: "
                        f"{2 * period} periodi (ce ne sono {len(series)}).\n\n"
                        "Riduci il periodo stagionale o usa una frequenza più fine.")
    if model == "multiplicative" and (series <= 0).any():
        raise DataError("Il modello moltiplicativo richiede valori tutti positivi: "
                        "usa il modello additivo.")
    try:
        return seasonal_decompose(series, model=model, period=period, extrapolate_trend="freq")
    except ValueError as exc:
        raise DataError(f"Scomposizione non riuscita: {exc}") from exc


def decomposition_figure(result, dark: bool):
    with themed(dark):
        fig = new_figure(dark, size=(9, 7))
        parts = [("Serie osservata", result.observed, ACCENT), ("Trend", result.trend, PALETTE[2]),
                 ("Stagionalità", result.seasonal, PALETTE[4]), ("Residui", result.resid, PALETTE[3])]
        axes = fig.subplots(4, 1, sharex=True)
        for ax, (title, data, color) in zip(axes, parts):
            if title == "Residui":
                ax.scatter(data.index, data.values, s=6, color=color)
                ax.axhline(0 if result.resid.mean() < 0.5 else 1, color="gray", lw=0.8)
            else:
                ax.plot(data.index, data.values, color=color, lw=1.3)
            ax.set_title(title, loc="left", fontsize=10)
    return fig


def decompose_code(model: str, period: int) -> str:
    return (f"scomp = seasonal_decompose(serie, model={model!r}, period={period}, "
            "extrapolate_trend='freq')\nscomp.plot()\nplt.show()")


def seasonal_strength(result) -> tuple[float, float]:
    resid = result.resid.dropna()
    var_r = resid.var()
    st = max(0.0, 1 - var_r / (result.seasonal + result.resid).dropna().var())
    tr = max(0.0, 1 - var_r / (result.trend + result.resid).dropna().var())
    return tr, st


# --------------------------------------------------------------------------- #
# Previsione
# --------------------------------------------------------------------------- #
def _fit_predict(train: pd.Series, method: str, horizon: int, window: int, season: int):
    """Restituisce (valori adattati in-sample, previsione di `horizon` passi)."""
    if method == "ma":
        fitted = train.rolling(window, min_periods=1).mean().shift(1)
        return fitted, np.repeat(train.iloc[-window:].mean(), horizon)

    from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == "ses":
            model = SimpleExpSmoothing(train, initialization_method="estimated").fit()
        elif method == "holt":
            model = ExponentialSmoothing(train, trend="add",
                                         initialization_method="estimated").fit()
        else:
            model = ExponentialSmoothing(train, trend="add", seasonal="add",
                                         seasonal_periods=season,
                                         initialization_method="estimated").fit()
        return model.fittedvalues, np.asarray(model.forecast(horizon))


def forecast(series: pd.Series, method: str, horizon: int, window: int = 3,
             season: int = 12) -> dict:
    if horizon < 1:
        raise DataError("L'orizzonte di previsione deve essere almeno 1 periodo.")
    s = series.astype(float)
    if s.isna().any():
        s = s.interpolate(limit_direction="both")
    n = len(s)
    if method == "ma" and window < 1:
        raise DataError("La finestra della media mobile deve essere almeno 1.")
    if method == "hw":
        if season < 2:
            raise DataError("Holt-Winters richiede un periodo stagionale di almeno 2.")
        if n < 2 * season + 2:
            raise DataError(f"Holt-Winters con stagionalità {season} richiede almeno "
                            f"{2 * season + 2} periodi (ce ne sono {n}).\n\n"
                            "Usa Holt, riduci la stagionalità o aggrega a frequenza più fine.")
    if n < 6:
        raise DataError("Servono almeno 6 periodi per stimare e validare una previsione.")

    # Validazione su una finestra finale (holdout) con il modello addestrato sul resto.
    holdout = max(1, min(horizon, n // 5))
    train, test = s.iloc[:-holdout], s.iloc[-holdout:]
    metrics = {}
    try:
        if method != "hw" or len(train) >= 2 * season + 2:
            _, pred_test = _fit_predict(train, method, holdout, window, season)
            err = test.to_numpy() - pred_test
            nz = test.to_numpy() != 0
            metrics = {"MAE": float(np.mean(np.abs(err))),
                       "RMSE": float(np.sqrt(np.mean(err ** 2))),
                       "MAPE %": float(np.mean(np.abs(err[nz] / test.to_numpy()[nz])) * 100)
                       if nz.any() else float("nan"),
                       "periodi di validazione": holdout}
        fitted, future = _fit_predict(s, method, horizon, window, season)
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise DataError(f"Il modello non è riuscito a stimare la serie: {exc}") from exc

    freq = s.index.freq or pd.infer_freq(s.index)
    index = pd.date_range(s.index[-1], periods=horizon + 1, freq=freq)[1:]
    resid_std = float(np.nanstd(s.to_numpy() - np.asarray(fitted, dtype=float)))
    steps = np.sqrt(np.arange(1, horizon + 1))
    fc = pd.DataFrame({"periodo": index, "previsione": future,
                       "limite_inferiore_95": future - 1.96 * resid_std * steps,
                       "limite_superiore_95": future + 1.96 * resid_std * steps})
    return {"series": s, "fitted": pd.Series(np.asarray(fitted, dtype=float), index=s.index),
            "forecast": fc, "metrics": metrics, "holdout": holdout}


def forecast_figure(res: dict, method_label: str, dark: bool):
    s, fc = res["series"], res["forecast"]
    with themed(dark):
        fig = new_figure(dark, size=(9, 4.8))
        ax = fig.add_subplot()
        ax.plot(s.index, s.values, color=ACCENT, lw=1.4, label="storico")
        ax.plot(res["fitted"].index, res["fitted"].values, color="gray", lw=1, ls="--",
                label="adattamento del modello")
        ax.plot(fc["periodo"], fc["previsione"], color=FORECAST, lw=2, label="previsione")
        ax.fill_between(fc["periodo"], fc["limite_inferiore_95"], fc["limite_superiore_95"],
                        color=FORECAST, alpha=0.18, label="intervallo 95% (approssimato)")
        if res["holdout"] and res["metrics"]:
            ax.axvspan(s.index[-res["holdout"]], s.index[-1], color="gray", alpha=0.1,
                       label="periodo di validazione")
        ax.set_title(f"Previsione — {method_label} — {short(s.name, 30)}")
        ax.legend(loc="upper left", fontsize=8)
    return fig


def series_figure(series: pd.Series, dark: bool, rolling: int | None = None):
    with themed(dark):
        fig = new_figure(dark, size=(9, 4.4))
        ax = fig.add_subplot()
        ax.plot(series.index, series.values, color=ACCENT, lw=1.4, label=str(series.name))
        if rolling and rolling > 1 and len(series) > rolling:
            ax.plot(series.index, series.rolling(rolling).mean(), color=FORECAST, lw=1.8,
                    label=f"media mobile ({rolling} periodi)")
        ax.set_title(f"Serie temporale — {short(series.name, 40)}")
        ax.legend(loc="upper left", fontsize=8)
    return fig


def forecast_code(method: str, horizon: int, window: int, season: int) -> str:
    if method == "ma":
        return (f"_livello = serie.iloc[-{window}:].mean()\n"
                f"_idx = pd.date_range(serie.index[-1], periods={horizon + 1}, freq=serie.index.freq)[1:]\n"
                "previsione = pd.Series(_livello, index=_idx)\nprint(previsione)")
    model = {"ses": "SimpleExpSmoothing(serie, initialization_method='estimated')",
             "holt": "ExponentialSmoothing(serie, trend='add', initialization_method='estimated')",
             "hw": (f"ExponentialSmoothing(serie, trend='add', seasonal='add', "
                    f"seasonal_periods={season}, initialization_method='estimated')")}[method]
    return f"modello = {model}.fit()\nprevisione = modello.forecast({horizon})\nprint(previsione)"
