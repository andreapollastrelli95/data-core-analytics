"""Auto-EDA: statistiche descrittive, correlazioni, outlier, test d'ipotesi e grafici."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from pandas.api import types as ptypes
from scipy import stats

from cleaner import DataError, OpResult, fmt_int, fmt_pct, is_numeric, names, pick
from codegen import lit
from plotting import ACCENT, new_figure, short, themed

MAX_SCATTER_POINTS = 5000


def numeric_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if is_numeric(df[c])]


def _require_numeric(df: pd.DataFrame, col) -> pd.Series:
    if col not in df.columns:
        raise DataError(f"Colonna '{col}' non trovata.")
    if not is_numeric(df[col]):
        raise DataError(f"La colonna '{col}' non è numerica ({df[col].dtype}).\n\n"
                        "Convertila con «Tipi di dato» oppure scegli un'altra colonna.")
    return df[col].astype(float)


# --------------------------------------------------------------------------- #
# Statistiche descrittive
# --------------------------------------------------------------------------- #
def describe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    cols = numeric_columns(df)
    if not cols:
        return pd.DataFrame()
    mem = df.memory_usage(deep=True, index=False)
    rows = []
    for c in cols:
        s = df[c].astype(float)
        v = s.dropna()
        q = v.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]) if len(v) else pd.Series(
            [np.nan] * 7, index=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        mean = v.mean()
        rows.append({
            "colonna": c, "conteggio": len(v), "nulli": int(s.isna().sum()),
            "media": mean, "dev_std": v.std(), "min": v.min(),
            "p1": q[0.01], "p5": q[0.05], "p25": q[0.25], "mediana": q[0.5],
            "p75": q[0.75], "p95": q[0.95], "p99": q[0.99], "max": v.max(),
            "IQR": q[0.75] - q[0.25], "skewness": v.skew(), "kurtosis": v.kurt(),
            "CV": (v.std() / mean) if mean else np.nan, "memoria_KB": mem[c] / 1024,
        })
    return pd.DataFrame(rows)


def describe_categorical(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if not is_numeric(df[c])]
    mem = df.memory_usage(deep=True, index=False)
    rows = []
    for c in cols:
        s = df[c]
        try:
            vc = s.value_counts(dropna=True)
        except TypeError:
            vc = s.astype(str).value_counts(dropna=True)
        rows.append({
            "colonna": c, "tipo": str(s.dtype), "conteggio": int(s.notna().sum()),
            "nulli": int(s.isna().sum()), "distinti": len(vc),
            "più frequente": short(vc.index[0], 40) if len(vc) else "",
            "frequenza": int(vc.iloc[0]) if len(vc) else 0,
            "% più frequente": (vc.iloc[0] / s.notna().sum() * 100) if len(vc) else 0.0,
            "memoria_KB": mem[c] / 1024,
        })
    return pd.DataFrame(rows)


DESCRIBE_CODE = (
    "stat = df.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T\n"
    "num = df.select_dtypes('number')\n"
    "stat['skewness'] = num.skew()\n"
    "stat['kurtosis'] = num.kurt()\n"
    "stat['memoria_KB'] = df.memory_usage(deep=True, index=False) / 1024\n"
    "print(stat)"
)


# --------------------------------------------------------------------------- #
# Correlazioni
# --------------------------------------------------------------------------- #
def correlation(df: pd.DataFrame, method: str = "pearson", columns: list | None = None,
                top: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = columns or numeric_columns(df)
    cols = [c for c in cols if is_numeric(df[c]) and df[c].nunique(dropna=True) > 1]
    if len(cols) < 2:
        raise DataError("Servono almeno due colonne numeriche non costanti per calcolare "
                        "le correlazioni.")
    data = df[cols].astype(float)
    corr = data.corr(method=method)
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r):
                pairs.append((a, b, r))
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    rows = []
    for a, b, r in pairs[:top]:
        sub = data[[a, b]].dropna()
        p = np.nan
        if len(sub) >= 3:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fn = stats.pearsonr if method == "pearson" else (
                    stats.spearmanr if method == "spearman" else stats.kendalltau)
                p = fn(sub[a], sub[b])[1]
        rows.append({"variabile A": a, "variabile B": b, "coefficiente": r,
                     "p-value": p, "n": len(sub), "forza": _corr_strength(r)})
    return corr, pd.DataFrame(rows)


def _corr_strength(r: float) -> str:
    a = abs(r)
    level = ("molto forte" if a >= .8 else "forte" if a >= .6 else "moderata" if a >= .4
             else "debole" if a >= .2 else "trascurabile")
    return f"{level} {'positiva' if r > 0 else 'negativa' if r < 0 else ''}".strip()


def correlation_figure(corr: pd.DataFrame, method: str, dark: bool):
    n = len(corr)
    with themed(dark):
        fig = new_figure(dark, size=(max(6, min(12, 0.55 * n + 3)), max(4.8, min(10, 0.5 * n + 2))))
        ax = fig.add_subplot()
        labels = [short(c, 18) for c in corr.columns]
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, cmap="RdBu_r", vmin=-1, vmax=1, center=0, ax=ax,
                    annot=n <= 14, fmt=".2f", annot_kws={"size": 8}, square=n <= 14,
                    linewidths=0.4, linecolor="#242424" if dark else "#ffffff",
                    xticklabels=labels, yticklabels=labels, cbar_kws={"shrink": 0.75})
        ax.set_title(f"Matrice di correlazione ({method.capitalize()})")
        ax.grid(False)
        ax.tick_params(axis="x", rotation=45)
        for t in ax.get_xticklabels():
            t.set_ha("right")
    return fig


def correlation_code(method: str) -> str:
    return (f"corr = df.select_dtypes('number').corr(method={method!r})\n"
            "print(corr.round(3))")


# --------------------------------------------------------------------------- #
# Outlier
# --------------------------------------------------------------------------- #
def outlier_bounds(s: pd.Series, method: str, threshold: float) -> tuple[float, float]:
    v = s.dropna().astype(float)
    if method == "iqr":
        q1, q3 = v.quantile(0.25), v.quantile(0.75)
        iqr = q3 - q1
        return q1 - threshold * iqr, q3 + threshold * iqr
    mean, std = v.mean(), v.std()
    return mean - threshold * std, mean + threshold * std


def detect_outliers(df: pd.DataFrame, columns: list, method: str = "iqr",
                    threshold: float = 1.5) -> tuple[pd.DataFrame, pd.Series]:
    """Restituisce (riepilogo per colonna, maschera booleana delle righe con outlier)."""
    cols = [c for c in (columns or numeric_columns(df)) if is_numeric(df[c])]
    if not cols:
        raise DataError("Nessuna colonna numerica su cui cercare outlier.")
    any_mask = pd.Series(False, index=df.index)
    rows = []
    for c in cols:
        s = df[c].astype(float)
        if s.notna().sum() < 3:
            continue
        lo, hi = outlier_bounds(s, method, threshold)
        mask = (s < lo) | (s > hi)
        any_mask |= mask.fillna(False)
        n = int(mask.sum())
        rows.append({"colonna": c, "outlier": n, "% righe": n / len(s) * 100 if len(s) else 0,
                     "limite inferiore": lo, "limite superiore": hi,
                     "min osservato": s.min(), "max osservato": s.max()})
    return pd.DataFrame(rows), any_mask


def _bounds_code(method: str, threshold: float) -> list[str]:
    if method == "iqr":
        return ["    q1, q3 = df[col].quantile([0.25, 0.75])",
                f"    lo, hi = q1 - {threshold} * (q3 - q1), q3 + {threshold} * (q3 - q1)"]
    return ["    m, sd = df[col].mean(), df[col].std()",
            f"    lo, hi = m - {threshold} * sd, m + {threshold} * sd"]


def remove_outliers(df: pd.DataFrame, columns: list, method: str, threshold: float) -> OpResult:
    summary, mask = detect_outliers(df, columns, method, threshold)
    n = int(mask.sum())
    if not n:
        return OpResult(df, "Nessun outlier trovato con i criteri scelti.", False)
    out = df[~mask].reset_index(drop=True)
    cols = list(summary["colonna"])
    label = "IQR" if method == "iqr" else "Z-score"
    msg = pick(n, "Rimossa {n} riga con outlier", "Rimosse {n} righe con outlier") + \
        f" ({label}, soglia {threshold}) su {names(cols)}"
    code = "\n".join(["_mask = pd.Series(False, index=df.index)", f"for col in {lit(cols)}:",
                      *_bounds_code(method, threshold),
                      "    _mask |= (df[col] < lo) | (df[col] > hi)",
                      "df = df[~_mask].reset_index(drop=True)"])
    return OpResult(out, msg, code=code)


def clip_outliers(df: pd.DataFrame, columns: list, method: str, threshold: float) -> OpResult:
    summary, mask = detect_outliers(df, columns, method, threshold)
    if not int(mask.sum()):
        return OpResult(df, "Nessun outlier trovato con i criteri scelti.", False)
    out = df.copy(deep=False)
    cells = 0
    for _, row in summary.iterrows():
        c = row["colonna"]
        s = df[c]
        lo, hi = row["limite inferiore"], row["limite superiore"]
        if ptypes.is_integer_dtype(s.dtype):  # limiti interni all'intervallo, tipo invariato
            lo, hi = int(np.ceil(lo)), int(np.floor(hi))
        clipped = s.clip(lo, hi)
        cells += int((clipped != s).fillna(False).sum())
        out[c] = clipped
    cols = list(summary["colonna"])
    msg = pick(cells, "Limitato {n} valore anomalo", "Limitati {n} valori anomali") + \
        f" ai confini {'IQR' if method == 'iqr' else 'Z-score'} (winsorizzazione) su {names(cols)}"
    code = "\n".join([f"for col in {lit(cols)}:", *_bounds_code(method, threshold),
                      "    df[col] = df[col].clip(lo, hi)"])
    return OpResult(out, msg, code=code)


def outlier_figure(df: pd.DataFrame, columns: list, method: str, threshold: float, dark: bool):
    cols = [c for c in (columns or numeric_columns(df)) if is_numeric(df[c])][:12]
    with themed(dark):
        fig = new_figure(dark, size=(8, max(3.2, 0.55 * len(cols) + 1.5)))
        ax = fig.add_subplot()
        data = df[cols].astype(float)
        z = (data - data.median()) / (data.quantile(.75) - data.quantile(.25)).replace(0, 1)
        sns.boxplot(data=z, orient="h", ax=ax, color=ACCENT, fliersize=3,
                    whis=threshold if method == "iqr" else 1.5,
                    flierprops={"markerfacecolor": "#ef4444", "markeredgecolor": "#ef4444"})
        ax.set_yticks(range(len(cols)), [short(c) for c in cols])
        ax.set_xlabel("valore standardizzato robusto  (x − mediana) / IQR")
        ax.set_title("Distribuzione e outlier per colonna")
    return fig


# --------------------------------------------------------------------------- #
# Test d'ipotesi
# --------------------------------------------------------------------------- #
TESTS = {
    "Normalità (Shapiro-Wilk + D'Agostino)": "normality",
    "t-test due gruppi (Welch) + Mann-Whitney": "ttest",
    "ANOVA a una via + Kruskal-Wallis": "anova",
    "Chi-quadrato di indipendenza": "chi2",
    "Test di correlazione (Pearson + Spearman)": "corr",
}
TEST_NEEDS = {  # (etichetta variabile 1, etichetta variabile 2 o None)
    "normality": ("Variabile numerica", None),
    "ttest": ("Variabile numerica", "Variabile di gruppo (2 livelli)"),
    "anova": ("Variabile numerica", "Variabile di gruppo"),
    "chi2": ("Variabile categorica A", "Variabile categorica B"),
    "corr": ("Variabile numerica X", "Variabile numerica Y"),
}


def _verdict(p: float, alpha: float, h0: str, h1: str) -> str:
    if np.isnan(p):
        return "  Esito: non determinabile."
    if p < alpha:
        return f"  Esito: p = {p:.4g} < α = {alpha} → si rifiuta H0.\n  Conclusione: {h1}."
    return f"  Esito: p = {p:.4g} ≥ α = {alpha} → non si rifiuta H0.\n  Conclusione: {h0}."


def _groups(df: pd.DataFrame, value, group, min_size: int = 2):
    y = _require_numeric(df, value)
    sub = pd.DataFrame({"y": y, "g": df[group]}).dropna()
    groups = {str(k): g["y"].to_numpy() for k, g in sub.groupby("g", observed=True)
              if len(g) >= min_size}
    return groups


def run_test(df: pd.DataFrame, test: str, var1, var2=None, alpha: float = 0.05) -> str:
    if var1 is None:
        raise DataError("Seleziona la variabile da analizzare.")
    if TEST_NEEDS[test][1] and var2 is None:
        raise DataError(f"Seleziona anche: {TEST_NEEDS[test][1]}.")
    if var2 is not None and var1 == var2:
        raise DataError("Scegli due variabili diverse.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {"normality": _test_normality, "ttest": _test_ttest, "anova": _test_anova,
                "chi2": _test_chi2, "corr": _test_corr}[test](df, var1, var2, alpha)


def _test_normality(df, col, _unused, alpha):
    v = _require_numeric(df, col).dropna()
    if len(v) < 8:
        raise DataError("Servono almeno 8 valori non nulli per il test di normalità.")
    sample = v.sample(5000, random_state=0) if len(v) > 5000 else v
    w, p_sw = stats.shapiro(sample)
    k2, p_k = stats.normaltest(v) if len(v) >= 20 else (np.nan, np.nan)
    h0, h1 = "i dati sono compatibili con una distribuzione normale", \
        "i dati NON seguono una distribuzione normale"
    lines = [f"TEST DI NORMALITÀ — '{col}'   (n = {fmt_int(len(v))})", "",
             "H0: la variabile segue una distribuzione normale.", "",
             f"Shapiro-Wilk{' (campione di 5.000 valori)' if len(v) > 5000 else ''}:",
             f"  W = {w:.4f},  p-value = {p_sw:.4g}", _verdict(p_sw, alpha, h0, h1), ""]
    if not np.isnan(k2):
        lines += ["D'Agostino-Pearson K²:", f"  K² = {k2:.3f},  p-value = {p_k:.4g}",
                  _verdict(p_k, alpha, h0, h1), ""]
    lines += [f"Skewness = {v.skew():.3f}   Kurtosis (eccesso) = {v.kurt():.3f}",
              "Nota: con campioni molto grandi anche piccole deviazioni risultano significative."]
    return "\n".join(lines)


def _test_ttest(df, value, group, alpha):
    groups = _groups(df, value, group)
    if len(groups) != 2:
        raise DataError(f"Il t-test richiede esattamente 2 gruppi con almeno 2 osservazioni: "
                        f"'{group}' ne ha {len(groups)}.\n\nCon più gruppi usa l'ANOVA.")
    (na, a), (nb, b) = groups.items()
    t, p = stats.ttest_ind(a, b, equal_var=False)
    u, p_u = stats.mannwhitneyu(a, b, alternative="two-sided")
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    d = (a.mean() - b.mean()) / pooled if pooled else np.nan
    h0 = f"nessuna differenza significativa tra le medie di '{na}' e '{nb}'"
    h1 = f"le medie di '{na}' e '{nb}' sono significativamente diverse"
    return "\n".join([
        f"CONFRONTO TRA DUE GRUPPI — '{value}' per '{group}'", "",
        f"  {na}: n = {fmt_int(len(a))}, media = {a.mean():.4g}, dev.std = {a.std(ddof=1):.4g}",
        f"  {nb}: n = {fmt_int(len(b))}, media = {b.mean():.4g}, dev.std = {b.std(ddof=1):.4g}", "",
        "H0: le medie dei due gruppi sono uguali.", "",
        "t-test di Welch (varianze non assunte uguali):",
        f"  t = {t:.4f},  p-value = {p:.4g}", _verdict(p, alpha, h0, h1),
        f"  Effect size (d di Cohen) = {d:.3f}", "",
        "Mann-Whitney U (alternativa non parametrica):",
        f"  U = {u:.1f},  p-value = {p_u:.4g}",
        _verdict(p_u, alpha, "distribuzioni non significativamente diverse",
                 "le distribuzioni dei due gruppi sono significativamente diverse"),
    ])


def _test_anova(df, value, group, alpha):
    groups = _groups(df, value, group)
    if len(groups) < 2:
        raise DataError(f"Servono almeno 2 gruppi con 2+ osservazioni in '{group}'.")
    if len(groups) > 50:
        raise DataError(f"'{group}' ha {len(groups)} gruppi: scegli una variabile con meno livelli.")
    f, p = stats.f_oneway(*groups.values())
    h, p_k = stats.kruskal(*groups.values())
    all_v = np.concatenate(list(groups.values()))
    ss_between = sum(len(g) * (g.mean() - all_v.mean()) ** 2 for g in groups.values())
    ss_total = ((all_v - all_v.mean()) ** 2).sum()
    eta2 = ss_between / ss_total if ss_total else np.nan
    lines = [f"ANOVA A UNA VIA — '{value}' per '{group}'   ({len(groups)} gruppi)", ""]
    for k, g in list(groups.items())[:15]:
        lines.append(f"  {short(k, 24):<24} n = {fmt_int(len(g)):>7}   media = {g.mean():.4g}")
    if len(groups) > 15:
        lines.append(f"  … e altri {len(groups) - 15} gruppi")
    lines += ["", "H0: tutte le medie dei gruppi sono uguali.", "",
              f"ANOVA:  F = {f:.4f},  p-value = {p:.4g}",
              _verdict(p, alpha, "nessuna differenza significativa tra le medie",
                       "almeno un gruppo ha una media significativamente diversa"),
              f"  Eta² (quota di varianza spiegata) = {eta2:.3f}", "",
              f"Kruskal-Wallis (non parametrico):  H = {h:.4f},  p-value = {p_k:.4g}",
              _verdict(p_k, alpha, "distribuzioni non significativamente diverse",
                       "almeno un gruppo ha una distribuzione diversa")]
    return "\n".join(lines)


def _test_chi2(df, a, b, alpha):
    sub = df[[a, b]].dropna()
    table = pd.crosstab(sub[a], sub[b])
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise DataError("Entrambe le variabili devono avere almeno 2 categorie.")
    if table.size > 2500:
        raise DataError("Troppe categorie per un test chi-quadrato leggibile (max ~50×50).")
    chi2, p, dof, expected = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    low = (expected < 5).mean() * 100
    return "\n".join([
        f"CHI-QUADRATO DI INDIPENDENZA — '{a}' × '{b}'", "",
        f"  Tabella di contingenza: {table.shape[0]} × {table.shape[1]},  n = {fmt_int(n)}", "",
        "H0: le due variabili sono indipendenti.", "",
        f"  χ² = {chi2:.4f},  gradi di libertà = {dof},  p-value = {p:.4g}",
        _verdict(p, alpha, "nessuna associazione significativa tra le variabili",
                 "le due variabili sono associate (non indipendenti)"),
        f"  V di Cramér = {v:.3f}  (0 = nessuna associazione, 1 = associazione perfetta)",
        "" if low <= 20 else
        f"\n  Attenzione: il {fmt_pct(low)} delle celle ha frequenza attesa < 5; "
        "il test potrebbe non essere affidabile.",
    ])


def _test_corr(df, x, y, alpha):
    sub = pd.DataFrame({"x": _require_numeric(df, x), "y": _require_numeric(df, y)}).dropna()
    if len(sub) < 3:
        raise DataError("Servono almeno 3 coppie di valori non nulli.")
    r, p = stats.pearsonr(sub["x"], sub["y"])
    rho, p_s = stats.spearmanr(sub["x"], sub["y"])
    return "\n".join([
        f"TEST DI CORRELAZIONE — '{x}' e '{y}'   (n = {fmt_int(len(sub))})", "",
        "H0: non esiste correlazione tra le due variabili.", "",
        f"Pearson (relazione lineare):  r = {r:.4f},  p-value = {p:.4g}",
        _verdict(p, alpha, "nessuna correlazione lineare significativa",
                 f"correlazione {_corr_strength(r)} significativa"),
        f"  R² = {r * r:.4f}", "",
        f"Spearman (relazione monotona):  ρ = {rho:.4f},  p-value = {p_s:.4g}",
        _verdict(p_s, alpha, "nessuna correlazione monotona significativa",
                 f"correlazione {_corr_strength(rho)} significativa"),
    ])


def test_code(test: str, var1, var2, alpha: float) -> str:
    v1, v2 = lit(var1), lit(var2)
    if test == "normality":
        return f"print(stats.shapiro(df[{v1}].dropna()))\nprint(stats.normaltest(df[{v1}].dropna()))"
    if test == "ttest":
        return (f"_g = [g[{v1}].dropna() for _, g in df.groupby({v2})]\n"
                "print(stats.ttest_ind(*_g, equal_var=False))\n"
                "print(stats.mannwhitneyu(*_g))")
    if test == "anova":
        return (f"_g = [g[{v1}].dropna() for _, g in df.groupby({v2})]\n"
                "print(stats.f_oneway(*_g))\nprint(stats.kruskal(*_g))")
    if test == "chi2":
        return f"print(stats.chi2_contingency(pd.crosstab(df[{v1}], df[{v2}])))"
    return (f"_d = df[[{v1}, {v2}]].dropna()\n"
            f"print(stats.pearsonr(_d[{v1}], _d[{v2}]))\nprint(stats.spearmanr(_d[{v1}], _d[{v2}]))")


# --------------------------------------------------------------------------- #
# Grafici
# --------------------------------------------------------------------------- #
CHARTS = {
    "Istogramma": "hist", "Boxplot": "box", "Scatterplot": "scatter",
    "Barre (frequenze)": "count", "Valori mancanti per colonna": "missing",
}
CHART_NEEDS = {  # (X richiesto, Y richiesto/consentito, colore consentito)
    "hist": ("X: variabile numerica", None, True),
    "box": ("Y: variabile numerica", "X: raggruppa per (opzionale)", False),
    "scatter": ("X: variabile numerica", "Y: variabile numerica", True),
    "count": ("X: variabile categorica", None, True),
    "missing": (None, None, False),
}


def _hue(df, hue, max_levels=10):
    if hue is None:
        return None
    if df[hue].nunique(dropna=True) > max_levels:
        raise DataError(f"'{hue}' ha troppi valori distinti per essere usata come colore "
                        f"(max {max_levels}).")
    return hue


def chart_figure(df: pd.DataFrame, kind: str, x=None, y=None, hue=None, bins: int = 30,
                 dark: bool = False):
    with themed(dark):
        fig = new_figure(dark)
        ax = fig.add_subplot()
        if kind == "hist":
            _require_numeric(df, x)
            sns.histplot(data=df, x=x, hue=_hue(df, hue), bins=bins, kde=True, ax=ax,
                         element="step" if hue else "bars", color=None if hue else ACCENT)
            ax.set_title(f"Distribuzione di {short(x, 40)}")
        elif kind == "box":
            _require_numeric(df, x)
            if y is not None:
                top = df[y].value_counts().index[:15]
                data = df[df[y].isin(top)]
                sns.boxplot(data=data, x=y, y=x, ax=ax, color=ACCENT, fliersize=3)
                ax.tick_params(axis="x", rotation=30)
                ax.set_title(f"{short(x, 30)} per {short(y, 30)}")
            else:
                sns.boxplot(data=df, y=x, ax=ax, color=ACCENT, fliersize=3)
                ax.set_title(f"Boxplot di {short(x, 40)}")
        elif kind == "scatter":
            _require_numeric(df, x)
            _require_numeric(df, y)
            data = df.sample(MAX_SCATTER_POINTS, random_state=0) if len(df) > MAX_SCATTER_POINTS else df
            sns.scatterplot(data=data, x=x, y=y, hue=_hue(df, hue), ax=ax, s=18, alpha=0.7,
                            edgecolor="none", color=None if hue else ACCENT)
            sub = data[[x, y]].dropna().astype(float)
            if len(sub) >= 3 and sub[x].nunique() > 1:
                k, q = np.polyfit(sub[x], sub[y], 1)
                xs = np.linspace(sub[x].min(), sub[x].max(), 50)
                ax.plot(xs, k * xs + q, color="#ef4444", lw=1.5, label="retta di regressione")
                ax.legend()
            note = f" (campione di {fmt_int(MAX_SCATTER_POINTS)} punti)" if len(df) > MAX_SCATTER_POINTS else ""
            ax.set_title(f"{short(y, 30)} vs {short(x, 30)}{note}")
        elif kind == "count":
            top = df[x].astype("string").value_counts().index[:20]
            data = df[df[x].astype("string").isin(top)].copy()
            data[x] = data[x].astype("string")
            sns.countplot(data=data, y=x, order=list(top), hue=_hue(df, hue), ax=ax,
                          color=None if hue else ACCENT)
            ax.set_title(f"Frequenze di {short(x, 40)}" + (" (prime 20)" if df[x].nunique() > 20 else ""))
        elif kind == "missing":
            pct = (df.isna().mean() * 100).sort_values(ascending=False)
            pct = pct[pct > 0][:30]
            if pct.empty:
                ax.text(0.5, 0.5, "Nessun valore mancante", ha="center",
                        va="center", fontsize=14, transform=ax.transAxes)
                ax.set_axis_off()
            else:
                ax.barh([short(c) for c in pct.index[::-1]], pct.values[::-1], color="#f59e0b")
                ax.set_xlabel("% di valori mancanti")
                ax.set_title("Valori mancanti per colonna")
        else:
            raise DataError(f"Tipo di grafico sconosciuto: {kind}")
    return fig


def chart_code(kind: str, x=None, y=None, hue=None, bins: int = 30) -> str:
    h = f", hue={lit(hue)}" if hue is not None else ""
    body = {
        "hist": f"sns.histplot(data=df, x={lit(x)}{h}, bins={bins}, kde=True, ax=ax)",
        "box": (f"sns.boxplot(data=df, x={lit(y)}, y={lit(x)}, ax=ax)" if y is not None
                else f"sns.boxplot(data=df, y={lit(x)}, ax=ax)"),
        "scatter": f"sns.scatterplot(data=df, x={lit(x)}, y={lit(y)}{h}, ax=ax)",
        "count": f"sns.countplot(data=df, y={lit(x)}{h}, ax=ax)",
        "missing": "(df.isna().mean() * 100).sort_values().plot.barh(ax=ax)",
    }[kind]
    return f"fig, ax = plt.subplots(figsize=(8, 5))\n{body}\nplt.tight_layout()\nplt.show()"
