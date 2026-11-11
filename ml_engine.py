"""ML Studio: addestramento no-code di modelli supervisionati con scikit-learn."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from cleaner import DataError, OpResult, fmt_int, is_numeric, names
from codegen import lit
from plotting import ACCENT, new_figure, short, themed

ALGORITHMS = {
    "regression": {"Regressione lineare": "linear", "Random Forest (regressione)": "rf"},
    "classification": {"Regressione logistica": "logistic", "Random Forest (classificazione)": "rf"},
}
TASK_LABELS = {"regression": "Regressione", "classification": "Classificazione"}
MAX_CLASSES = 50
MAX_CATEGORIES = 30  # categorie per variabile nel one-hot encoding (le altre → "infrequent")


@dataclass
class MLResult:
    task: str
    algorithm: str
    algorithm_label: str
    target: str
    features: list
    metrics: dict
    importance: pd.DataFrame
    pipeline: object
    y_test: np.ndarray
    y_pred: np.ndarray
    n_train: int
    n_test: int
    classes: list = field(default_factory=list)
    confusion: np.ndarray | None = None
    report: str = ""
    notes: list = field(default_factory=list)
    params: dict = field(default_factory=dict)


def suggest_task(s: pd.Series) -> str:
    if is_numeric(s) and s.nunique(dropna=True) > 15:
        return "regression"
    return "classification"


def _prepare_X(X: pd.DataFrame) -> tuple[pd.DataFrame, list, list, list]:
    """Converte date e booleani in numeri; restituisce (X, numeriche, categoriche, note)."""
    X = X.copy(deep=False)
    notes, num, cat = [], [], []
    for c in X.columns:
        s = X[c]
        if ptypes.is_datetime64_any_dtype(s.dtype):
            X[c] = (s - pd.Timestamp("1970-01-01")).dt.days.astype(float)
            notes.append(f"'{c}' (data) convertita in giorni dal 1970")
            num.append(c)
        elif ptypes.is_bool_dtype(s.dtype):
            X[c] = s.astype(float)
            num.append(c)
        elif is_numeric(s):
            X[c] = s.astype(float)
            num.append(c)
        else:
            X[c] = s.astype("string").astype(object).where(s.notna(), np.nan)
            cat.append(c)
    return X, num, cat, notes


def _build_pipeline(task, algorithm, num, cat, params, random_state):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    linear = algorithm in ("linear", "logistic")
    num_steps = [("imputa", SimpleImputer(strategy="median"))]
    if linear:
        num_steps.append(("scala", StandardScaler()))
    transformers = []
    if num:
        transformers.append(("num", Pipeline(num_steps), num))
    if cat:
        transformers.append(("cat", Pipeline([
            ("imputa", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                     max_categories=MAX_CATEGORIES, sparse_output=False)),
        ]), cat))
    pre = ColumnTransformer(transformers, verbose_feature_names_out=True)

    n_est = int(params.get("n_estimators", 200))
    depth = params.get("max_depth") or None
    if task == "regression":
        model = LinearRegression() if algorithm == "linear" else RandomForestRegressor(
            n_estimators=n_est, max_depth=depth, random_state=random_state, n_jobs=-1)
    else:
        model = LogisticRegression(max_iter=2000) if algorithm == "logistic" else \
            RandomForestClassifier(n_estimators=n_est, max_depth=depth,
                                   random_state=random_state, n_jobs=-1)
    return Pipeline([("preprocessing", pre), ("modello", model)])


def _clean_feature_name(name: str, cat: list) -> str:
    kind, _, rest = name.partition("__")
    if kind == "cat":
        for c in sorted(map(str, cat), key=len, reverse=True):
            if rest.startswith(c + "_"):
                value = rest[len(c) + 1:]
                return f"{c} = {'(altre)' if value == 'infrequent_sklearn' else value}"
    return rest or name


def train(df: pd.DataFrame, target, features: list, task: str, algorithm: str,
          test_size: float = 0.2, random_state: int = 42, params: dict | None = None) -> MLResult:
    from sklearn import metrics as skm
    from sklearn.model_selection import train_test_split

    params = params or {}
    if target is None:
        raise DataError("Seleziona la variabile target.")
    features = [f for f in features if f != target]
    if not features:
        raise DataError("Seleziona almeno una feature (variabile esplicativa).")

    data = df[[*features, target]]
    data = data[data[target].notna()]
    dropped = len(df) - len(data)
    if len(data) < 20:
        raise DataError(f"Servono almeno 20 righe con target valorizzato (ce ne sono {len(data)}).")

    y = data[target]
    if task == "regression":
        if not is_numeric(y):
            raise DataError(f"Per la regressione il target deve essere numerico: '{target}' è "
                            f"{y.dtype}. Scegli la classificazione o converti il tipo.")
        y = y.astype(float)
    else:
        n_classes = y.nunique()
        if n_classes < 2:
            raise DataError("Il target ha un solo valore: impossibile addestrare un classificatore.")
        if n_classes > MAX_CLASSES:
            raise DataError(f"Il target ha {n_classes} classi distinte (max {MAX_CLASSES}). "
                            "Se è una variabile continua usa la regressione.")
        y = y.astype(str)

    X, num, cat, notes = _prepare_X(data[features])
    id_like = [c for c in cat if X[c].nunique() > 0.9 * len(X) and len(X) > 50]
    if id_like:
        X = X.drop(columns=id_like)
        cat = [c for c in cat if c not in id_like]
        notes.append(f"escluse colonne simili a identificativi (valori quasi tutti diversi): "
                     f"{names(id_like)}")
    if not num and not cat:
        raise DataError("Nessuna feature utilizzabile dopo la preparazione dei dati.")
    if dropped:
        notes.append(f"{fmt_int(dropped)} righe escluse perché il target è nullo")

    stratify = None
    if task == "classification":
        counts = y.value_counts()
        if counts.min() >= 2 and int(len(y) * test_size) >= len(counts):
            stratify = y
        else:
            notes.append("divisione train/test non stratificata (classi troppo piccole)")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size,
                                              random_state=random_state, stratify=stratify)
    pipe = _build_pipeline(task, algorithm, num, cat, params, random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            pipe.fit(X_tr, y_tr)
        except (ValueError, MemoryError) as exc:
            raise DataError(f"Addestramento non riuscito: {exc}") from exc
        pred = pipe.predict(X_te)

    if task == "regression":
        m = {"RMSE": float(np.sqrt(skm.mean_squared_error(y_te, pred))),
             "MAE": float(skm.mean_absolute_error(y_te, pred)),
             "R²": float(skm.r2_score(y_te, pred))}
        nz = y_te.to_numpy() != 0
        if nz.any():
            m["MAPE %"] = float(np.mean(np.abs((y_te.to_numpy()[nz] - pred[nz]) / y_te.to_numpy()[nz])) * 100)
        classes, cm, report = [], None, ""
    else:
        classes = sorted(y.unique().tolist())
        binary = len(classes) == 2
        m = {"Accuracy": float(skm.accuracy_score(y_te, pred)),
             "F1-Score": float(skm.f1_score(y_te, pred, average="binary" if binary else "weighted",
                                            pos_label=classes[-1] if binary else 1,
                                            zero_division=0)),
             "Precision": float(skm.precision_score(y_te, pred, average="weighted", zero_division=0)),
             "Recall": float(skm.recall_score(y_te, pred, average="weighted", zero_division=0))}
        if binary and hasattr(pipe, "predict_proba"):
            try:
                proba = pipe.predict_proba(X_te)[:, list(pipe.classes_).index(classes[-1])]
                m["ROC AUC"] = float(skm.roc_auc_score(y_te == classes[-1], proba))
            except ValueError:
                pass
        cm = skm.confusion_matrix(y_te, pred, labels=classes)
        report = skm.classification_report(y_te, pred, zero_division=0)

    model = pipe.named_steps["modello"]
    feat_names = [_clean_feature_name(n, cat)
                  for n in pipe.named_steps["preprocessing"].get_feature_names_out()]
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        kind = "importanza (riduzione impurità)"
    else:
        coef = np.atleast_2d(model.coef_)
        imp = np.abs(coef).mean(axis=0)
        kind = "|coefficiente| (feature standardizzate)"
    importance = pd.DataFrame({"feature": feat_names, "importanza": imp, "metrica": kind}) \
        .sort_values("importanza", ascending=False, ignore_index=True)

    label = next(k for k, v in ALGORITHMS[task].items() if v == algorithm)
    return MLResult(task=task, algorithm=algorithm, algorithm_label=label, target=target,
                    features=list(X.columns), metrics=m, importance=importance, pipeline=pipe,
                    y_test=np.asarray(y_te), y_pred=np.asarray(pred), n_train=len(X_tr),
                    n_test=len(X_te), classes=classes, confusion=cm, report=report, notes=notes,
                    params={"test_size": test_size, "random_state": random_state, **params})


def metrics_text(res: MLResult) -> str:
    lines = [f"{TASK_LABELS[res.task].upper()} — {res.algorithm_label}",
             f"Target: {res.target}    Feature usate: {len(res.features)}",
             f"Righe di addestramento: {fmt_int(res.n_train)}    di test: {fmt_int(res.n_test)}", ""]
    for k, v in res.metrics.items():
        lines.append(f"  {k:<10} {v:,.4f}".replace(",", "X").replace(".", ",").replace("X", "."))
    if res.task == "regression":
        lines += ["", "RMSE/MAE: errore medio nelle unità del target (più basso è meglio).",
                  "R²: quota di varianza spiegata (1 = perfetto, 0 = come predire la media)."]
    else:
        lines += ["", "Report per classe:", res.report]
    if res.notes:
        lines += ["", "Note:", *[f"  • {n}" for n in res.notes]]
    return "\n".join(lines)


def importance_figure(res: MLResult, dark: bool, top: int = 20):
    imp = res.importance.head(top).iloc[::-1]
    with themed(dark):
        fig = new_figure(dark, size=(8, max(3.5, 0.32 * len(imp) + 1.2)))
        ax = fig.add_subplot()
        ax.barh([short(f, 32) for f in imp["feature"]], imp["importanza"], color=ACCENT)
        ax.set_xlabel(res.importance["metrica"].iloc[0] if len(res.importance) else "")
        ax.set_title(f"Feature importance — {res.algorithm_label}")
        ax.grid(axis="y", visible=False)
    return fig


def evaluation_figure(res: MLResult, dark: bool):
    import seaborn as sns

    with themed(dark):
        if res.task == "regression":
            fig = new_figure(dark, size=(7, 5))
            ax = fig.add_subplot()
            idx = np.arange(len(res.y_test))
            if len(idx) > 4000:
                idx = np.random.default_rng(0).choice(idx, 4000, replace=False)
            yt, yp = res.y_test[idx], res.y_pred[idx]
            ax.scatter(yt, yp, s=14, alpha=0.6, color=ACCENT, edgecolor="none")
            lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
            ax.plot([lo, hi], [lo, hi], color="#ef4444", lw=1.2, label="predizione perfetta")
            ax.set_xlabel(f"valore reale ({short(res.target)})")
            ax.set_ylabel("valore predetto")
            ax.set_title("Predetto vs reale (set di test)")
            ax.legend()
        else:
            n = len(res.classes)
            fig = new_figure(dark, size=(max(5, 0.5 * n + 3), max(4.2, 0.45 * n + 2.5)))
            ax = fig.add_subplot()
            labels = [short(c, 14) for c in res.classes]
            sns.heatmap(res.confusion, annot=n <= 15, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=labels, yticklabels=labels, cbar=False)
            ax.set_xlabel("classe predetta")
            ax.set_ylabel("classe reale")
            ax.set_title("Matrice di confusione (set di test)")
            ax.grid(False)
    return fig


def add_predictions(df: pd.DataFrame, res: MLResult) -> OpResult:
    missing = [f for f in res.features if f not in df.columns]
    if missing:
        raise DataError(f"Il dataset non contiene più le feature usate dal modello: "
                        f"{names(missing)}. Riaddestra il modello.")
    X, *_ = _prepare_X(df[res.features])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = res.pipeline.predict(X)
    col = f"pred_{res.target}"
    out = df.copy(deep=False)
    out[col] = pred
    msg = f"Aggiunta la colonna '{col}' con le predizioni del modello {res.algorithm_label}"
    # Codice autonomo: riaddestra il modello così lo script funziona anche senza le analisi.
    code = training_code(res) + f"\ndf[{lit(col)}] = pipeline.predict(prepara_X(df[{lit(res.features)}]))"
    return OpResult(out, msg, code=code, imports=ML_IMPORTS)


def save_model(res: MLResult, path: str) -> None:
    import joblib

    try:
        joblib.dump({"pipeline": res.pipeline, "target": res.target, "features": res.features,
                     "task": res.task, "metrics": res.metrics}, path)
    except OSError as exc:
        raise DataError(f"Impossibile salvare il modello:\n{exc}") from exc


def training_code(res: MLResult) -> str:
    p = res.params
    num_steps = "[('imputa', SimpleImputer(strategy='median'))" + (
        ", ('scala', StandardScaler())]" if res.algorithm in ("linear", "logistic") else "]")
    if res.task == "regression":
        model = "LinearRegression()" if res.algorithm == "linear" else (
            f"RandomForestRegressor(n_estimators={p.get('n_estimators', 200)}, "
            f"max_depth={p.get('max_depth') or None}, random_state={p['random_state']}, n_jobs=-1)")
        y_prep, eval_code = (f"y = df_ml[{lit(res.target)}].astype(float)",
                             "print('RMSE', np.sqrt(metrics.mean_squared_error(y_test, pred)))\n"
                             "print('MAE ', metrics.mean_absolute_error(y_test, pred))\n"
                             "print('R2  ', metrics.r2_score(y_test, pred))")
    else:
        model = "LogisticRegression(max_iter=2000)" if res.algorithm == "logistic" else (
            f"RandomForestClassifier(n_estimators={p.get('n_estimators', 200)}, "
            f"max_depth={p.get('max_depth') or None}, random_state={p['random_state']}, n_jobs=-1)")
        y_prep, eval_code = (f"y = df_ml[{lit(res.target)}].astype(str)",
                             "print(metrics.classification_report(y_test, pred))")
    return "\n".join([
        "def prepara_X(X):",
        "    X = X.copy()",
        "    for c in X.columns:",
        "        if pd.api.types.is_datetime64_any_dtype(X[c]):",
        "            X[c] = (X[c] - pd.Timestamp('1970-01-01')).dt.days.astype(float)",
        "        elif pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c]):",
        "            X[c] = X[c].astype(float)",
        "        else:",
        "            X[c] = X[c].astype('string').astype(object).where(X[c].notna(), np.nan)",
        "    return X",
        "",
        f"df_ml = df[df[{lit(res.target)}].notna()]",
        f"X = prepara_X(df_ml[{lit(res.features)}])", y_prep,
        "num = X.select_dtypes('number').columns.tolist()",
        "cat = [c for c in X.columns if c not in num]",
        "pre = ColumnTransformer([",
        f"    ('num', Pipeline({num_steps}), num),",
        "    ('cat', Pipeline([('imputa', SimpleImputer(strategy='most_frequent')),",
        f"                      ('onehot', OneHotEncoder(handle_unknown='infrequent_if_exist', max_categories={MAX_CATEGORIES}, sparse_output=False))]), cat),",
        "])",
        f"pipeline = Pipeline([('preprocessing', pre), ('modello', {model})])",
        f"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={p['test_size']}, "
        f"random_state={p['random_state']})",
        "pipeline.fit(X_train, y_train)",
        "pred = pipeline.predict(X_test)",
        eval_code,
    ])


ML_IMPORTS = (
    "from sklearn import metrics",
    "from sklearn.compose import ColumnTransformer",
    "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor",
    "from sklearn.impute import SimpleImputer",
    "from sklearn.linear_model import LinearRegression, LogisticRegression",
    "from sklearn.model_selection import train_test_split",
    "from sklearn.pipeline import Pipeline",
    "from sklearn.preprocessing import OneHotEncoder, StandardScaler",
)
