"""Caricamento, analisi, pulizia, conversione dei tipi ed esportazione dei dati.

Il modulo non dipende dall'interfaccia grafica: ogni operazione riceve un
DataFrame e restituisce un nuovo DataFrame (l'originale non viene mai
modificato) insieme al codice pandas equivalente, usato dal Code Generator.
"""
from __future__ import annotations

import csv
import os
import re
import unicodedata
from dataclasses import dataclass

import pandas as pd
from pandas.api import types as ptypes

from codegen import lit

# Copy-on-Write: le copie "superficiali" restano sicure ed economiche in memoria.
pd.set_option("mode.copy_on_write", True)
pd.set_option("future.no_silent_downcasting", True)

CSV_EXTS = {".csv", ".tsv", ".txt"}
EXCEL_EXTS = {".xlsx", ".xlsm"}
PARQUET_EXTS = {".parquet"}
READ_EXTS = CSV_EXTS | EXCEL_EXTS | PARQUET_EXTS | {".xls"}
EXPORT_EXTS = {".csv", ".xlsx", ".parquet"}

EXCEL_MAX_ROWS = 1_048_576
EXCEL_MAX_COLS = 16_384

MISSING_STRATEGIES = {
    "drop": "Elimina righe con valori mancanti",
    "mean": "Riempi con la media (colonne numeriche)",
    "median": "Riempi con la mediana (colonne numeriche)",
    "mode": "Riempi con la moda (valore più frequente)",
    "value": "Riempi con un valore personalizzato",
}

CAST_TARGETS = {
    "Decimale (float)": "float",
    "Intero (int)": "int",
    "Data/ora (datetime)": "datetime",
    "Testo (string)": "str",
    "Booleano (bool)": "bool",
    "Categoria (category)": "category",
}
TRUE_TOKENS = ("true", "vero", "1", "si", "sì", "yes", "y", "s", "t", "x")
FALSE_TOKENS = ("false", "falso", "0", "no", "n", "f")


class DataError(Exception):
    """Errore con un messaggio già pronto da mostrare all'utente."""


@dataclass
class OpResult:
    df: pd.DataFrame
    message: str
    changed: bool = True
    code: str = ""
    imports: tuple = ()


# --------------------------------------------------------------------------- #
# Utilità di formattazione
# --------------------------------------------------------------------------- #
def fmt_int(n: int) -> str:
    return f"{int(n):,}".replace(",", ".")


def fmt_pct(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}%".replace(".", ",")


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}".replace(".", ",")
        n /= 1024
    return f"{n:.1f} GB"


def pick(count: int, singular: str, plural: str) -> str:
    """Sceglie la forma singolare/plurale; '{n}' viene sostituito dal numero."""
    return (singular if count == 1 else plural).replace("{n}", fmt_int(count))


def names(cols, limit: int = 5) -> str:
    cols = [str(c) for c in cols]
    shown = ", ".join(cols[:limit])
    return shown + (f" e altre {len(cols) - limit}" if len(cols) > limit else "")


def is_numeric(s: pd.Series) -> bool:
    return ptypes.is_numeric_dtype(s.dtype) and not ptypes.is_bool_dtype(s.dtype)


# --------------------------------------------------------------------------- #
# Caricamento
# --------------------------------------------------------------------------- #
def excel_sheets(path: str) -> list[str]:
    """Restituisce i fogli di un file Excel (lista vuota se non è Excel)."""
    if os.path.splitext(path)[1].lower() not in EXCEL_EXTS:
        return []
    try:
        with pd.ExcelFile(path, engine="openpyxl") as xf:
            return [str(s) for s in xf.sheet_names]
    except Exception as exc:  # noqa: BLE001 - tradotto in messaggio utente
        raise _excel_read_error(exc) from exc


def load_file(path: str, sheet: str | None = None, var: str = "df") -> tuple[pd.DataFrame, str]:
    """Carica un file e restituisce (DataFrame, codice pandas equivalente)."""
    if not os.path.isfile(path):
        raise DataError(f"Il file non esiste:\n{path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xls":
        raise DataError("Il formato .xls (Excel 97-2003) non è supportato.\n\n"
                        "Apri il file in Excel e salvalo come .xlsx, poi riprova.")
    if ext not in READ_EXTS:
        raise DataError(f"Formato '{ext or 'senza estensione'}' non supportato.\n\n"
                        "Formati accettati: CSV, TSV, TXT, XLSX, XLSM, Parquet.")
    if os.path.getsize(path) == 0:
        raise DataError("Il file è vuoto.")

    try:
        if ext in CSV_EXTS:
            df, kwargs = _read_csv(path)
            extra = "".join(f", {k}={lit(v)}" for k, v in kwargs.items())
            code = f"{var} = pd.read_csv({path!r}{extra})"
        elif ext in EXCEL_EXTS:
            try:
                df = pd.read_excel(path, sheet_name=sheet or 0, engine="openpyxl")
            except Exception as exc:  # noqa: BLE001
                raise _excel_read_error(exc) from exc
            code = f"{var} = pd.read_excel({path!r}, sheet_name={lit(sheet or 0)})"
        else:
            df = _read_parquet(path)
            code = f"{var} = pd.read_parquet({path!r})"
    except PermissionError as exc:
        raise DataError("Impossibile leggere il file: potrebbe essere aperto in un altro "
                        "programma (es. Excel) oppure non hai i permessi necessari.") from exc
    except MemoryError as exc:
        raise DataError("Memoria insufficiente per caricare questo file.") from exc

    if df.shape[1] == 0:
        raise DataError("Il file non contiene colonne leggibili.")
    return df, code


def _excel_read_error(exc: Exception) -> DataError:
    if isinstance(exc, DataError):
        return exc
    if isinstance(exc, PermissionError):
        return DataError("Impossibile leggere il file: potrebbe essere aperto in Excel. "
                         "Chiudilo e riprova.")
    if isinstance(exc, ValueError) and "Worksheet" in str(exc):
        return DataError(f"Foglio di lavoro non trovato: {exc}")
    return DataError("Il file non è un file Excel valido oppure è danneggiato.\n\n"
                     f"Dettaglio: {type(exc).__name__}: {exc}")


def _read_parquet(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        raise DataError("Per leggere file Parquet serve il pacchetto 'pyarrow'.") from exc
    except Exception as exc:  # noqa: BLE001
        raise DataError(f"File Parquet non valido o danneggiato.\n\nDettaglio: {exc}") from exc


def _read_csv(path: str) -> tuple[pd.DataFrame, dict]:
    with open(path, "rb") as fh:
        sample = fh.read(64 * 1024)
    if not sample.strip():
        raise DataError("Il file è vuoto.")

    if sample.startswith((b"\xff\xfe", b"\xfe\xff")):
        encodings = ["utf-16"]
    elif b"\x00" in sample:
        raise DataError("Il file non sembra un file di testo CSV (contiene dati binari).\n\n"
                        "Se è un file Excel, verifica che abbia estensione .xlsx.")
    else:
        encodings = ["utf-8-sig", "cp1252", "latin-1"]

    text = sample.decode("utf-16" if encodings == ["utf-16"] else "latin-1", errors="ignore")
    sep = _sniff_delimiter(text, default="\t" if path.lower().endswith(".tsv") else ",")

    # Con il separatore ";" (export di Excel in italiano) i decimali usano spesso la virgola:
    # si sceglie in base ai numeri presenti nel campione.
    decimal_comma = sep != "," and _comma_decimals(text, sep)

    last_error: Exception | None = None
    for enc in encodings:
        kwargs = {"sep": sep, "encoding": enc}
        if decimal_comma:
            kwargs["decimal"] = ","
        try:
            return pd.read_csv(path, low_memory=False, **kwargs), kwargs
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except pd.errors.EmptyDataError as exc:
            raise DataError("Il file è vuoto o non contiene colonne.") from exc
        except pd.errors.ParserError as exc:
            raise DataError("Il CSV ha una struttura non valida (numero di campi non coerente "
                            f"tra le righe).\n\nDettaglio: {exc}") from exc
    raise DataError(f"Impossibile decodificare il testo del file.\n\nDettaglio: {last_error}")


def _comma_decimals(text: str, sep: str) -> bool:
    body = "\n".join(text.splitlines()[1:200])
    cells = [cell.strip().strip('"') for line in body.splitlines() for cell in line.split(sep)]
    comma = sum(bool(re.fullmatch(r"-?\d+,\d+", x)) for x in cells)
    dot = sum(bool(re.fullmatch(r"-?\d+\.\d+", x)) for x in cells)
    return comma > dot


def _sniff_delimiter(text: str, default: str = ",") -> str:
    lines = [ln for ln in text.splitlines()[:50] if ln.strip()]
    if not lines:
        return default
    try:
        return csv.Sniffer().sniff("\n".join(lines), delimiters=",;\t|").delimiter
    except csv.Error:
        counts = {d: lines[0].count(d) for d in ",;\t|"}
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else default


# --------------------------------------------------------------------------- #
# Analisi
# --------------------------------------------------------------------------- #
def dtype_label(s: pd.Series) -> str:
    dt = s.dtype
    if ptypes.is_bool_dtype(dt):
        return "Booleano"
    if ptypes.is_integer_dtype(dt):
        return "Intero"
    if ptypes.is_float_dtype(dt):
        return "Decimale"
    if ptypes.is_datetime64_any_dtype(dt):
        return "Data/ora"
    if ptypes.is_timedelta64_dtype(dt):
        return "Durata"
    if isinstance(dt, pd.CategoricalDtype):
        return "Categoria"
    if ptypes.is_object_dtype(dt):
        kind = ptypes.infer_dtype(s, skipna=True)
        if kind in ("string", "empty"):
            return "Testo"
        if kind.startswith("mixed"):
            return "Misto"
        return {"boolean": "Booleano", "integer": "Intero", "floating": "Decimale",
                "datetime": "Data/ora", "date": "Data"}.get(kind, "Testo")
    if ptypes.is_string_dtype(dt):
        return "Testo"
    return str(dt)


def summarize(df: pd.DataFrame) -> dict:
    rows = len(df)
    nulls = df.isna().sum()
    columns = []
    for col in df.columns:
        s = df[col]
        try:
            unique = int(s.nunique(dropna=True))
        except TypeError:
            unique = None
        n_null = int(nulls[col])
        columns.append({
            "name": col, "label": dtype_label(s), "dtype": str(s.dtype),
            "non_null": rows - n_null, "nulls": n_null,
            "pct": (n_null / rows * 100) if rows else 0.0, "unique": unique,
        })
    try:
        duplicates = int(df.duplicated().sum())
    except TypeError:
        duplicates = None
    total_cells = rows * df.shape[1]
    return {
        "rows": rows, "cols": df.shape[1],
        "memory": int(df.memory_usage(deep=True).sum()),
        "total_nulls": int(nulls.sum()),
        "null_pct": (nulls.sum() / total_cells * 100) if total_cells else 0.0,
        "duplicates": duplicates, "columns": columns,
    }


# --------------------------------------------------------------------------- #
# Pulizia
# --------------------------------------------------------------------------- #
def to_snake_case(name) -> str:
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)   # "HTTPCode" -> "HTTP_Code"
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)      # "camelCase" -> "camel_Case"
    s = re.sub(r"[^0-9A-Za-z]+", "_", s)
    return s.strip("_").lower()


def clean_column_names(df: pd.DataFrame) -> OpResult:
    new_names: list[str] = []
    seen: set[str] = set()
    for i, col in enumerate(df.columns):
        base = to_snake_case(col) or f"colonna_{i + 1}"
        if base[0].isdigit():
            base = f"col_{base}"
        name, k = base, 2
        while name in seen:
            name, k = f"{base}_{k}", k + 1
        seen.add(name)
        new_names.append(name)

    mapping = {old: new for old, new in zip(df.columns, new_names) if str(old) != new}
    if not mapping:
        return OpResult(df, "I nomi delle colonne sono già in formato snake_case.", False)
    msg = pick(len(mapping), "Convertita {n} colonna in snake_case",
               "Convertite {n} colonne in snake_case")
    code = f"df = df.rename(columns={lit(mapping)})"
    return OpResult(df.set_axis(new_names, axis=1), msg, code=code)


def drop_columns(df: pd.DataFrame, columns: list) -> OpResult:
    if not columns:
        raise DataError("Seleziona almeno una colonna nell'elenco.")
    if len(columns) >= df.shape[1]:
        raise DataError("Non puoi eliminare tutte le colonne del dataset.")
    msg = pick(len(columns), "Eliminata {n} colonna: ", "Eliminate {n} colonne: ") + names(columns)
    return OpResult(df.drop(columns=columns), msg, code=f"df = df.drop(columns={lit(list(columns))})")


def handle_missing(df: pd.DataFrame, strategy: str, columns: list | None = None,
                   value: str | None = None) -> OpResult:
    targets = list(columns) if columns else list(df.columns)
    null_counts = df[targets].isna().sum()
    with_nulls = [c for c in targets if null_counts[c] > 0]
    scope = "nelle colonne selezionate" if columns else "nel dataset"
    if not with_nulls:
        return OpResult(df, f"Nessun valore mancante {scope}.", False)

    if strategy == "drop":
        out = df.dropna(subset=with_nulls).reset_index(drop=True)
        removed = len(df) - len(out)
        msg = pick(removed, "Rimossa {n} riga con valori mancanti",
                   "Rimosse {n} righe con valori mancanti")
        subset = f"subset={lit(with_nulls)}" if columns else ""
        return OpResult(out, msg, code=f"df = df.dropna({subset}).reset_index(drop=True)")

    fill: dict = {}
    skipped: list = []
    if strategy in ("mean", "median"):
        for c in with_nulls:
            s = df[c]
            if is_numeric(s):
                stat = s.mean() if strategy == "mean" else s.median()
                if pd.notna(stat):
                    fill[c] = stat
                    continue
            skipped.append(c)
        what = "la media" if strategy == "mean" else "la mediana"
        if not fill:
            raise DataError(f"Non è possibile calcolare {what}: nessuna colonna numerica con "
                            f"valori mancanti {scope}.\n\nPer le colonne di testo usa la moda "
                            "o un valore personalizzato.")
        label = what
        code = f"df = df.fillna({{c: df[c].{strategy}() for c in {lit(list(fill))}}})"
    elif strategy == "mode":
        for c in with_nulls:
            mode = df[c].mode(dropna=True)
            if len(mode):
                fill[c] = mode.iloc[0]
            else:
                skipped.append(c)
        if not fill:
            raise DataError("Le colonne selezionate non contengono valori da cui calcolare la moda.")
        label = "la moda"
        code = f"df = df.fillna({{c: df[c].mode().iloc[0] for c in {lit(list(fill))}}})"
    elif strategy == "value":
        if value is None or value.strip() == "":
            raise DataError("Inserisci il valore con cui riempire i valori mancanti.")
        for c in with_nulls:
            try:
                fill[c] = _coerce_fill_value(value, df[c])
            except (ValueError, TypeError):
                skipped.append(c)
        if not fill:
            raise DataError(f"Il valore '{value}' non è compatibile con il tipo delle colonne "
                            "che contengono valori mancanti (es. testo in una colonna numerica).")
        label = f"il valore '{value}'"
        code = f"df = df.fillna({lit(fill)})"
    else:
        raise DataError(f"Strategia sconosciuta: {strategy}")

    filled = int(null_counts[list(fill)].sum())
    out = df.fillna(fill)
    msg = (pick(filled, "Riempito {n} valore mancante", "Riempiti {n} valori mancanti")
           + f" con {label} in " + pick(len(fill), "{n} colonna", "{n} colonne"))
    if skipped:
        msg += " (" + pick(len(skipped), "ignorata {n} colonna non compatibile",
                           "ignorate {n} colonne non compatibili") + f": {names(skipped)})"
    return OpResult(out, msg, code=code)


def _coerce_fill_value(value: str, s: pd.Series):
    """Converte il valore inserito dall'utente nel tipo della colonna."""
    raw = value.strip()
    dt = s.dtype
    if ptypes.is_bool_dtype(dt):
        if raw.lower() in TRUE_TOKENS:
            return True
        if raw.lower() in FALSE_TOKENS:
            return False
        raise ValueError(raw)
    if ptypes.is_numeric_dtype(dt):
        num = raw.replace(",", ".")
        try:
            return int(num)
        except ValueError:
            return float(num)
    if ptypes.is_datetime64_any_dtype(dt):
        return pd.to_datetime(raw, dayfirst=True)
    return value


def remove_duplicates(df: pd.DataFrame, columns: list | None = None) -> OpResult:
    try:
        dup = int(df.duplicated(subset=columns or None).sum())
    except TypeError as exc:
        raise DataError("Alcune celle contengono valori non confrontabili (es. liste).") from exc
    if not dup:
        return OpResult(df, "Nessuna riga duplicata trovata.", False)
    out = df.drop_duplicates(subset=columns or None).reset_index(drop=True)
    msg = pick(dup, "Rimossa {n} riga duplicata", "Rimosse {n} righe duplicate")
    if columns:
        msg += f" (confronto su: {names(columns)})"
    subset = f"subset={lit(list(columns))}" if columns else ""
    return OpResult(out, msg, code=f"df = df.drop_duplicates({subset}).reset_index(drop=True)")


def trim_text(df: pd.DataFrame, collapse_inner: bool = True,
              empty_to_null: bool = False) -> OpResult:
    out = df.copy(deep=False)
    cells = 0
    touched = []
    for col in df.columns:
        s = df[col]
        if not (ptypes.is_object_dtype(s.dtype) or ptypes.is_string_dtype(s.dtype)):
            continue
        mask = s.map(lambda v: isinstance(v, str)).astype(bool)
        if not mask.any():
            continue
        vals = s[mask]
        new = vals.str.strip()
        if collapse_inner:
            new = new.str.replace(r"\s+", " ", regex=True)
        if empty_to_null:
            new = new.mask(new == "")
        diff = new.ne(vals)
        n = int(diff.sum())
        if n:
            col_out = s.copy()
            col_out[mask] = new
            out[col] = col_out
            cells += n
            touched.append(col)
    if not cells:
        return OpResult(df, "Nessuno spazio superfluo trovato nelle colonne di testo.", False)
    msg = (pick(cells, "Ripulita {n} cella di testo", "Ripulite {n} celle di testo")
           + " in " + pick(len(touched), "{n} colonna", "{n} colonne")
           + " (rimossi spazi superflui)")
    steps = ["    testo = df.loc[mask, col].str.strip()"]
    if collapse_inner:
        steps.append('    testo = testo.str.replace(r"\\s+", " ", regex=True)')
    if empty_to_null:
        steps.append('    testo = testo.mask(testo == "")')
    code = "\n".join([
        f"for col in {lit(touched)}:",
        "    mask = df[col].map(lambda v: isinstance(v, str))",
        *steps,
        "    df.loc[mask, col] = testo",
    ])
    return OpResult(out, msg, code=code)


# --------------------------------------------------------------------------- #
# Conversione dei tipi
# --------------------------------------------------------------------------- #
_NUM_STRIP = r"[\s €$£]"


def _numeric_from(s: pd.Series, decimal_comma: bool) -> pd.Series:
    if ptypes.is_bool_dtype(s.dtype):
        return s.astype("Float64").astype(float)
    if ptypes.is_numeric_dtype(s.dtype):
        return s.astype(float)
    txt = s.astype("string").str.replace(_NUM_STRIP, "", regex=True)
    if decimal_comma:
        txt = txt.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(txt, errors="coerce").astype(float)


def cast_columns(df: pd.DataFrame, columns: list, target: str, fmt: str = "",
                 strict: bool = False, decimal_comma: bool = False,
                 dayfirst: bool = True) -> OpResult:
    """Forza il tipo di dato delle colonne indicate.

    target: float | int | datetime | str | bool | category.
    strict: se True, interrompe l'operazione quando un valore non è convertibile;
            altrimenti i valori non convertibili diventano nulli.
    """
    if not columns:
        raise DataError("Seleziona almeno una colonna da convertire.")
    out = df.copy(deep=False)
    lost_total = 0
    code_lines: list[str] = []
    for col in columns:
        s = df[col]
        if target in ("float", "int"):
            new = _numeric_from(s, decimal_comma)
            if target == "int":
                frac = new.notna() & (new % 1 != 0)
                if frac.any():
                    if strict:
                        raise DataError(f"La colonna '{col}' contiene valori decimali (es. "
                                        f"{names(s[frac].head(5).tolist())}): non è possibile "
                                        "convertirla in intero senza perdere informazioni.")
                    new = new.mask(frac)
                new = new.round().astype("Int64")
        elif target == "datetime":
            if is_numeric(s):
                raise DataError(f"La colonna '{col}' è numerica: la conversione diretta in data "
                                "non è supportata. Convertila prima in testo.")
            if ptypes.is_datetime64_any_dtype(s.dtype) and not fmt:
                new = s
            else:
                try:
                    new = pd.to_datetime(s.astype("string") if not ptypes.is_datetime64_any_dtype(s.dtype) else s,
                                         format=fmt or "mixed", dayfirst=dayfirst, errors="coerce")
                except (ValueError, TypeError) as exc:
                    raise DataError(f"Formato data non valido '{fmt}': {exc}\n\n"
                                    "Esempi: %d/%m/%Y, %Y-%m-%d %H:%M, %d.%m.%y") from exc
        elif target == "str":
            new = s.astype("string")
        elif target == "bool":
            low = s.astype("string").str.strip().str.lower()
            new = pd.Series(pd.NA, index=s.index, dtype="boolean")
            new[low.isin(TRUE_TOKENS).fillna(False)] = True
            new[low.isin(FALSE_TOKENS).fillna(False)] = False
        elif target == "category":
            new = s.astype("category")
        else:
            raise DataError(f"Tipo di destinazione sconosciuto: {target}")

        invalid = new.isna() & s.notna()
        n_bad = int(invalid.sum())
        if n_bad and strict:
            raise DataError(f"{pick(n_bad, '{n} valore', '{n} valori')} della colonna '{col}' "
                            f"non sono convertibili, ad esempio: "
                            f"{names(s[invalid].astype(str).unique()[:5].tolist())}.\n\n"
                            "Disattiva la modalità rigorosa per impostarli a null.")
        lost_total += n_bad
        out[col] = new
        code_lines.append(_cast_code(col, target, fmt, decimal_comma, dayfirst))

    label = next(k for k, v in CAST_TARGETS.items() if v == target)
    msg = pick(len(columns), "Convertita {n} colonna", "Convertite {n} colonne") + \
        f" in {label}: {names(columns)}"
    if lost_total:
        msg += " (" + pick(lost_total, "{n} valore non convertibile impostato a null",
                           "{n} valori non convertibili impostati a null") + ")"
    return OpResult(out, msg, code="\n".join(code_lines))


def _cast_code(col, target: str, fmt: str, decimal_comma: bool, dayfirst: bool) -> str:
    c = lit(col)
    if target in ("float", "int"):
        lines = [f'_s = df[{c}].astype("string").str.replace(r"{_NUM_STRIP}", "", regex=True)']
        if decimal_comma:
            lines.append('_s = _s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)')
        conv = 'pd.to_numeric(_s, errors="coerce")'
        if target == "int":
            lines.append(f"_n = {conv}")
            lines.append(f'df[{c}] = _n.mask(_n % 1 != 0).round().astype("Int64")')
        else:
            lines.append(f"df[{c}] = {conv}.astype(float)")
        return "\n".join(lines)
    if target == "datetime":
        return (f'df[{c}] = pd.to_datetime(df[{c}], format={lit(fmt or "mixed")}, '
                f'dayfirst={dayfirst}, errors="coerce")')
    if target == "str":
        return f'df[{c}] = df[{c}].astype("string")'
    if target == "bool":
        return (f'_low = df[{c}].astype("string").str.strip().str.lower()\n'
                f'df[{c}] = pd.Series(pd.NA, index=df.index, dtype="boolean")'
                f'.mask(_low.isin({lit(TRUE_TOKENS)}).fillna(False), True)'
                f'.mask(_low.isin({lit(FALSE_TOKENS)}).fillna(False), False)')
    return f'df[{c}] = df[{c}].astype("category")'


# --------------------------------------------------------------------------- #
# Esportazione
# --------------------------------------------------------------------------- #
def export(df: pd.DataFrame, path: str) -> str:
    """Salva il dataset; restituisce un'eventuale nota per il log ('' se nessuna)."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in EXPORT_EXTS:
        raise DataError("Formato di esportazione non supportato. Usa .csv, .xlsx o .parquet.")
    note = ""
    try:
        if ext == ".csv":
            # utf-8 con BOM: Excel apre correttamente lettere accentate e simboli.
            df.to_csv(path, index=False, encoding="utf-8-sig")
        elif ext == ".xlsx":
            check_excel_limits(df)
            _export_excel(df, path)
        else:
            note = _export_parquet(df, path)
    except PermissionError as exc:
        raise DataError("Impossibile scrivere il file: potrebbe essere aperto in un altro "
                        "programma (es. Excel) oppure la cartella non è accessibile.") from exc
    except OSError as exc:
        raise DataError(f"Errore durante la scrittura del file:\n{exc}") from exc
    return note


def export_code(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return f'df.to_csv({path!r}, index=False, encoding="utf-8-sig")'
    if ext == ".xlsx":
        return f"df.to_excel({path!r}, index=False)"
    return f"df.to_parquet({path!r}, index=False)"


def check_excel_limits(df: pd.DataFrame) -> None:
    if len(df) + 1 > EXCEL_MAX_ROWS or df.shape[1] > EXCEL_MAX_COLS:
        raise DataError(f"Excel supporta al massimo {fmt_int(EXCEL_MAX_ROWS - 1)} righe e "
                        f"{fmt_int(EXCEL_MAX_COLS)} colonne. Esporta in CSV o Parquet.")


def excel_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Rimuove fusi orari e caratteri di controllo non ammessi da Excel."""
    out = df.copy(deep=False)
    for col in out.columns:
        s = out[col]
        if isinstance(s.dtype, pd.DatetimeTZDtype):
            out[col] = s.dt.tz_localize(None)
        elif ptypes.is_object_dtype(s.dtype) or ptypes.is_string_dtype(s.dtype):
            out[col] = s.map(lambda v: re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", v)
                             if isinstance(v, str) else v)
    return out


def _export_excel(df: pd.DataFrame, path: str) -> None:
    from openpyxl.utils.exceptions import IllegalCharacterError

    try:
        df.to_excel(path, index=False, engine="openpyxl")
    except (IllegalCharacterError, ValueError):
        # Fusi orari o caratteri di controllo: riprova con una copia compatibile.
        excel_safe(df).to_excel(path, index=False, engine="openpyxl")


def _export_parquet(df: pd.DataFrame, path: str) -> str:
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise DataError("Per esportare in Parquet serve il pacchetto 'pyarrow'.") from exc

    out = df.copy(deep=False)
    out.columns = [str(c) for c in out.columns]
    try:
        out.to_parquet(path, index=False)
        return ""
    except (pa.ArrowInvalid, pa.ArrowTypeError, ValueError):
        # Colonne con tipi misti: Parquet richiede un tipo unico per colonna.
        mixed = [c for c in out.columns if ptypes.is_object_dtype(out[c].dtype)]
        for c in mixed:
            out[c] = out[c].map(lambda v: v if pd.isna(v) else str(v))
        out.to_parquet(path, index=False)
        return " (" + pick(len(mixed), "{n} colonna mista convertita in testo",
                           "{n} colonne miste convertite in testo") + ")"
