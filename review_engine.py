"""Revisore: controlli automatici e indipendenti sui risultati dell'analisi guidata.

Questi controlli non usano l'IA: ricalcolano. Servono a poter difendere ogni numero.

1. Riproducibilità: ogni passo viene rieseguito e il risultato confrontato con il primo.
2. Totali che tornano: la somma dei gruppi deve coincidere con il totale del dataset.
3. Join: se le chiavi non sono univoche da nessuna delle due parti le righe si moltiplicano.
4. Dati visti dal modello: avvisa se una tabella è stata troncata o è vuota.
5. Numeri del testo: ogni cifra scritta nella risposta deve comparire nei risultati calcolati
   (direttamente, oppure come somma, differenza o rapporto di due risultati).

Il modulo non dipende dall'interfaccia grafica né dalla rete.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

import ai_tools
from ai_tools import ToolContext, ToolResult
from cleaner import fmt_num

OK, INFO, WARN, ERROR = "ok", "info", "avviso", "errore"
MAX_EVIDENCE_CELLS = 20_000
MAX_PAIR_EVIDENCE = 700


@dataclass
class Finding:
    level: str
    title: str
    detail: str = ""

    def line(self) -> str:
        mark = {OK: "✓", INFO: "•", WARN: "⚠", ERROR: "✗"}[self.level]
        return f"{mark} {self.title}" + (f" — {self.detail}" if self.detail else "")


# --------------------------------------------------------------------------- #
# Numeri nel testo
# --------------------------------------------------------------------------- #
@dataclass
class Claim:
    raw: str
    values: tuple          # interpretazioni possibili (formato italiano o inglese)
    tolerance: float       # mezzo passo dell'ultima cifra scritta
    percent: bool = False


_DATE = re.compile(r"\b\d{4}-\d{1,2}(?:-\d{1,2})?\b|\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b"
                   r"|\b\d{1,2}:\d{2}(?::\d{2})?\b")
_CODE = re.compile(r"```.*?```|`[^`\n]*`", re.S)
_NUM = re.compile(r"(?<![\w.,])[-−–]?\d(?:[\d.,']*\d)?(?:[eE][-+]?\d+)?(?![\w])(\s?%|\s?(?:mila|migliaia|"
                  r"milioni|mln|miliardi|mld)\b|\s?[kKM]\b)?")
_SCALE = {"mila": 1e3, "migliaia": 1e3, "k": 1e3, "K": 1e3, "milioni": 1e6, "mln": 1e6, "M": 1e6,
          "miliardi": 1e9, "mld": 1e9}
_GROUPED_DOT = re.compile(r"^\d{1,3}(\.\d{3})+$")
_GROUPED_COMMA = re.compile(r"^\d{1,3}(,\d{3})+$")


def _interpret(token: str) -> list[tuple[float, int]]:
    """Possibili letture di un numero scritto: [(valore, cifre decimali)]."""
    token = token.replace("'", "")
    if re.search(r"[eE]", token):
        try:
            mantissa = token.lower().split("e")[0]
            digits = len(mantissa.replace(",", ".").partition(".")[2])
            value = float(token.replace(",", "."))
            exp = int(token.lower().split("e")[1])
            return [(value, max(digits - exp, 0))]
        except ValueError:
            return []
    has_dot, has_comma = "." in token, "," in token
    out = []
    try:
        if has_dot and has_comma:
            if token.rfind(",") > token.rfind("."):          # 1.234,56
                dec = token.rpartition(",")[2]
                out.append((float(token.replace(".", "").replace(",", ".")), len(dec)))
            else:                                             # 1,234.56
                dec = token.rpartition(".")[2]
                out.append((float(token.replace(",", "")), len(dec)))
        elif has_comma:
            if _GROUPED_COMMA.match(token):
                out.append((float(token.replace(",", "")), 0))
            if token.count(",") == 1:
                out.append((float(token.replace(",", ".")), len(token.partition(",")[2])))
        elif has_dot:
            if _GROUPED_DOT.match(token):
                out.append((float(token.replace(".", "")), 0))
            if token.count(".") == 1:
                out.append((float(token), len(token.partition(".")[2])))
        else:
            out.append((float(token), 0))
    except ValueError:
        return []
    return out


def extract_numbers(text: str) -> list[Claim]:
    """Estrae i numeri da un testo, ignorando date, orari e codice."""
    text = _CODE.sub(" ", text or "")
    text = _DATE.sub(" ", text)
    claims = []
    for m in _NUM.finditer(text):
        token = m.group(0)
        suffix = (m.group(1) or "").strip()
        body = token[: len(token) - len(m.group(1) or "")].strip()
        negative = body[:1] in "-−–"
        body = body.lstrip("-−–")
        scale = _SCALE.get(suffix, 1.0)
        readings = _interpret(body)
        if not readings:
            continue
        values, tol = [], 0.0
        for value, decimals in readings:
            v = value * scale * (-1 if negative else 1)
            values += [v, -v] if negative else [v]      # un trattino può essere solo un separatore
            tol = max(tol, 0.5 * 10 ** (-decimals) * scale)
        claims.append(Claim(token.strip(), tuple(dict.fromkeys(values)), tol, suffix == "%"))
    return claims


def _numbers_in(obj, out: list, budget: list) -> None:
    if budget[0] <= 0:
        return
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float, np.integer, np.floating)):
        f = float(obj)
        if not (math.isnan(f) or math.isinf(f)):
            out.append(f)
            budget[0] -= 1
    elif isinstance(obj, str):
        for c in extract_numbers(obj):
            out.extend(c.values)
            budget[0] -= 1
    elif isinstance(obj, dict):
        for v in obj.values():
            _numbers_in(v, out, budget)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _numbers_in(v, out, budget)


def evidence_numbers(results: list[ToolResult], extra=None, full: bool = True) -> np.ndarray:
    """Tutti i numeri calcolati dall'app: risultati degli strumenti e dati aggiuntivi.

    Con `full=False` si considerano solo i numeri effettivamente mostrati al modello (senza le
    righe delle tabelle rimaste in locale): è l'insieme, più piccolo, usato per i numeri derivati.
    """
    out: list = []
    budget = [MAX_EVIDENCE_CELLS]
    for r in results:
        if r is None or r.error:
            continue
        _numbers_in(r.payload, out, budget)
        _numbers_in(r.meta.get("text", ""), out, budget)
        _numbers_in(r.meta.get("metrics", {}), out, budget)
        if full and r.table is not None and budget[0] > 0:
            num = r.table.select_dtypes("number")
            values = num.to_numpy(dtype=float, na_value=np.nan).ravel()[: budget[0]]
            values = values[np.isfinite(values)]
            out.extend(values.tolist())
            budget[0] -= len(values)
    _numbers_in(extra, out, budget)
    arr = np.unique(np.asarray(out, dtype=float)) if out else np.empty(0)
    return arr


def _close(values: np.ndarray, target: float, tol: float) -> bool:
    if values.size == 0:
        return False
    return bool(np.any(np.abs(values - target) <= tol + 1e-9 + abs(target) * 1e-9))


def _derived(evidence: np.ndarray, target: float, tol: float, percent: bool) -> bool:
    """Il numero è la somma, la differenza o il rapporto di due risultati calcolati?"""
    e = evidence
    if e.size < 2 or e.size > MAX_PAIR_EVIDENCE:
        return False
    a, b = e[:, None], e[None, :]
    eps = tol + 1e-9 + abs(target) * 1e-9
    with np.errstate(divide="ignore", invalid="ignore"):
        if percent:
            ratio = np.where(b != 0, a / b * 100, np.nan)
            change = np.where(b != 0, (a - b) / np.abs(b) * 100, np.nan)
            return bool(np.nanmin(np.abs(ratio - target)) <= eps
                        or np.nanmin(np.abs(change - target)) <= eps)
        if np.min(np.abs((a - b) - target)) <= eps or np.min(np.abs((a + b) - target)) <= eps:
            return True
        ratio = np.where(b != 0, a / b, np.nan)
        return bool(np.nanmin(np.abs(ratio - target)) <= eps)


def _trivial(claim: Claim) -> bool:
    """Numeri che non sono risultati: piccoli interi (elenchi, conteggi a vista) e anni."""
    if claim.percent or len(claim.values) == 0:
        return False
    v = abs(claim.values[0])
    if claim.tolerance >= 0.5 and float(v).is_integer():
        return v <= 12 or 1900 <= v <= 2100
    return False


def ground_numbers(text: str, evidence: np.ndarray, known_text: str = "",
                   derive_from: np.ndarray | None = None) -> dict:
    """Confronta i numeri del testo con quelli calcolati.

    Restituisce {"verificati": [...], "derivati": [...], "non_trovati": [...], "totale": n}.
    `known_text` è testo fornito dall'utente (domanda, dizionario): i suoi numeri non sono risultati.
    `derive_from` è l'insieme (ridotto) su cui cercare somme, differenze e rapporti.
    """
    pairs = evidence if derive_from is None else derive_from
    given = set()
    for c in extract_numbers(known_text):
        given.update(c.values)
    verified, derived, missing = [], [], []
    seen = set()
    for c in extract_numbers(text):
        if c.raw in seen or _trivial(c) or any(v in given for v in c.values):
            continue
        seen.add(c.raw)
        direct = any(_close(evidence, v, c.tolerance) for v in c.values)
        if not direct and c.percent:      # 0,242 calcolato → «24,2%» scritto
            direct = any(_close(evidence * 100, v, c.tolerance) for v in c.values)
        if direct:
            verified.append(c.raw)
        elif any(_derived(pairs, v, c.tolerance, c.percent) for v in c.values):
            derived.append(c.raw)
        else:
            missing.append(c.raw)
    return {"verificati": verified, "derivati": derived, "non_trovati": missing,
            "totale": len(verified) + len(derived) + len(missing)}


# --------------------------------------------------------------------------- #
# Controlli sui passi
# --------------------------------------------------------------------------- #
def tables_equal(a: pd.DataFrame | None, b: pd.DataFrame | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    try:
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True),
                                      check_dtype=False, rtol=1e-9, atol=1e-12)
        return True
    except AssertionError:
        return False


def check_totals(result: ToolResult, df: pd.DataFrame) -> list[Finding]:
    """Per lo strumento «aggrega»: la somma dei gruppi deve dare il totale del dataset."""
    out = []
    table = result.table
    if table is None:
        return out
    if "n_righe" in table.columns:
        got, expected = int(table["n_righe"].sum()), len(df)
        out.append(Finding(OK if got == expected else ERROR,
                           "Righe dei gruppi = righe del dataset" if got == expected
                           else "Le righe dei gruppi non coincidono con il dataset",
                           f"{got} su {expected}"))
    for col, func in result.meta.get("metrics", []):
        name = f"{col}_{func}"
        if func != "sum" or name not in table.columns:
            continue
        got, expected = float(table[name].sum()), float(df[col].sum())
        same = math.isclose(got, expected, rel_tol=1e-9, abs_tol=1e-9)
        out.append(Finding(OK if same else ERROR,
                           f"Totale di {col}: somma dei gruppi = totale del dataset" if same
                           else f"Totale di {col}: la somma dei gruppi non coincide",
                           f"{fmt_num(got)} contro {fmt_num(expected)}"))
    for key in result.meta.get("keys", []):
        if key in table.columns:
            n_null = int(table.loc[table[key].isna(), "n_righe"].sum()) if "n_righe" in table else 0
            if n_null:
                out.append(Finding(WARN, f"{n_null} righe senza valore di «{key}»",
                                   f"{fmt_num(n_null / max(len(df), 1) * 100, 1)}% del totale finisce in un "
                                   "gruppo «vuoto»: i confronti tra gruppi non le includono."))
    return out


_IDENT = r'(?:"[^"]+"|\w+)'
_ON_PAIR = re.compile(rf'({_IDENT})\.({_IDENT})\s*=\s*({_IDENT})\.({_IDENT})')
_TABLE_REF = re.compile(r'\b(?:from|join)\s+(df2|df)\b(?:\s+(?:as\s+)?(?!on\b|join\b|where\b|group\b|'
                        r'order\b|left\b|right\b|inner\b|full\b|cross\b|using\b|limit\b)(\w+))?', re.I)


def _unquote(name: str) -> str:
    return name[1:-1] if name.startswith('"') else name


def check_join(query: str, ctx: ToolContext) -> list[Finding]:
    """Avvisa se un JOIN può moltiplicare le righe (chiavi non univoche da entrambe le parti)."""
    if not re.search(r"\bjoin\b", query, re.I):
        return []
    tables = {"df": ctx.df, "df2": ctx.df2}
    alias = {}
    for m in _TABLE_REF.finditer(query):
        table = m.group(1).lower()
        alias[table] = table
        if m.group(2):
            alias[m.group(2).lower()] = table
    keys: dict[tuple, tuple[list, list]] = {}
    for m in _ON_PAIR.finditer(query):
        ta, ca, tb, cb = (_unquote(g) for g in m.groups())
        a, b = alias.get(ta.lower()), alias.get(tb.lower())
        if a is None or b is None or tables.get(a) is None or tables.get(b) is None:
            continue
        left, right = keys.setdefault((ta.lower(), tb.lower()), ([], []))
        left.append((a, ca))
        right.append((b, cb))
    if not keys:
        return [Finding(WARN, "JOIN con chiavi non verificabili automaticamente",
                        "Controlla che la chiave sia univoca in almeno una delle due tabelle.")]
    out = []
    for left, right in keys.values():
        unique = []
        for side in (left, right):
            table = tables[side[0][0]]
            cols = []
            for _t, name in side:
                try:
                    cols.append(ai_tools.resolve(table, name))
                except Exception:  # noqa: BLE001 - colonna non riconosciuta: non verificabile
                    cols = None
                    break
            unique.append(None if cols is None else not bool(table.duplicated(subset=cols).any()))
        label = ", ".join(name for _t, name in left)
        if None in unique:
            out.append(Finding(WARN, f"JOIN su {label}: chiave non verificabile"))
        elif not any(unique):
            out.append(Finding(ERROR, f"JOIN su {label}: chiave ripetuta in entrambe le tabelle",
                               "Relazione molti-a-molti: le righe si moltiplicano e somme e "
                               "conteggi risultano gonfiati."))
        else:
            out.append(Finding(OK, f"JOIN su {label}: chiave univoca in almeno una tabella",
                               "Le righe non vengono duplicate."))
    return out


def review_steps(calls: list[tuple], ctx: ToolContext) -> list[Finding]:
    """calls: [(etichetta, nome strumento, argomenti, ToolResult)] dei passi eseguiti."""
    findings = []
    ok_steps = [(lbl, n, a, r) for lbl, n, a, r in calls if not r.error]
    failed = len(calls) - len(ok_steps)
    if not calls:
        return [Finding(ERROR, "Nessun calcolo eseguito",
                        "La risposta non si basa su risultati calcolati dall'app.")]
    if failed:
        findings.append(Finding(INFO, f"{failed} chiamate non riuscite e corrette dal modello"
                                if ok_steps else f"{failed} chiamate non riuscite",
                                "I loro risultati non sono usati."))
    reproduced = 0
    for label, name, args, result in ok_steps:
        again = ai_tools.run_tool(name, args, ctx)
        if again.error or not tables_equal(result.table, again.table):
            findings.append(Finding(ERROR, f"Passo non riproducibile: {label}",
                                    "Rieseguito, ha dato un risultato diverso."))
        else:
            reproduced += 1
        if name == "aggrega":
            findings += check_totals(result, ctx.df)
        if name == "esegui_sql":
            findings += check_join(result.meta.get("query", ""), ctx)
        if result.table is not None and name in ("esegui_sql", "aggrega"):
            if len(result.table) == 0:
                findings.append(Finding(WARN, f"Risultato vuoto: {label}",
                                        "Nessuna riga soddisfa i criteri."))
            elif name == "esegui_sql" and len(result.table) > ai_tools.MAX_ROWS_TO_MODEL:
                findings.append(Finding(WARN, f"Risultato troncato: {label}",
                                        f"Il modello ha visto {ai_tools.MAX_ROWS_TO_MODEL} righe su "
                                        f"{len(result.table)}: le conclusioni valgono solo per quelle."))
    if ok_steps:
        findings.insert(0, Finding(OK if reproduced == len(ok_steps) else ERROR,
                                   f"Riproducibilità: {reproduced} passi su {len(ok_steps)} "
                                   "rieseguiti con lo stesso risultato"))
    else:
        findings.insert(0, Finding(ERROR, "Nessun calcolo riuscito",
                                   "La risposta non si basa su risultati calcolati dall'app."))
    return findings


def grounding_finding(grounding: dict, what: str = "nel report") -> Finding:
    total, missing = grounding["totale"], grounding["non_trovati"]
    if total == 0:
        return Finding(INFO, f"Nessun numero da verificare {what}")
    if missing:
        return Finding(ERROR, f"{len(missing)} numeri {what} non trovati nei risultati calcolati",
                       ", ".join(missing[:15]) + ("…" if len(missing) > 15 else ""))
    detail = f"{len(grounding['verificati'])} riscontrati direttamente"
    if grounding["derivati"]:
        detail += (f", {len(grounding['derivati'])} ricavabili dai risultati (somma, differenza o "
                   "rapporto): " + ", ".join(grounding["derivati"][:10]))
    return Finding(OK, f"Tutti i {total} numeri {what} hanno riscontro nei risultati", detail)


def worst(findings: list[Finding]) -> str:
    levels = {f.level for f in findings}
    return ERROR if ERROR in levels else WARN if WARN in levels else OK
