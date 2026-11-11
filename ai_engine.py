"""Assistente IA: unico modulo che comunica con Google Gemini (SDK ufficiale google-genai).

La chiave API si legge solo dalla variabile d'ambiente indicata in ai_config.API_KEY_ENV e non
viene mai stampata, registrata o salvata. Il modulo non dipende dall'interfaccia grafica.
"""
from __future__ import annotations

import mimetypes
import os
import sys
import threading
from dataclasses import dataclass, field

import ai_config
from cleaner import DataError

MODEL = ai_config.GEMINI_MODEL

SYSTEM_PROMPT = (
    "Sei l'assistente di data analysis integrato in Data Core Analytics, un'applicazione desktop "
    "per pulizia dati, statistica, time series e machine learning in Python (pandas, scikit-learn, "
    "statsmodels, DuckDB). Rispondi in italiano, in modo chiaro e pratico, usando Markdown. "
    "Quando proponi codice usa Python/pandas e chiama `df` il DataFrame corrente. Se ti viene "
    "fornito il contesto del dataset, basa le risposte su quei dati e non inventare colonne."
)

IMAGE_EXTS = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
              ".gif": "image/gif", ".webp": "image/webp"}
TEXT_EXTS = {".txt", ".csv", ".tsv", ".md", ".json", ".py", ".sql", ".log", ".xml", ".html",
             ".htm", ".yaml", ".yml", ".ini", ".cfg", ".r", ".js", ".ts"}
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_PDF_BYTES = 15 * 1024 * 1024
MAX_TEXT_CHARS = 400_000
MAX_TOTAL_BYTES = 18 * 1024 * 1024   # le richieste con dati inline sono limitate a ~20 MB
RETRY_STATUS_CODES = [408, 500, 502, 503, 504]  # 429 escluso: ritentare consumerebbe quota


# --------------------------------------------------------------------------- #
# Chiave API (solo da variabile d'ambiente)
# --------------------------------------------------------------------------- #
def api_key() -> str:
    """Legge la chiave dall'ambiente del processo o, su Windows, dalle variabili dell'utente.

    Il secondo caso copre la variabile impostata con «setx» dopo l'avvio della sessione: i
    programmi già aperti (Esplora risorse, terminali) non la vedono finché non vengono riavviati.
    """
    key = os.environ.get(ai_config.API_KEY_ENV, "").strip()
    if key or sys.platform != "win32":
        return key
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as reg:
            value, kind = winreg.QueryValueEx(reg, ai_config.API_KEY_ENV)
        if kind == winreg.REG_EXPAND_SZ:
            value = os.path.expandvars(value)
        return str(value).strip()
    except OSError:
        return ""


def has_api_key() -> bool:
    return bool(api_key())


MISSING_KEY_MESSAGE = (
    f"Chiave Gemini non trovata.\n\nImposta la variabile d'ambiente {ai_config.API_KEY_ENV} con la "
    "chiave gratuita creata su aistudio.google.com/apikey, ad esempio dal Prompt dei comandi:\n\n"
    f'    setx {ai_config.API_KEY_ENV} "la-tua-chiave"\n\n'
    "poi chiudi e riapri l'app."
)

_client_lock = threading.Lock()
_client_cache: tuple | None = None   # (hash della chiave, client) per riusare la connessione


def _client():
    from google import genai
    from google.genai import types

    key = api_key()
    if not key:
        raise DataError(MISSING_KEY_MESSAGE)
    global _client_cache
    with _client_lock:
        if _client_cache is None or _client_cache[0] != hash(key):
            options = types.HttpOptions(
                timeout=ai_config.TIMEOUT_SECONDS * 1000,
                retry_options=types.HttpRetryOptions(attempts=ai_config.RETRY_ATTEMPTS,
                                                     initial_delay=1.0, max_delay=8.0,
                                                     http_status_codes=RETRY_STATUS_CODES))
            _client_cache = (hash(key), genai.Client(api_key=key, http_options=options))
        return _client_cache[1]


# --------------------------------------------------------------------------- #
# Allegati
# --------------------------------------------------------------------------- #
@dataclass
class Attachment:
    name: str
    kind: str          # "image" | "pdf" | "text"
    mime: str
    data: bytes = b""
    text: str = ""

    @property
    def size(self) -> int:
        return len(self.data) if self.data else len(self.text.encode("utf-8"))


def load_attachment(path: str) -> Attachment:
    name = os.path.basename(path)
    ext = os.path.splitext(path)[1].lower()
    try:
        size = os.path.getsize(path)
        if ext in IMAGE_EXTS:
            if size > MAX_IMAGE_BYTES:
                raise DataError(f"L'immagine '{name}' supera i 5 MB consentiti: riducila e riprova.")
            with open(path, "rb") as fh:
                return Attachment(name, "image", IMAGE_EXTS[ext], data=fh.read())
        if ext == ".pdf":
            if size > MAX_PDF_BYTES:
                raise DataError(f"Il PDF '{name}' supera i 15 MB consentiti.")
            with open(path, "rb") as fh:
                data = fh.read()
            if not data.startswith(b"%PDF"):
                raise DataError(f"'{name}' non è un PDF valido.")
            return Attachment(name, "pdf", "application/pdf", data=data)
        if ext in TEXT_EXTS or (mimetypes.guess_type(path)[0] or "").startswith("text/"):
            with open(path, "rb") as fh:
                raw = fh.read(MAX_TEXT_CHARS * 4)
            for enc in ("utf-8-sig", "cp1252", "latin-1"):
                try:
                    text = raw.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            if "\x00" in text:
                raise DataError(f"'{name}' sembra un file binario, non un file di testo.")
            if len(text) > MAX_TEXT_CHARS:
                text = text[:MAX_TEXT_CHARS] + "\n[... file troncato ...]"
            return Attachment(name, "text", "text/plain", text=text)
    except OSError as exc:
        raise DataError(f"Impossibile leggere '{name}':\n{exc}") from exc
    raise DataError(f"Tipo di file non supportato: '{name}'.\n\n"
                    "Puoi allegare immagini (PNG, JPG, GIF, WEBP), PDF e file di testo "
                    "(TXT, CSV, MD, JSON, PY, SQL…).")


def check_total_size(attachments: list[Attachment]) -> None:
    if sum(a.size for a in attachments) > MAX_TOTAL_BYTES:
        raise DataError("Gli allegati superano complessivamente i 18 MB: rimuovine qualcuno.")


# --------------------------------------------------------------------------- #
# Contesto del dataset
# --------------------------------------------------------------------------- #
def dataset_context(df, source: str, max_rows: int = 20) -> str:
    lines = [f"Dataset corrente: {os.path.basename(source) or 'senza nome'} — "
             f"{len(df)} righe × {df.shape[1]} colonne", "", "Colonne (nome | dtype | % nulli):"]
    for c in df.columns[:200]:
        lines.append(f"- {c} | {df[c].dtype} | {df[c].isna().mean() * 100:.1f}%")
    num = df.select_dtypes("number")
    if not num.empty:
        lines += ["", "Statistiche numeriche:", num.describe().T.round(4).to_csv()]
    lines += ["", f"Prime {min(max_rows, len(df))} righe (CSV):",
              df.head(max_rows).to_csv(index=False)[:20000]]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Conversazione
# --------------------------------------------------------------------------- #
@dataclass
class Turn:
    role: str                                   # "user" | "assistant"
    text: str
    attachments: list[Attachment] = field(default_factory=list)
    context: str = ""


def _contents(history: list[Turn]):
    """Converte la cronologia nel formato Gemini inviando solo il necessario.

    - solo gli ultimi MAX_HISTORY_MESSAGES messaggi;
    - allegati e contesto del dataset solo dell'ultima domanda che li conteneva (così le
      domande successive possono ancora riferirsi a quei file), non di quelle precedenti.
    """
    from google.genai import types

    recent = history[-ai_config.MAX_HISTORY_MESSAGES:]
    while recent and recent[0].role != "user":
        recent = recent[1:]
    last_files = max((i for i, t in enumerate(recent) if t.attachments), default=-1)
    last_ctx = max((i for i, t in enumerate(recent) if t.context), default=-1)
    out = []
    for i, t in enumerate(recent):
        if t.role == "assistant":
            out.append(types.Content(role="model", parts=[types.Part(text=t.text)]))
            continue
        parts = []
        if t.attachments and i != last_files:
            names = ", ".join(a.name for a in t.attachments)
            parts.append(types.Part(text=f"[In questo messaggio erano allegati: {names}]"))
        elif i == last_files:
            for a in t.attachments:
                if a.kind == "text":
                    parts.append(types.Part(text=f"--- File: {a.name} ---\n{a.text}\n--- Fine file ---"))
                else:
                    parts.append(types.Part.from_bytes(data=a.data, mime_type=a.mime))
        text = t.text
        if t.context and i == last_ctx:
            text = f"<contesto_dataset>\n{t.context}\n</contesto_dataset>\n\n{text}"
        parts.append(types.Part(text=text))
        out.append(types.Content(role="user", parts=parts))
    return out


# --------------------------------------------------------------------------- #
# Errori
# --------------------------------------------------------------------------- #
@dataclass
class QuotaInfo:
    daily: bool = False
    limit: str = ""                 # numero massimo di richieste, se indicato da Google
    delay: float | None = None      # secondi di attesa suggeriti da Google


def _quota_info(exc) -> QuotaInfo:
    """Legge dai dettagli dell'errore 429 il tipo di limite e l'attesa suggerita."""
    info = QuotaInfo()
    details = getattr(exc, "details", None)
    items = details.get("error", {}).get("details", []) if isinstance(details, dict) else []
    for d in items if isinstance(items, list) else []:
        kind = str(d.get("@type", ""))
        if kind.endswith("QuotaFailure"):
            for v in d.get("violations", []):
                if "PerDay" in str(v.get("quotaId", "")):
                    info.daily = True
                    info.limit = str(v.get("quotaValue", "")) or info.limit
        elif kind.endswith("RetryInfo"):
            try:
                info.delay = float(str(d.get("retryDelay", "")).rstrip("s"))
            except ValueError:
                pass
    msg = str(getattr(exc, "message", "") or "")
    if "PerDay" in msg or (info.delay is not None and info.delay > 120):
        info.daily = True
    return info


def _api_error(exc, quota: QuotaInfo | None = None) -> DataError:
    code = getattr(exc, "code", None)
    status = str(getattr(exc, "status", "") or "")
    msg = str(getattr(exc, "message", None) or exc)
    low = msg.lower()
    if "api key not valid" in low or "api_key_invalid" in low or code == 401:
        return DataError(f"La chiave in {ai_config.API_KEY_ENV} non è valida o è stata revocata.\n\n"
                         "Creane una nuova su aistudio.google.com/apikey, aggiorna la variabile "
                         "d'ambiente e riavvia l'app.")
    if code == 429 or status == "RESOURCE_EXHAUSTED":
        quota = quota or _quota_info(exc)
        if quota.daily:
            from datetime import datetime, timedelta

            limit = f" ({quota.limit} richieste al giorno per {MODEL})" if quota.limit else ""
            when = (f"Hai esaurito il limite giornaliero gratuito{limit}. Si azzera ogni giorno "
                    "alle 9:00 italiane (mezzanotte nel Pacifico)")
            if quota.delay:
                when += f": potrai riprovare dalle {datetime.now() + timedelta(seconds=quota.delay):%H:%M}"
            when += "."
        else:
            when = "Hai raggiunto il limite di richieste al minuto: attendi un minuto e riprova."
        return DataError("Limite del piano gratuito Gemini raggiunto.\n\n" + when +
                         "\n\nPer consumare meno, raggruppa le domande in un unico messaggio. Non serve "
                         "attivare la fatturazione; l'utilizzo è visibile su ai.dev/rate-limit.")
    if "location is not supported" in low or "user location" in low:
        return DataError("La Gemini API non è disponibile dalla tua posizione geografica o rete.")
    if code == 403:
        return DataError(f"La chiave in {ai_config.API_KEY_ENV} non ha accesso alla Gemini API.\n\n"
                         "Verifica in aistudio.google.com/apikey che la chiave sia attiva.")
    if code == 404:
        return DataError(f"Il modello '{MODEL}' non è disponibile per la tua chiave.\n\n"
                         "Aggiorna GEMINI_MODEL in ai_config.py con un modello Flash attuale.")
    if code in (400, 413) and ("size" in low or "too large" in low or "exceeds" in low):
        return DataError("La richiesta è troppo grande: riduci gli allegati o disattiva il contesto "
                         "del dataset.")
    if code and code >= 500:
        return DataError("Il servizio Gemini è temporaneamente non disponibile (ritentato "
                         f"{ai_config.RETRY_ATTEMPTS} volte).\n\nRiprova tra qualche minuto.")
    return DataError(f"Errore della Gemini API ({code or status}): {msg}")


def _network_error() -> DataError:
    return DataError("Impossibile contattare Gemini: controlla la connessione a Internet.\n\n"
                     "Se la connessione funziona, un firewall, un proxy o una VPN potrebbero "
                     "bloccare generativelanguage.googleapis.com.")


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #
def stream_reply(history: list[Turn], cancel: threading.Event | None = None):
    """Generatore di eventi: ("text", frammento) e ("note", messaggio informativo)."""
    import httpx
    from google.genai import errors, types

    cancel = cancel or threading.Event()
    client = _client()
    config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)
    contents = _contents(history)
    finish = None
    waited = False
    try:
        while True:
            produced = False
            try:
                for chunk in client.models.generate_content_stream(model=MODEL, contents=contents,
                                                                   config=config):
                    if cancel.is_set():
                        yield ("note", "Risposta interrotta.")
                        return
                    feedback = getattr(chunk, "prompt_feedback", None)
                    if feedback is not None and getattr(feedback, "block_reason", None):
                        yield ("note", "Gemini ha bloccato la richiesta per i suoi filtri di sicurezza.")
                        return
                    if chunk.candidates:
                        finish = chunk.candidates[0].finish_reason or finish
                    text = chunk.text
                    if text:
                        produced = True
                        yield ("text", text)
                break
            except errors.APIError as exc:
                quota = _quota_info(exc) if exc.code == 429 else None
                # Limite al minuto con attesa breve: un solo nuovo tentativo automatico.
                if (quota and not quota.daily and not produced and not waited
                        and quota.delay is not None and quota.delay <= 60):
                    waited = True
                    yield ("wait", quota.delay)
                    if cancel.wait(quota.delay + 1):
                        yield ("note", "Risposta interrotta.")
                        return
                    continue
                raise _api_error(exc, quota) from exc
    except (httpx.TransportError, OSError) as exc:
        if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
            raise DataError(f"Gemini non ha risposto entro {ai_config.TIMEOUT_SECONDS} secondi.\n\n"
                            "Riprova; se succede spesso, controlla la connessione.") from exc
        raise _network_error() from exc
    reason = str(getattr(finish, "name", finish) or "")
    if reason == "MAX_TOKENS":
        yield ("note", "Risposta troncata: raggiunta la lunghezza massima.")
    elif reason in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"):
        yield ("note", "Risposta interrotta dai filtri di sicurezza di Gemini.")
