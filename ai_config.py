"""Configurazione dell'assistente IA (unico punto in cui si sceglie il modello)."""

# Modello Flash più recente disponibile nel piano gratuito della Gemini API
# (verificato su aistudio.google.com/docs/models e /docs/pricing il 05/11/2026).
GEMINI_MODEL = "gemini-3.8-flash"

# Nome della variabile d'ambiente che contiene la chiave API (mai scritta nel codice).
API_KEY_ENV = "GEMINI_API_KEY"

TIMEOUT_SECONDS = 90          # attesa massima tra due blocchi di risposta
RETRY_ATTEMPTS = 3            # tentativi totali per gli errori temporanei (rete, 5xx)
MAX_HISTORY_MESSAGES = 10     # messaggi di cronologia inviati (5 domande + 5 risposte)
