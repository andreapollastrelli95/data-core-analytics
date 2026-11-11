"""Pannello laterale dell'Assistente IA (Google Gemini), disponibile da tutte le schede."""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import filedialog

import customtkinter as ctk

import ai_config
import ai_engine as ai
from cleaner import DataError
from gui import markdown
from gui import widgets as W

ATTACH_TYPES = [("File supportati", "*.png *.jpg *.jpeg *.gif *.webp *.pdf *.txt *.csv *.tsv *.md "
                                    "*.json *.py *.sql *.log *.xml *.html *.yaml *.yml"),
                ("Immagini", "*.png *.jpg *.jpeg *.gif *.webp"), ("PDF", "*.pdf"),
                ("Testo", "*.txt *.csv *.tsv *.md *.json *.py *.sql *.log"), ("Tutti i file", "*.*")]
PRIVACY_NOTE = ("Piano gratuito: Google può usare richieste e risposte per migliorare i suoi servizi. "
                "Non inviare dati personali, finanziari o sanitari.")


class AIPanel(ctk.CTkFrame):
    def __init__(self, master, app, width: int = 440):
        super().__init__(master, width=width)
        self.app = app
        self.history: list[ai.Turn] = []
        self.attachments: list[ai.Attachment] = []
        self.current = ""                 # risposta in arrivo
        self.notes: list[str] = []
        self.worker: threading.Thread | None = None
        self.cancel = threading.Event()
        self.events: queue.Queue = queue.Queue()
        self._dirty = False
        self._privacy_ok = False          # avviso sul piano gratuito confermato in questa sessione
        self.grid_propagate(False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self._build()
        W.register_themed(self)
        self.restyle(W.palette(W.is_dark()))
        self.refresh_key_status()
        self._render_all()

    # ------------------------------------------------------------------ UI --
    def _build(self):
        head = ctk.CTkFrame(self, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        head.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(head, text="Assistente IA", font=(W.FONT, 16, "bold"), anchor="w").grid(
            row=0, column=0, sticky="w")
        ctk.CTkButton(head, text="Nascondi", width=80, height=26, fg_color="gray40",
                      hover_color="gray30", command=self.app.toggle_ai).grid(row=0, column=1)
        W.hint(head, f"Google Gemini · {ai.MODEL}", wrap=400).grid(
            row=1, column=0, columnspan=2, sticky="w")

        info = ctk.CTkFrame(self)
        info.grid(row=1, column=0, sticky="ew", padx=10, pady=4)
        info.grid_columnconfigure(0, weight=1)
        self.key_status = ctk.CTkLabel(info, text="", anchor="w", justify="left", wraplength=400)
        self.key_status.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 2))
        W.hint(info, PRIVACY_NOTE, wrap=400).grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 6))

        box = ctk.CTkFrame(self)
        box.grid(row=2, column=0, sticky="nsew", padx=10, pady=4)
        box.grid_columnconfigure(0, weight=1)
        box.grid_rowconfigure(0, weight=1)
        self.out = tk.Text(box, wrap="word", borderwidth=0, highlightthickness=0, padx=10, pady=8,
                           state="disabled", cursor="arrow")
        self.out.grid(row=0, column=0, sticky="nsew", padx=(4, 0), pady=4)
        sb = ctk.CTkScrollbar(box, command=self.out.yview)
        sb.grid(row=0, column=1, sticky="ns", pady=4)
        self.out.configure(yscrollcommand=sb.set)

        comp = ctk.CTkFrame(self, fg_color="transparent")
        comp.grid(row=3, column=0, sticky="ew", padx=10, pady=(4, 10))
        comp.grid_columnconfigure(0, weight=1)
        att = ctk.CTkFrame(comp, fg_color="transparent")
        att.grid(row=0, column=0, sticky="ew")
        att.grid_columnconfigure(2, weight=1)
        ctk.CTkButton(att, text="Allega file…", width=100, height=26, fg_color="gray40",
                      hover_color="gray30", command=self.app.guard(self._attach)).grid(row=0, column=0)
        ctk.CTkButton(att, text="Rimuovi", width=70, height=26, fg_color="gray40",
                      hover_color="gray30", command=self._clear_attachments).grid(row=0, column=1, padx=4)
        self.att_label = W.hint(att, "Nessun allegato", wrap=230)
        self.att_label.grid(row=0, column=2, sticky="w", padx=4)
        self.with_context = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(comp, text="Includi il contesto del dataset (schema, statistiche, 20 righe)",
                        variable=self.with_context).grid(row=1, column=0, sticky="w", pady=(6, 4))
        self.prompt = ctk.CTkTextbox(comp, height=80, wrap="word")
        self.prompt.grid(row=2, column=0, sticky="ew")
        self.prompt.bind("<Control-Return>", lambda _e: (self.app.guard(self._send)(), "break")[1])
        bar = ctk.CTkFrame(comp, fg_color="transparent")
        bar.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        bar.grid_columnconfigure(3, weight=1)
        self.btn_send = ctk.CTkButton(bar, text="Invia", width=80, command=self.app.guard(self._send))
        self.btn_send.grid(row=0, column=0)
        self.btn_stop = ctk.CTkButton(bar, text="Interrompi", width=86, fg_color="#b23b3b",
                                      hover_color="#8f2f2f", command=self.cancel.set, state="disabled")
        self.btn_stop.grid(row=0, column=1, padx=4)
        ctk.CTkButton(bar, text="Copia risposta", width=110, fg_color="gray40", hover_color="gray30",
                      command=self._copy).grid(row=0, column=2)
        ctk.CTkButton(bar, text="Nuova chat", width=86, fg_color="gray40", hover_color="gray30",
                      command=self._new_chat).grid(row=0, column=4)
        self.status = W.hint(comp, "Ctrl+Invio per inviare.", wrap=400)
        self.status.grid(row=4, column=0, sticky="ew", pady=(4, 0))

    def restyle(self, c: dict):
        markdown.setup_tags(self.out, c)

    def refresh_key_status(self) -> bool:
        """Controllo locale (nessuna chiamata di rete): la chiave è presente nell'ambiente?"""
        ok = ai.has_api_key()
        self.key_status.configure(
            text=(f"✓ Chiave letta dalla variabile {ai_config.API_KEY_ENV}." if ok else
                  f"Chiave non trovata: imposta la variabile d'ambiente {ai_config.API_KEY_ENV} "
                  "e riapri l'app (istruzioni nel messaggio che appare premendo «Invia»)."),
            text_color=("#15803d", "#4ade80") if ok else ("#a85d00", "#e0a040"))
        return ok

    # ------------------------------------------------------------ Allegati --
    def _attach(self):
        paths = filedialog.askopenfilenames(parent=self.app, title="Allega file", filetypes=ATTACH_TYPES)
        added = [ai.load_attachment(p) for p in paths]
        ai.check_total_size(self.attachments + added)
        self.attachments += added
        self._update_attachments()

    def _clear_attachments(self):
        self.attachments.clear()
        self._update_attachments()

    def _update_attachments(self):
        if not self.attachments:
            self.att_label.configure(text="Nessun allegato")
            return
        kinds = {"image": "immagine", "pdf": "PDF", "text": "testo"}
        self.att_label.configure(text=", ".join(f"{a.name} ({kinds[a.kind]})" for a in self.attachments))

    # ---------------------------------------------------------- Invio ------
    def _confirm_sending(self, context: bool) -> bool:
        """Avvisa prima di inviare dati a Google (piano gratuito: dati usati per migliorare i servizi)."""
        sent = [a.name for a in self.attachments] + (["contesto del dataset (colonne, statistiche, "
                                                      "20 righe)"] if context else [])
        if sent:
            return self.app.confirm(
                "Stai per inviare a Google Gemini:\n\n• " + "\n• ".join(sent) +
                "\n\nCon il piano gratuito Google può usare questi contenuti per migliorare i suoi "
                "servizi e alcuni possono essere letti da revisori. Se contengono dati personali, "
                "finanziari o sanitari, annulla e rimuovili.\n\nInviare?", warning=True)
        if not self._privacy_ok:
            self._privacy_ok = self.app.confirm(
                "Con il piano gratuito della Gemini API, Google può usare domande e risposte per "
                "migliorare i suoi servizi. Non scrivere dati personali, finanziari o sanitari.\n\n"
                "Continuare?")
            return self._privacy_ok
        return True

    def _send(self):
        if self.worker and self.worker.is_alive():
            return
        text = self.prompt.get("1.0", "end").strip()
        if not text:
            raise DataError("Scrivi una domanda o una descrizione per il modello.")
        if not self.refresh_key_status():
            raise DataError(ai.MISSING_KEY_MESSAGE)
        context = ""
        if self.with_context.get():
            if self.app.df is None:
                raise DataError("Nessun dataset caricato: disattiva «Includi il contesto del dataset» "
                                "oppure carica un file.")
            context = ai.dataset_context(self.app.df, self.app.source)
        if not self._confirm_sending(bool(context)):
            self.status.configure(text="Invio annullato.")
            return
        self.history.append(ai.Turn("user", text, list(self.attachments), context))
        self.prompt.delete("1.0", "end")
        self._clear_attachments()
        self.current, self.notes = "", []
        self.cancel.clear()
        self._set_streaming(True)
        self._render_all()
        history = list(self.history)

        def work():
            try:
                for kind, chunk in ai.stream_reply(history, self.cancel):
                    self.events.put((kind, chunk))
                self.events.put(("done", None))
            except Exception as exc:  # noqa: BLE001 - mostrato come popup nel thread GUI
                self.events.put(("error", exc))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()
        self.status.configure(text=f"In attesa della risposta di Gemini ({ai.MODEL})…")
        self.after(60, self._poll)

    def _poll(self):
        finished = False
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "text":
                    self.current += payload
                    self._dirty = True
                elif kind == "note":
                    self.notes.append(payload)
                    self._dirty = True
                elif kind == "wait":
                    self.status.configure(text=f"Limite al minuto raggiunto: nuovo tentativo automatico "
                                               f"tra {int(payload) + 1} secondi…")
                elif kind == "done":
                    finished = True
                elif kind == "error":
                    self._finish(error=payload)
                    return
        except queue.Empty:
            pass
        if self._dirty:
            self._dirty = False
            self._render_current()
        if finished:
            self._finish()
        elif (self.worker and self.worker.is_alive()) or not self.events.empty():
            self.after(80, self._poll)

    def _finish(self, error: BaseException | None = None):
        self._set_streaming(False)
        if error is not None:
            self.history.pop()  # la domanda non ha ricevuto risposta: si può riprovare
            self._render_all()
            self.status.configure(text="Richiesta non riuscita.")
            self.app.show_error(error)
            return
        if self.current:
            self.history.append(ai.Turn("assistant", self.current))
        elif self.history and self.history[-1].role == "user":
            self.history.pop()  # nessun testo ricevuto: la cronologia resta coerente
        self._render_all()
        self.status.configure(text=" ".join(self.notes) or "Risposta completata.")

    def _set_streaming(self, on: bool):
        self.btn_send.configure(state="disabled" if on else "normal")
        self.btn_stop.configure(state="normal" if on else "disabled")

    # -------------------------------------------------------- Rendering ----
    def _at_bottom(self) -> bool:
        return self.out.yview()[1] > 0.98

    def _render_all(self):
        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        if not self.history:
            self.out.insert("end", "Fai una domanda sui tuoi dati, allega immagini, PDF o file di "
                                   "testo, oppure chiedi codice pandas pronto all'uso.\n\n", ("quote",))
            self.out.insert("end", "Esempi:\n", ("bold",))
            for ex in ("Quali colonne hanno problemi di qualità e come le pulisco?",
                       "Suggerisci un modello ML adatto per prevedere la colonna target.",
                       "Spiega questo grafico (allega lo screenshot)."):
                self.out.insert("end", f"•  {ex}\n", ("bullet",))
        for t in self.history:
            self._render_turn(t)
        self.out.mark_set("stream_start", "end-1c")
        self.out.mark_gravity("stream_start", "left")
        if self.worker and self.worker.is_alive():
            self._render_current(scroll=False)
        self.out.configure(state="disabled")
        self.out.see("end")

    def _render_turn(self, t: ai.Turn):
        if t.role == "user":
            self.out.insert("end", "TU\n", ("role_user",))
            extras = [a.name for a in t.attachments] + (["contesto del dataset"] if t.context else [])
            if extras:
                self.out.insert("end", "Allegati: " + ", ".join(extras) + "\n", ("note",))
            self.out.insert("end", t.text + "\n", ("user_msg",))
        else:
            self.out.insert("end", "ASSISTENTE\n", ("role_ai",))
            markdown.render(self.out, t.text)

    def _render_current(self, scroll: bool = True):
        follow = self._at_bottom()
        self.out.configure(state="normal")
        self.out.delete("stream_start", "end")
        self.out.insert("end", "ASSISTENTE\n", ("role_ai",))
        if self.current:
            markdown.render(self.out, self.current)
        else:
            self.out.insert("end", "…\n", ("note",))
        for n in self.notes:
            self.out.insert("end", n + "\n", ("note",))
        self.out.configure(state="disabled")
        if scroll and follow:
            self.out.see("end")

    # -------------------------------------------------------------- Azioni --
    def _copy(self):
        last = next((t.text for t in reversed(self.history) if t.role == "assistant"), "")
        text = self.current if self.worker and self.worker.is_alive() else last
        if not text:
            self.status.configure(text="Nessuna risposta da copiare.")
            return
        self.app.clipboard_clear()
        self.app.clipboard_append(text)
        self.status.configure(text="Risposta copiata negli appunti (formato Markdown).")

    def _new_chat(self):
        if self.worker and self.worker.is_alive():
            self.cancel.set()
        self.history.clear()
        self.current, self.notes = "", []
        self._render_all()
        self.status.configure(text="Nuova conversazione.")
