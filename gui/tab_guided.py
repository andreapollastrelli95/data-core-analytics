"""Tab — Analisi guidata: dalla domanda al report verificato (controllo dati → analisi →
revisore → report), con dizionario dati e procedure salvate."""
from __future__ import annotations

import os
import queue
import threading
import tkinter as tk

import customtkinter as ctk

import agent_engine as agent
import ai_engine as ai
import ai_config
import context_store as store
import report_engine as rep
from cleaner import DataError, pick
from gui import markdown
from gui import widgets as W

NEW_QUESTION = "(domanda libera)"
PHASE_NAMES = {"controllo": "CONTROLLO DATI", "analisi": "ANALISI", "revisione": "REVISORE",
               "report": "REPORT"}
INTRO = """## Analisi guidata

Scrivi una domanda di business (o scegli una procedura) e premi **Avvia analisi guidata**. \
L'app esegue quattro passaggi e ti consegna un report che puoi difendere numero per numero:

1. **Controllo dati** — duplicati, valori mancanti, anomalie. In locale, senza IA.
2. **Analisi** — Gemini sceglie i calcoli, l'app li esegue sui tuoi dati con i suoi motori.
3. **Revisore** — i calcoli vengono rieseguiti, i totali riconciliati e ogni numero del testo \
confrontato con i risultati; poi una seconda lettura critica indipendente.
4. **Report** — risposta in breve, evidenze, ipotesi e limiti dichiarati.

Ogni passo finisce nel registro con il codice Python equivalente.

**Solo controllo dati** esegue il primo passaggio senza chiave e senza inviare nulla.
"""


class DictionaryDialog(ctk.CTkToplevel):
    """Editor del dizionario dati del dataset corrente."""

    def __init__(self, app, df, source: str, dictionary: dict, on_saved):
        super().__init__(app)
        self.app, self.source, self.on_saved = app, source, on_saved
        self.title(f"Dizionario dati — {os.path.basename(source) or 'dataset'}")
        self.geometry("760x680")
        self.minsize(620, 480)
        self.transient(app)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        W.hint(self, "Spiega all'assistente che cosa significano i dati. Queste definizioni vengono "
                     "date al modello a ogni analisi guidata e hanno la precedenza sulle sue ipotesi. "
                     "Sono salvate sul tuo PC e valgono per ogni file con questo nome.",
               wrap=700).grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 6))
        ctk.CTkLabel(self, text="Che cosa contiene il dataset", font=(W.FONT, 13, "bold"),
                     anchor="w").grid(row=1, column=0, sticky="ew", padx=16)
        self.description = ctk.CTkTextbox(self, height=54, wrap="word")
        self.description.grid(row=2, column=0, sticky="ew", padx=16, pady=(2, 8))
        self.description.insert("1.0", dictionary.get("descrizione", ""))

        body = ctk.CTkScrollableFrame(self, label_text="Significato delle colonne")
        body.grid(row=3, column=0, sticky="nsew", padx=16, pady=4)
        body.grid_columnconfigure(1, weight=1)
        self.entries: dict[str, ctk.CTkEntry] = {}
        for i, col in enumerate(df.columns[:150]):
            name = str(col)
            ctk.CTkLabel(body, text=name if len(name) <= 30 else name[:29] + "…", anchor="w",
                         width=210).grid(row=i, column=0, sticky="w", padx=(4, 8), pady=2)
            entry = ctk.CTkEntry(body, placeholder_text=f"{df[col].dtype}")
            entry.grid(row=i, column=1, sticky="ew", padx=(0, 4), pady=2)
            if dictionary.get("colonne", {}).get(name):
                entry.insert(0, dictionary["colonne"][name])
            self.entries[name] = entry
        self.other_columns = {k: v for k, v in dictionary.get("colonne", {}).items()
                              if k not in self.entries}

        ctk.CTkLabel(self, text="Metriche — una per riga: «nome: definizione»",
                     font=(W.FONT, 13, "bold"), anchor="w").grid(row=4, column=0, sticky="ew",
                                                                 padx=16, pady=(8, 0))
        self.metrics = ctk.CTkTextbox(self, height=74, wrap="word")
        self.metrics.grid(row=5, column=0, sticky="ew", padx=16, pady=2)
        self.metrics.insert("1.0", store.format_metrics(dictionary.get("metriche", [])))
        ctk.CTkLabel(self, text="Regole di business — una per riga", font=(W.FONT, 13, "bold"),
                     anchor="w").grid(row=6, column=0, sticky="ew", padx=16, pady=(8, 0))
        self.rules = ctk.CTkTextbox(self, height=74, wrap="word")
        self.rules.grid(row=7, column=0, sticky="ew", padx=16, pady=2)
        self.rules.insert("1.0", "\n".join(dictionary.get("regole", [])))

        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.grid(row=8, column=0, sticky="ew", padx=16, pady=(10, 14))
        ctk.CTkButton(bar, text="Annulla", width=110, fg_color="transparent", border_width=1,
                      text_color=("gray10", "gray90"), command=self.destroy).pack(side="right")
        ctk.CTkButton(bar, text="Salva dizionario", width=150,
                      command=app.guard(self._save)).pack(side="right", padx=(0, 8))
        self.bind("<Escape>", lambda _e: self.destroy())
        self.after(60, self._raise)

    def _raise(self):
        try:
            self.lift()
            self.focus_force()
        except tk.TclError:
            pass

    def collect(self) -> dict:
        columns = dict(self.other_columns)
        columns.update({name: e.get().strip() for name, e in self.entries.items() if e.get().strip()})
        return {"descrizione": self.description.get("1.0", "end").strip(), "colonne": columns,
                "metriche": store.parse_metrics(self.metrics.get("1.0", "end")),
                "regole": store.parse_rules(self.rules.get("1.0", "end"))}

    def _save(self):
        dictionary = self.collect()
        store.save_dictionary(self.source, dictionary)
        self.on_saved()
        self.destroy()


class GuidedTab:
    def __init__(self, app, parent):
        self.app = app
        self.run: agent.GuidedRun | None = None
        self.worker: threading.Thread | None = None
        self.cancel = threading.Event()
        self.events: queue.Queue = queue.Queue()
        self.dictionary = store.empty_dictionary()
        self._steps_shown = 0
        parent.grid_columnconfigure(1, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        self._build_controls(parent)
        self._build_output(parent)
        W.register_themed(self)
        self.restyle(W.palette(W.is_dark()))
        self._load_procedures()
        self._show_markdown(INTRO)
        self.refresh_key_status()
        self._refresh_dictionary()

    # ------------------------------------------------------------------ UI --
    def _build_controls(self, parent):
        app = self.app
        side = ctk.CTkScrollableFrame(parent, width=390, fg_color="transparent")
        side.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        side.grid_columnconfigure(0, weight=1)

        sec = self._section(side, 0, "Domanda", "Scegli una procedura oppure scrivi la domanda a "
                                                "cui l'analisi deve rispondere.")
        self.procedure = W.option_row(sec, 2, "Procedura", [NEW_QUESTION], self._pick_procedure,
                                      width=230)
        self.question = ctk.CTkTextbox(sec, height=130, wrap="word")
        self.question.grid(row=3, column=0, columnspan=2, sticky="ew", padx=12, pady=(4, 6))
        self.with_rows = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(sec, text="Mostra al modello anche 5 righe di esempio",
                        variable=self.with_rows).grid(row=4, column=0, columnspan=2, sticky="w",
                                                      padx=12, pady=(0, 8))
        self.btn_run = ctk.CTkButton(sec, text="Avvia analisi guidata", height=38,
                                     font=(W.FONT, 14, "bold"), fg_color="#6d4ac2",
                                     hover_color="#573a9c", command=app.guard(self._start))
        self.btn_run.grid(row=5, column=0, columnspan=2, sticky="ew", padx=12, pady=2)
        row = ctk.CTkFrame(sec, fg_color="transparent")
        row.grid(row=6, column=0, columnspan=2, sticky="ew", padx=12, pady=(4, 10))
        row.grid_columnconfigure((0, 1), weight=1)
        self.btn_quality = ctk.CTkButton(row, text="Solo controllo dati", fg_color="gray40",
                                         hover_color="gray30", command=app.guard(self._quality_only))
        self.btn_quality.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.btn_stop = ctk.CTkButton(row, text="Interrompi", fg_color="#b23b3b",
                                      hover_color="#8f2f2f", state="disabled",
                                      command=self._stop)
        self.btn_stop.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.key_status = ctk.CTkLabel(sec, text="", anchor="w", justify="left", wraplength=350)
        self.key_status.grid(row=7, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 10))

        sec = self._section(side, 1, "Procedure salvate", "Salva la domanda corrente per "
                                                          "rieseguirla con un clic su qualsiasi "
                                                          "dataset.")
        row = ctk.CTkFrame(sec, fg_color="transparent")
        row.grid(row=2, column=0, columnspan=2, sticky="ew", padx=12, pady=(2, 10))
        row.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(row, text="Salva come procedura…", fg_color="gray40", hover_color="gray30",
                      command=app.guard(self._save_procedure)).grid(row=0, column=0, sticky="ew",
                                                                    padx=(0, 4))
        ctk.CTkButton(row, text="Elimina procedura", fg_color="gray40", hover_color="gray30",
                      command=app.guard(self._delete_procedure)).grid(row=0, column=1, sticky="ew",
                                                                      padx=(4, 0))

        sec = self._section(side, 2, "Dizionario dati", "Significato delle colonne, definizione "
                                                        "delle metriche e regole di business.")
        self.dict_status = W.hint(sec, "", wrap=350)
        self.dict_status.configure(text_color=("gray10", "gray90"))
        self.dict_status.grid(row=2, column=0, columnspan=2, sticky="ew", padx=12)
        self.btn_dict = ctk.CTkButton(sec, text="Modifica dizionario…", fg_color="gray40",
                                      hover_color="gray30", command=app.guard(self._edit_dictionary))
        self.btn_dict.grid(row=3, column=0, columnspan=2, sticky="ew", padx=12, pady=(6, 10))

        sec = self._section(side, 3, "Risultato", "Disponibile al termine di un'analisi.")
        self.btn_html = ctk.CTkButton(sec, text="Esporta report HTML…",
                                      command=app.guard(lambda: self._export("html")))
        self.btn_html.grid(row=2, column=0, columnspan=2, sticky="ew", padx=12, pady=2)
        row = ctk.CTkFrame(sec, fg_color="transparent")
        row.grid(row=3, column=0, columnspan=2, sticky="ew", padx=12, pady=(4, 10))
        row.grid_columnconfigure((0, 1), weight=1)
        self.btn_md = ctk.CTkButton(row, text="Salva Markdown…", fg_color="gray40",
                                    hover_color="gray30",
                                    command=app.guard(lambda: self._export("md")))
        self.btn_md.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.btn_copy = ctk.CTkButton(row, text="Copia report", fg_color="gray40",
                                      hover_color="gray30", command=self._copy)
        self.btn_copy.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.result_buttons = [self.btn_html, self.btn_md, self.btn_copy]
        app.data_widgets += [self.btn_run, self.btn_quality, self.btn_dict]

    @staticmethod
    def _section(parent, row: int, title: str, subtitle: str) -> ctk.CTkFrame:
        """Sezione del pannello laterale, con il sottotitolo a capo sulla larghezza del pannello."""
        frame = W.section(parent, title)
        frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        W.hint(frame, subtitle, wrap=350).grid(row=1, column=0, columnspan=2, sticky="ew", padx=12,
                                               pady=(0, 6))
        return frame

    def _build_output(self, parent):
        box = ctk.CTkFrame(parent, fg_color="transparent")
        box.grid(row=0, column=1, sticky="nsew")
        box.grid_columnconfigure(0, weight=1)
        box.grid_rowconfigure(1, weight=1)
        self.phase = ctk.CTkLabel(box, text="", anchor="w", font=(W.FONT, 13, "bold"))
        self.phase.grid(row=0, column=0, sticky="ew", pady=(0, 2))
        self.sub = ctk.CTkTabview(box, height=10)
        self.sub.grid(row=1, column=0, sticky="nsew")
        tab = self.sub.add("Report")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        self.out = tk.Text(tab, wrap="word", borderwidth=0, highlightthickness=0, padx=14, pady=10,
                           state="disabled", cursor="arrow")
        self.out.grid(row=0, column=0, sticky="nsew")
        sb = ctk.CTkScrollbar(tab, command=self.out.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.out.configure(yscrollcommand=sb.set)
        tab = self.sub.add("Passi e registro")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        self.steps_box = W.TextPanel(tab)
        self.steps_box.grid(row=0, column=0, sticky="nsew")
        tab = self.sub.add("Codice")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        self.code_box = W.TextPanel(tab)
        self.code_box.configure(wrap="none")
        self.code_box.grid(row=0, column=0, sticky="nsew")

    def restyle(self, c: dict):
        markdown.setup_tags(self.out, c)

    # ----------------------------------------------------------- Stato ------
    def running(self) -> bool:
        return bool(self.worker and self.worker.is_alive())

    def refresh_key_status(self) -> bool:
        ok = ai.has_api_key()
        self.key_status.configure(
            text=(f"✓ Gemini ({ai.MODEL}) · al massimo {agent.MAX_MODEL_CALLS} chiamate per analisi."
                  if ok else f"Chiave non trovata: imposta la variabile {ai_config.API_KEY_ENV} per "
                             "l'analisi con IA. «Solo controllo dati» funziona comunque."),
            text_color=("#15803d", "#4ade80") if ok else ("#a85d00", "#e0a040"))
        return ok

    def on_data_changed(self):
        self._refresh_dictionary()

    def on_controls(self, state: str):
        busy = self.running()
        for b in (self.btn_run, self.btn_quality):
            b.configure(state="disabled" if busy else state)
        self.btn_stop.configure(state="normal" if busy else "disabled")
        ready = "normal" if self.run is not None and not busy else "disabled"
        for b in self.result_buttons:
            b.configure(state=ready)

    def _refresh_dictionary(self):
        if self.app.df is None:
            self.dictionary = store.empty_dictionary()
            self.dict_status.configure(text="Carica un dataset per compilarne il dizionario.")
            return
        try:
            self.dictionary = store.load_dictionary(self.app.source)
        except DataError:
            self.dictionary = store.empty_dictionary(self.app.source)
        d = self.dictionary
        if store.is_empty(d):
            self.dict_status.configure(text="Non compilato: il modello dovrà intuire il significato "
                                            "delle colonne dai nomi.")
        else:
            self.dict_status.configure(
                text="Compilato: " + ", ".join([
                    pick(len(d["colonne"]), "{n} colonna descritta", "{n} colonne descritte"),
                    pick(len(d["metriche"]), "{n} metrica", "{n} metriche"),
                    pick(len(d["regole"]), "{n} regola", "{n} regole")]) + ".")

    # ------------------------------------------------------- Procedure ------
    def _load_procedures(self, select: str | None = None):
        try:
            self.procedures = store.list_procedures()
        except DataError:
            self.procedures = [{**p, "utente": False} for p in store.BUILTIN]
        names = [NEW_QUESTION] + [p["nome"] for p in self.procedures]
        self.procedure.configure(values=names)
        self.procedure.set(select if select in names else NEW_QUESTION)

    def _pick_procedure(self, name: str):
        proc = next((p for p in self.procedures if p["nome"] == name), None)
        if proc:
            self.question.delete("1.0", "end")
            self.question.insert("1.0", proc["domanda"])

    def _question(self) -> str:
        return self.question.get("1.0", "end").strip()

    def _save_procedure(self):
        question = self._question()
        if not question:
            raise DataError("Scrivi prima la domanda da salvare.")
        dialog = ctk.CTkInputDialog(title="Salva procedura", text="Nome della procedura:")
        name = (dialog.get_input() or "").strip()
        if not name:
            return
        store.save_procedure(name, question)
        self._load_procedures(select=name)
        self.app.status.configure(text=f"Procedura «{name}» salvata.")

    def _delete_procedure(self):
        name = self.procedure.get()
        if name == NEW_QUESTION:
            raise DataError("Scegli nel menu la procedura da eliminare.")
        if not self.app.confirm(f"Eliminare la procedura «{name}»?"):
            return
        store.delete_procedure(name)
        self._load_procedures()
        self.app.status.configure(text=f"Procedura «{name}» eliminata.")

    # ------------------------------------------------------ Dizionario ------
    def _edit_dictionary(self):
        if not self.app.require_data():
            return
        DictionaryDialog(self.app, self.app.df, self.app.source, self.dictionary,
                         self._refresh_dictionary)

    # ---------------------------------------------------------- Avvio -------
    def _quality_only(self):
        if self.running() or not self.app.require_data():
            return
        df, source, summary = self.app.df, self.app.source, self.app.summary
        self.app.run_task("Controllo qualità dei dati…",
                          lambda: agent.run_quality_only(df, source, summary), self._finished)

    def _start(self):
        app = self.app
        if self.running() or not app.require_data():
            return
        question = self._question()
        if not question:
            raise DataError("Scrivi la domanda a cui l'analisi deve rispondere, oppure scegli una "
                            "procedura.")
        if not self.refresh_key_status():
            raise DataError(ai.MISSING_KEY_MESSAGE)
        rows = 5 if self.with_rows.get() else 0
        sent = ["nomi, tipi, minimo e massimo delle colonne e i valori di quelle con poche categorie",
                "i risultati dei calcoli richiesti dal modello (tabelle fino a 40 righe, che "
                "possono contenere righe del dataset)"]
        if rows:
            sent.append("le prime 5 righe del dataset")
        if not store.is_empty(self.dictionary):
            sent.append("il dizionario dati")
        if not app.confirm("L'analisi guidata invia a Google Gemini:\n\n• " + "\n• ".join(sent) +
                           "\n\nCon il piano gratuito Google può usare questi contenuti per migliorare "
                           "i suoi servizi. Se il dataset contiene dati personali, finanziari o "
                           "sanitari, annulla.\n\nL'analisi usa al massimo "
                           f"{agent.MAX_MODEL_CALLS} richieste della quota gratuita. Avviare?",
                           warning=True):
            app.status.configure(text="Analisi annullata.")
            return
        df, source, summary = app.df, app.source, app.summary
        df2 = app.secondary["df"] if app.secondary else None
        dictionary = dict(self.dictionary)
        self.cancel.clear()
        self.run = None
        self._steps_shown = 0
        self.steps_box.set_text("")
        self.code_box.set_text("")
        self._show_markdown(f"## {question}\n\nAnalisi in corso…")
        cancel, events = self.cancel, self.events

        def llm(messages, system, **kw):
            return ai.generate(messages, system, cancel=cancel,
                               on_wait=lambda s: events.put(("wait", s)), **kw)

        def work():
            try:
                run = agent.run_guided(df, question, source, llm=llm, df2=df2, summary=summary,
                                       dictionary=dictionary, sample_rows=rows,
                                       on_event=lambda k, p: events.put((k, p)), cancel=cancel)
                events.put(("done", run))
            except BaseException as exc:  # noqa: BLE001 - mostrato come popup nel thread GUI
                events.put(("error", exc))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()
        app.update_controls()
        app.after(80, self._poll)

    def _stop(self):
        self.cancel.set()
        self.phase.configure(text="Interruzione in corso…")

    def _poll(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "phase":
                    self.phase.configure(text=payload)
                elif kind == "step":
                    self._append_step(payload)
                elif kind == "wait":
                    self.phase.configure(text="Limite al minuto raggiunto: nuovo tentativo tra "
                                              f"{int(payload) + 1} secondi…")
                elif kind == "done":
                    self._finished(payload)
                    return
                elif kind == "error":
                    self._failed(payload)
                    return
        except queue.Empty:
            pass
        if self.running() or not self.events.empty():
            self.app.after(100, self._poll)

    # --------------------------------------------------------- Risultato ----
    def _append_step(self, step: agent.Step):
        self._steps_shown += 1
        lines = [f"[{step.n}] {PHASE_NAMES.get(step.phase, step.phase.upper())} · {step.title}"]
        if step.result is not None:
            if step.result.error:
                lines.append("    ERRORE: " + step.result.error.replace("\n", " "))
            else:
                lines.append("    " + step.result.description)
                table = step.result.table
                if table is not None and len(table):
                    preview = table.head(12).to_string(index=False, max_cols=10, max_colwidth=28)
                    lines += ["      " + ln for ln in preview.splitlines()]
                    if len(table) > 12:
                        lines.append(f"      … {len(table)} righe in tutto")
                elif step.result.meta.get("text"):
                    lines += ["      " + ln for ln in step.result.meta["text"].splitlines()]
        elif step.detail:
            lines += ["    " + ln for ln in step.detail.splitlines()]
        self.steps_box.append("\n".join(lines) + "\n\n")

    def _finished(self, run: agent.GuidedRun):
        app = self.app
        self.run = run
        self.worker = None
        if self._steps_shown == 0:
            self.steps_box.set_text("")
            for step in run.steps:
                self._append_step(step)
        self._show_markdown(f"## {run.question}\n\n{run.report_md}" if run.ai_used else run.report_md)
        self.code_box.set_text(agent.run_script(run) if run.tool_steps() else
                               "# Il controllo dati non genera codice: usa i pulsanti della scheda\n"
                               "# «Pulizia & Validazione» per correggere i problemi trovati.")
        for step in run.tool_steps():
            if not step.failed and step.result.code:
                app.record(f"Analisi guidata — {step.result.description}", step.result.code,
                           step.result.imports)
        if run.ai_used:
            verdict = agent.VERDICT_LABELS.get(run.verdict.get("verdetto", ""), "")
            missing = len(run.grounding.get("non_trovati", []))
            app.log(f"Analisi guidata «{run.question[:70]}»: {len(run.tool_steps())} calcoli, "
                    f"{run.model_calls} chiamate a Gemini, revisore: {verdict.lower()}"
                    + (f", {missing} numeri senza riscontro" if missing else ""))
            self.phase.configure(text=f"Completata · revisore: {verdict} · {run.model_calls} "
                                      f"chiamate a Gemini · {run.seconds:.0f} s")
        else:
            app.log(f"Controllo qualità dei dati: {len(run.findings)} segnalazioni")
            self.phase.configure(text="Controllo dati completato (nessun dato inviato)")
        self.sub.set("Report")
        app.update_controls()

    def _failed(self, exc: BaseException):
        self.worker = None
        interrupted = isinstance(exc, agent.Interrupted) or self.cancel.is_set()
        self.phase.configure(text="Analisi interrotta." if interrupted else "Analisi non riuscita.")
        self._show_markdown(INTRO)
        self.app.update_controls()
        if not interrupted:
            self.app.show_error(exc)

    def _show_markdown(self, md: str):
        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        markdown.render(self.out, md)
        self.out.configure(state="disabled")
        self.out.yview_moveto(0)

    # ------------------------------------------------------------ Export ----
    def _document(self) -> str:
        run = self.run
        head = (f"# Analisi guidata\n\n**Domanda:** {run.question}\n\n"
                f"_{os.path.basename(run.source) or 'dataset'} · {run.rows} righe × {run.cols} "
                f"colonne · {run.started:%d/%m/%Y %H:%M}_\n\n") if run.ai_used else ""
        return head + run.report_md + "\n"

    def _copy(self):
        if self.run is None:
            return
        self.app.clipboard_clear()
        self.app.clipboard_append(self._document())
        self.app.status.configure(text="Report copiato negli appunti (formato Markdown).")

    def _export(self, kind: str):
        run = self.run
        if run is None:
            raise DataError("Esegui prima un'analisi.")
        stem = os.path.splitext(os.path.basename(run.source or "dataset"))[0] or "dataset"
        types = {"Pagina HTML": ".html"} if kind == "html" else {"Markdown": ".md"}
        path = self.app.ask_save("Salva report dell'analisi guidata", f"analisi_{stem}", types)
        if not path:
            return
        if kind == "html":
            rep.guided_html(path, run)
        else:
            try:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(self._document())
            except OSError as exc:
                raise DataError(f"Impossibile salvare il report:\n{exc}") from exc
        self.app.log(f"Esportato il report dell'analisi guidata in '{os.path.basename(path)}'")
        try:
            os.startfile(path)  # noqa: S606 - apre il file con l'app predefinita
        except (OSError, AttributeError):
            self.app.info(f"Report salvato in:\n{path}")
