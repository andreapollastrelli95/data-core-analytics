"""Tab 6 — Codice & Report: Code Generator, report HTML/Excel, esportazione e log."""
from __future__ import annotations

import os
import tkinter as tk

import customtkinter as ctk

import codegen
import report_engine as rep
from cleaner import DataError
from gui import widgets as W


class ReportTab:
    def __init__(self, app, parent):
        self.app = app
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        self.sub = ctk.CTkTabview(parent, height=10)
        self.sub.grid(row=0, column=0, sticky="nsew")
        self._build_code(self.sub.add("Codice Python"))
        self._build_export(self.sub.add("Report & Export"))
        self._build_log(self.sub.add("Log operazioni"))

    # ------------------------------------------------------------ Codice
    def _build_code(self, tab):
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)
        bar = ctk.CTkFrame(tab, fg_color="transparent")
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.include_analysis = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(bar, text="Includi analisi (statistiche, grafici, SQL, time series, ML)",
                        variable=self.include_analysis, command=self.on_ops_changed).pack(side="left")
        ctk.CTkButton(bar, text="Salva script .py…", width=150, fg_color="gray40",
                      hover_color="gray30", command=self.app.guard(self._save_script)).pack(side="right")
        ctk.CTkButton(bar, text="Copia negli appunti", width=160,
                      command=self._copy).pack(side="right", padx=8)
        self.code_box = W.TextPanel(tab)
        self.code_box.configure(wrap="none")
        self.code_box.grid(row=1, column=0, sticky="nsew")
        self.code_info = W.hint(tab, "", wrap=1000)
        self.code_info.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        self.on_ops_changed()

    def _script(self) -> str:
        return codegen.build_script(self.app.ops, self.include_analysis.get())

    def on_ops_changed(self):
        self.code_box.set_text(self._script())
        n_t = sum(op.kind == "transform" for op in self.app.ops)
        n_a = sum(op.kind == "analysis" for op in self.app.ops)
        self.code_info.configure(text=f"{n_t} trasformazioni e {n_a} analisi registrate. Lo script si "
                                      "aggiorna automaticamente a ogni operazione (anche con "
                                      "«Annulla ultima»).")

    def _copy(self):
        self.app.clipboard_clear()
        self.app.clipboard_append(self._script())
        self.app.status.configure(text="Script copiato negli appunti.")

    def _save_script(self):
        path = self.app.ask_save("Salva script Python", "pipeline_dati", {"Script Python": ".py"})
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self._script())
        except OSError as exc:
            raise DataError(f"Impossibile salvare lo script:\n{exc}") from exc
        self.app.info(f"Script salvato in:\n{path}")

    # ------------------------------------------------------------ Export
    def _build_export(self, tab):
        app = self.app
        tab.grid_columnconfigure(0, weight=1)
        cards = [
            ("Dataset elaborato", "Esporta i dati correnti in CSV (UTF-8, compatibile con Excel), "
                                  "Excel .xlsx o Parquet. Il file originale non viene modificato.",
             "Salva dataset con nome…", app.save_dataset),
            ("Report HTML", "Pagina autonoma con indicatori chiave, struttura delle colonne, "
                            "statistiche, heatmap delle correlazioni, valori mancanti, ultimo modello "
                            "ML e log delle operazioni. Si apre in qualsiasi browser.",
             "Genera report HTML…", app.guard(lambda: self._report("html"))),
            ("Report Excel multi-foglio", "Cartella di lavoro con i fogli Riepilogo, Colonne, "
                                          "Statistiche, Categoriche, Correlazioni, ML, Log e Dati.",
             "Genera report Excel…", app.guard(lambda: self._report("xlsx"))),
        ]
        for i, (title, text, button, cmd) in enumerate(cards):
            sec = W.section(tab, title, text)
            sec.grid(row=i, column=0, sticky="ew", pady=(0, 10))
            b = ctk.CTkButton(sec, text=button, width=220, height=36, command=cmd,
                              fg_color="#2d8a4e" if i == 0 else None,
                              hover_color="#236b3c" if i == 0 else None)
            b.grid(row=2, column=0, sticky="w", padx=12, pady=(4, 12))
            app.data_widgets.append(b)
        self.open_after = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(tab, text="Apri il report al termine", variable=self.open_after).grid(
            row=len(cards), column=0, sticky="w", pady=4)

    def _report(self, kind: str):
        app = self.app
        if not app.require_data():
            return
        stem = os.path.splitext(os.path.basename(app.source or "dataset"))[0] or "dataset"
        types = {"Pagina HTML": ".html"} if kind == "html" else {"Excel": ".xlsx"}
        path = app.ask_save("Salva report", f"report_{stem}", types)
        if not path:
            return
        df, summary, log, ml, source = app.df, app.summary, list(app.log_entries), app.last_ml, app.source

        def task():
            if kind == "html":
                rep.html_report(path, df, summary, source, log, ml)
                return ""
            return rep.excel_report(path, df, summary, source, log, ml)

        def done(note):
            app.log(f"Generato report {kind.upper()} '{os.path.basename(path)}'{note}")
            if self.open_after.get():
                try:
                    os.startfile(path)  # noqa: S606 - apre il file con l'app predefinita
                except (OSError, AttributeError):
                    app.info(f"Report salvato in:\n{path}")
            else:
                app.info(f"Report salvato in:\n{path}")

        app.run_task("Generazione del report…", task, done)

    # --------------------------------------------------------------- Log
    def _build_log(self, tab):
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        self.log_box = W.TextPanel(tab)
        self.log_box.grid(row=0, column=0, sticky="nsew")
        ctk.CTkButton(tab, text="Esporta log (.txt)…", width=170, fg_color="gray40",
                      hover_color="gray30", command=self.app.guard(self._export_log)).grid(
            row=1, column=0, sticky="e", pady=(6, 0))

    def on_log(self, entry: str):
        if len(self.app.log_entries) == 1:
            self.log_box.set_text("")
        self.log_box.append(entry + "\n")

    def _export_log(self):
        if not self.app.log_entries:
            self.app.info("Il log è vuoto.")
            return
        path = self.app.ask_save("Esporta log", "log_operazioni", {"Testo": ".txt"})
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(self.app.log_entries) + "\n")
        except OSError as exc:
            raise DataError(f"Impossibile salvare il log:\n{exc}") from exc

    def on_data_changed(self):
        pass
