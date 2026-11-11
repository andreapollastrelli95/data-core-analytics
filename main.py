"""Data Core Analytics — piattaforma desktop di Data Cleaning, Analytics e Machine Learning.

Avvio:            python main.py   (oppure run.bat)
Creazione .exe:   build_exe.bat    (vedi README.md)
"""
from __future__ import annotations

import gc
import multiprocessing
import os
import queue
import sys
import threading
import traceback
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox

import customtkinter as ctk

import cleaner
from cleaner import DataError
from codegen import Op
from gui import widgets as W
from gui.tab_clean import CleanTab
from gui.tab_insights import InsightsTab
from gui.tab_ml import MLTab
from gui.tab_report import ReportTab
from gui.tab_timeseries import TimeSeriesTab
from gui.tab_transform import TransformTab

APP_NAME = "Data Core Analytics"
MAX_HISTORY = 15
TABS = [
    ("Pulizia & Validazione", CleanTab),
    ("Trasformazione & SQL", TransformTab),
    ("Insights & Statistica", InsightsTab),
    ("Time Series", TimeSeriesTab),
    ("Machine Learning", MLTab),
    ("Codice & Report", ReportTab),
]
OPEN_TYPES = [("File di dati", "*.csv *.tsv *.txt *.xlsx *.xlsm *.xls *.parquet"),
              ("CSV / testo", "*.csv *.tsv *.txt"), ("Excel", "*.xlsx *.xlsm *.xls"),
              ("Parquet", "*.parquet"), ("Tutti i file", "*.*")]


class SheetDialog(ctk.CTkToplevel):
    """Finestra modale per scegliere il foglio di un file Excel."""

    def __init__(self, master, sheets: list[str]):
        super().__init__(master)
        self.title("Seleziona foglio")
        self.resizable(False, False)
        self.result: str | None = None
        self.transient(master)
        ctk.CTkLabel(self, text="Il file contiene più fogli.\nQuale vuoi caricare?",
                     font=(W.FONT, 14), justify="left").pack(padx=24, pady=(20, 10), anchor="w")
        self.choice = ctk.CTkOptionMenu(self, values=sheets, width=260, dynamic_resizing=False)
        self.choice.pack(padx=24, pady=6)
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.pack(padx=24, pady=(14, 20), fill="x")
        ctk.CTkButton(bar, text="Annulla", width=110, fg_color="transparent", border_width=1,
                      text_color=("gray10", "gray90"), command=self.destroy).pack(side="right")
        ctk.CTkButton(bar, text="Carica", width=110, command=self._ok).pack(side="right", padx=(0, 8))
        self.bind("<Return>", lambda _e: self._ok())
        self.bind("<Escape>", lambda _e: self.destroy())
        self.after(50, self._grab)

    def _grab(self):
        self.lift()
        self.focus_force()
        try:
            self.grab_set()
        except tk.TclError:
            pass

    def _ok(self):
        self.result = self.choice.get()
        self.destroy()


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        scale = ctk.ScalingTracker.get_window_scaling(self)
        width = min(1320, int(self.winfo_screenwidth() / scale * 0.94))
        height = min(840, int(self.winfo_screenheight() / scale * 0.84))
        self.geometry(f"{width}x{height}")
        self.minsize(min(1040, width), min(600, height))

        # Stato condiviso
        self.df = None
        self.original_df = None
        self.summary: dict | None = None
        self.source: str = ""                    # file o origine del dataset principale
        self.file_size = 0
        self.history: list[tuple] = []          # [(DataFrame, ops, descrizione)]
        self.ops: list[Op] = []                 # registro per il Code Generator
        self.log_entries: list[str] = []
        self.secondary: dict | None = None      # {"df", "name", "code"}
        self.last_ml = None                     # ultimo MLResult (per report e predizioni)
        self.dirty = False
        self.busy = False
        self.data_widgets: list = []
        self.ai_panel = None                    # creato al primo utilizzo
        self.ai_visible = False
        self._last_mode = ctk.get_appearance_mode()

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._build_header()
        self.tabview = ctk.CTkTabview(self, anchor="nw")
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=12, pady=(4, 0))
        self.tabs = []
        for name, cls in TABS:
            self.tabs.append(cls(self, self.tabview.add(name)))
        self._build_statusbar()
        W.apply_theme(self)
        self.update_controls()

        self.bind_all("<Control-o>", lambda _e: self.open_file())
        self.bind_all("<Control-s>", lambda _e: self.save_dataset())
        self.bind_all("<Control-z>", self._ctrl_z)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(1000, self._watch_appearance)

    # ------------------------------------------------------------------ UI --
    def _build_header(self):
        bar = ctk.CTkFrame(self, corner_radius=0)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        bar.grid_columnconfigure(5, weight=1)
        ctk.CTkLabel(bar, text=APP_NAME, font=(W.FONT, 21, "bold")).grid(
            row=0, column=0, padx=(18, 18), pady=12)
        self.btn_open = ctk.CTkButton(bar, text="Apri file…", height=36, width=120,
                                      font=(W.FONT, 14, "bold"), command=self.open_file)
        self.btn_open.grid(row=0, column=1, padx=4)
        self.btn_undo = ctk.CTkButton(bar, text="Annulla ultima", width=120, height=36,
                                      fg_color="gray40", hover_color="gray30", command=self.undo)
        self.btn_undo.grid(row=0, column=2, padx=4)
        self.btn_reset = ctk.CTkButton(bar, text="Ripristina originale", width=150, height=36,
                                       fg_color="gray40", hover_color="gray30",
                                       command=self.reset_original)
        self.btn_reset.grid(row=0, column=3, padx=4)
        self.btn_save = ctk.CTkButton(bar, text="Salva dataset…", width=130, height=36,
                                      fg_color="#2d8a4e", hover_color="#236b3c",
                                      command=self.save_dataset)
        self.btn_save.grid(row=0, column=4, padx=4)
        self.data_widgets += [self.btn_reset, self.btn_save]
        self.btn_ai = ctk.CTkButton(bar, text="Assistente IA", width=130, height=36,
                                    fg_color="#6d4ac2", hover_color="#573a9c",
                                    font=(W.FONT, 14, "bold"), command=self.toggle_ai)
        self.btn_ai.grid(row=0, column=5, sticky="e", padx=8)
        ctk.CTkLabel(bar, text="Tema:").grid(row=0, column=6, padx=(10, 6))
        self.theme_switch = ctk.CTkSegmentedButton(bar, values=["Chiaro", "Scuro", "Sistema"],
                                                   command=self._change_theme)
        self.theme_switch.set("Sistema")
        self.theme_switch.grid(row=0, column=7, padx=(0, 18))

    def _build_statusbar(self):
        bar = ctk.CTkFrame(self, corner_radius=0, height=30)
        bar.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        bar.grid_columnconfigure(0, weight=1)
        self.status = ctk.CTkLabel(bar, text="", anchor="w")
        self.status.grid(row=0, column=0, sticky="ew", padx=14, pady=4)
        self.progress = ctk.CTkProgressBar(bar, mode="indeterminate", width=180)
        self.update_status()

    # ------------------------------------------------------ Assistente IA --
    def toggle_ai(self):
        """Mostra/nasconde il pannello dell'assistente, affiancato a qualsiasi scheda."""
        if self.ai_panel is None:
            from gui.ai_panel import AIPanel

            try:
                self.ai_panel = AIPanel(self, self)
            except Exception as exc:  # noqa: BLE001 - l'app resta utilizzabile senza assistente
                self.ai_panel = None
                self.show_error(exc)
                return
        self.ai_visible = not self.ai_visible
        if self.ai_visible:
            self.ai_panel.grid(row=1, column=1, sticky="ns", padx=(0, 12), pady=(10, 0))
            self.btn_ai.configure(text="Chiudi assistente")
        else:
            self.ai_panel.grid_remove()
            self.btn_ai.configure(text="Assistente IA")

    # --------------------------------------------------------------- Tema --
    def _change_theme(self, choice: str):
        ctk.set_appearance_mode({"Chiaro": "light", "Scuro": "dark"}.get(choice, "system"))
        self._last_mode = ctk.get_appearance_mode()
        W.apply_theme(self)

    def _watch_appearance(self):
        if ctk.get_appearance_mode() != self._last_mode:  # tema di sistema cambiato a runtime
            self._last_mode = ctk.get_appearance_mode()
            W.apply_theme(self)
        self.after(1000, self._watch_appearance)

    # ---------------------------------------------------- Esecuzione task --
    def run_task(self, status: str, func, on_success, on_error=None):
        """Esegue `func` in un thread separato; `on_success(risultato)` gira nel thread GUI."""
        if self.busy:
            self.status.configure(text="Attendi il termine dell'operazione in corso…")
            return
        self._set_busy(True, status)
        # Raccoglie qui eventuali cicli con oggetti Tk: se lo facesse il garbage collector
        # dentro il thread di lavoro, Tk solleverebbe errori e rallenterebbe l'interfaccia.
        gc.collect()
        results: queue.Queue = queue.Queue(maxsize=1)

        def worker():
            try:
                results.put(("ok", func()))
            except BaseException as exc:  # noqa: BLE001 - riportato all'utente
                results.put(("err", exc))

        threading.Thread(target=worker, daemon=True).start()
        self.after(40, self._poll_task, results, on_success, on_error)

    def _poll_task(self, results: queue.Queue, on_success, on_error):
        try:
            kind, payload = results.get_nowait()
        except queue.Empty:
            self.after(40, self._poll_task, results, on_success, on_error)
            return
        self._set_busy(False)
        if kind == "ok":
            try:
                on_success(payload)
            except Exception as exc:  # noqa: BLE001
                self.show_error(exc)
        else:
            if on_error:
                on_error(payload)
            self.show_error(payload)

    def _set_busy(self, busy: bool, status: str = ""):
        self.busy = busy
        if busy:
            self.status.configure(text=status)
            self.progress.grid(row=0, column=1, padx=14)
            self.progress.start()
            self.configure(cursor="watch")
        else:
            self.progress.stop()
            self.progress.grid_remove()
            self.configure(cursor="")
            self.update_status()
        self.update_controls()

    def guard(self, func):
        """Avvolge un callback dei pulsanti: gli errori diventano popup, mai crash."""
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                self.show_error(exc)
        return wrapper

    # ----------------------------------------------------------- Dialoghi --
    def show_error(self, exc: BaseException):
        if isinstance(exc, DataError):
            messagebox.showerror(APP_NAME, str(exc), parent=self)
        elif isinstance(exc, MemoryError):
            messagebox.showerror(APP_NAME, "Memoria insufficiente per completare l'operazione.\n\n"
                                 "Prova a ridurre il dataset (filtri, campionamento SQL).", parent=self)
        else:
            if sys.stderr:
                traceback.print_exception(type(exc), exc, exc.__traceback__)
            messagebox.showerror(APP_NAME, "Si è verificato un errore imprevisto:\n\n"
                                 f"{type(exc).__name__}: {exc}\n\nL'applicazione resta utilizzabile.",
                                 parent=self)

    def report_callback_exception(self, exc, val, tb):  # errori nei callback Tk
        self.show_error(val)

    def info(self, message: str):
        messagebox.showinfo(APP_NAME, message, parent=self)

    def confirm(self, message: str, warning: bool = False) -> bool:
        return messagebox.askyesno(APP_NAME, message, parent=self,
                                   icon="warning" if warning else "question")

    def ask_open(self, title: str, filetypes=None) -> str:
        return filedialog.askopenfilename(parent=self, title=title, filetypes=filetypes or OPEN_TYPES)

    def ask_save(self, title: str, initial: str, types: dict[str, str]) -> str:
        """types: {nome formato: estensione}; aggiunge l'estensione se manca."""
        type_var = tk.StringVar(self, value=next(iter(types)))
        first_ext = next(iter(types.values()))
        path = filedialog.asksaveasfilename(
            parent=self, title=title, initialfile=initial, defaultextension=first_ext,
            typevariable=type_var, filetypes=[(n, f"*{e}") for n, e in types.items()])
        if not path:
            return ""
        if os.path.splitext(path)[1].lower() not in types.values():
            path += types.get(type_var.get(), first_ext)
        return path

    def require_data(self) -> bool:
        if self.df is None:
            messagebox.showwarning(APP_NAME, "Carica prima un dataset (pulsante «Apri file…»).",
                                   parent=self)
            return False
        return not self.busy

    # ------------------------------------------------------------- Colonne --
    def column_labels(self, kind: str = "all") -> list[str]:
        if self.df is None:
            return []
        out = []
        for c in self.df.columns:
            s = self.df[c]
            if kind == "numeric" and not cleaner.is_numeric(s):
                continue
            if kind == "categorical" and cleaner.is_numeric(s):
                continue
            out.append(str(c))
        return out

    def col(self, label):
        """Converte l'etichetta mostrata nella GUI nel nome di colonna originale."""
        if label is None or self.df is None:
            return None
        for c in self.df.columns:
            if str(c) == label:
                return c
        raise DataError(f"La colonna '{label}' non esiste più nel dataset.")

    def cols(self, labels: list) -> list:
        return [self.col(lbl) for lbl in labels]

    # ---------------------------------------------------------- Caricamento --
    def open_file(self):
        if self.busy:
            return
        if self.dirty and not self.confirm("Ci sono modifiche non salvate.\n"
                                           "Vuoi aprire un altro file e perderle?", warning=True):
            return
        path = self.ask_open("Seleziona un file di dati")
        if not path:
            return
        self.run_task("Lettura dei fogli…", lambda: cleaner.excel_sheets(path),
                      lambda sheets: self._choose_sheet_and_load(path, sheets))

    def choose_sheet(self, sheets: list[str]):
        """Restituisce il foglio scelto, None se non serve, False se annullato."""
        if len(sheets) > 1:
            dlg = SheetDialog(self, sheets)
            self.wait_window(dlg)
            return dlg.result if dlg.result is not None else False
        return sheets[0] if sheets else None

    def _choose_sheet_and_load(self, path: str, sheets: list[str]):
        sheet = self.choose_sheet(sheets)
        if sheet is False:
            return

        def task():
            df, code = cleaner.load_file(path, sheet)
            return df, code, cleaner.summarize(df)

        def done(res):
            df, code, summary = res
            where = f" (foglio '{sheet}')" if sheet and len(sheets) > 1 else ""
            self.load_dataset(df, summary, path, f"Caricato '{os.path.basename(path)}'{where}",
                              code, file_size=os.path.getsize(path))

        self.run_task(f"Caricamento di {os.path.basename(path)}…", task, done)

    def load_dataset(self, df, summary, source: str, description: str, code: str,
                     imports: tuple = (), file_size: int = 0):
        """Imposta un nuovo dataset principale (da file o database)."""
        self.df = self.original_df = df
        self.source = source
        self.file_size = file_size
        self.history.clear()
        self.ops = [Op("load", description, code, imports)]
        self.last_ml = None
        self.dirty = False
        self.log_entries.clear()
        self.log(f"{description}: {cleaner.fmt_int(len(df))} righe × {df.shape[1]} colonne")
        self.refresh(summary)
        self.tabview.set(TABS[0][0])

    # ---------------------------------------------------------- Operazioni --
    def apply_op(self, status: str, op, on_done=None):
        """Applica `op(df) -> OpResult` in background; aggiorna cronologia, codice e log."""
        if not self.require_data():
            return
        df = self.df

        def task():
            res = op(df)
            return res, (cleaner.summarize(res.df) if res.changed else None)

        def done(payload):
            res, summary = payload
            if not res.changed:
                self.status.configure(text=res.message)
                self.info(res.message)
                return
            self._push_history(res.message)
            self.df = res.df
            self.dirty = True
            self.ops.append(Op("transform", res.message, res.code, res.imports))
            self.log(res.message)
            self.refresh(summary)
            if on_done:
                on_done(res)

        self.run_task(status, task, done)

    def record(self, description: str, code: str, imports: tuple = ()):
        """Registra un'analisi (non modifica i dati) nel Code Generator."""
        if self.ops and self.ops[-1].code == code:
            return
        self.ops.append(Op("analysis", description, code, tuple(imports)))
        self.notify_ops()

    def notify_ops(self):
        for tab in self.tabs:
            if hasattr(tab, "on_ops_changed"):
                tab.on_ops_changed()

    def _push_history(self, description: str):
        self.history.append((self.df, list(self.ops), description))
        del self.history[:-MAX_HISTORY]

    def _ctrl_z(self, event):
        # Nei campi di testo Ctrl+Z resta l'annullamento del testo.
        if isinstance(event.widget, (tk.Text, tk.Entry)):
            return
        self.undo()

    def undo(self):
        if self.busy or not self.history:
            return
        prev_df, prev_ops, description = self.history.pop()
        self.run_task("Annullamento…", lambda: cleaner.summarize(prev_df),
                      lambda s: self._restore(prev_df, prev_ops, s,
                                              f"Annullata operazione: {description}"))

    def reset_original(self):
        if self.busy or self.original_df is None or self.df is self.original_df:
            return
        if not self.confirm("Ripristinare il dataset originale?\nPotrai comunque annullare "
                            "questa azione."):
            return
        self._push_history("Ripristino del dataset originale")
        original, ops = self.original_df, self.ops[:1]
        self.run_task("Ripristino…", lambda: cleaner.summarize(original),
                      lambda s: self._restore(original, ops, s, "Ripristinato il dataset originale"))

    def _restore(self, df, ops, summary, message: str):
        self.df = df
        self.ops = ops
        self.dirty = True
        self.log(message)
        self.refresh(summary)

    # -------------------------------------------------------- Salvataggio --
    def save_dataset(self):
        if self.busy or self.df is None:
            return
        stem = os.path.splitext(os.path.basename(self.source or "dataset"))[0] or "dataset"
        path = self.ask_save("Salva dataset elaborato", f"{stem}_elaborato",
                             {"CSV": ".csv", "Excel": ".xlsx", "Parquet": ".parquet"})
        if not path:
            return
        if self.source and os.path.abspath(path) == os.path.abspath(self.source):
            if not self.confirm("Stai per sovrascrivere il file originale.\nContinuare?", True):
                return
        df = self.df

        def done(note):
            self.dirty = False
            self.log(f"Esportato il dataset in '{os.path.basename(path)}': "
                     f"{cleaner.fmt_int(len(df))} righe × {df.shape[1]} colonne{note}")
            self.record("Esportazione del dataset", cleaner.export_code(path))
            self.update_status()
            self.info(f"File salvato correttamente:\n{path}")

        self.run_task(f"Salvataggio di {os.path.basename(path)}…",
                      lambda: cleaner.export(df, path), done)

    # ------------------------------------------------------------- Log ----
    def log(self, message: str):
        entry = f"[{datetime.now():%H:%M:%S}]  {message}"
        self.log_entries.append(entry)
        for tab in self.tabs:
            if hasattr(tab, "on_log"):
                tab.on_log(entry)

    # ------------------------------------------------------ Aggiornamento --
    def refresh(self, summary: dict):
        self.summary = summary
        for tab in self.tabs:
            try:
                tab.on_data_changed()
            except Exception as exc:  # noqa: BLE001 - un tab non deve bloccare gli altri
                self.show_error(exc)
        self.notify_ops()
        self.update_status()
        self.update_controls()

    def update_status(self):
        name = os.path.basename(self.source) if self.source else ""
        self.title(f"{APP_NAME} — {name}{' *' if self.dirty else ''}" if name else APP_NAME)
        if self.df is None:
            self.status.configure(text="Pronto. Apri un file per iniziare (Ctrl+O).")
            return
        self.status.configure(
            text=f"{cleaner.fmt_int(len(self.df))} righe × {self.df.shape[1]} colonne   |   "
                 f"operazioni annullabili: {len(self.history)}   |   "
                 + ("modifiche non salvate" if self.dirty else "nessuna modifica da salvare"))

    def update_controls(self):
        state = "normal" if self.df is not None and not self.busy else "disabled"
        for w in self.data_widgets:
            try:
                (w.set_state if isinstance(w, W.ColumnList) else
                 lambda s, w=w: w.configure(state=s))(state)
            except tk.TclError:
                pass
        self.btn_open.configure(state="disabled" if self.busy else "normal")
        self.btn_undo.configure(state="normal" if self.history and not self.busy else "disabled")
        for tab in self.tabs:
            if hasattr(tab, "on_controls"):
                tab.on_controls(state)

    # ------------------------------------------------------------ Uscita --
    def _on_close(self):
        if self.busy and not self.confirm("Un'operazione è in corso. Uscire comunque?", True):
            return
        if self.dirty and not self.confirm("Ci sono modifiche non salvate.\nUscire senza salvare?",
                                           True):
            return
        self.destroy()


def main():
    multiprocessing.freeze_support()  # necessario per l'eseguibile PyInstaller
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme("blue")
    App().mainloop()


if __name__ == "__main__":
    main()
