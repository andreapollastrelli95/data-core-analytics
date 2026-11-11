"""Widget riutilizzabili: tabelle, grafici incorporati, elenchi di colonne, sezioni."""
from __future__ import annotations

import gc
import sys
import tkinter as tk
import weakref
from tkinter import ttk

import customtkinter as ctk
import numpy as np
import pandas as pd

FONT = "Segoe UI" if sys.platform == "win32" else "Helvetica"
MONO = "Consolas" if sys.platform == "win32" else "Courier"
MUTED = ("gray35", "gray65")
NONE = "(nessuna)"

_TABLES: "weakref.WeakSet[DataTable]" = weakref.WeakSet()
_LISTS: "weakref.WeakSet[ColumnList]" = weakref.WeakSet()
_FIGURES: "weakref.WeakSet[FigurePanel]" = weakref.WeakSet()
_THEMED: weakref.WeakSet = weakref.WeakSet()  # oggetti con metodo restyle(colori)


def register_themed(obj) -> None:
    _THEMED.add(obj)


def is_dark() -> bool:
    return ctk.get_appearance_mode() == "Dark"


def palette(dark: bool) -> dict:
    return {
        "bg": "#242424" if dark else "#ffffff",
        "alt": "#2c2c2c" if dark else "#f4f6f8",
        "fg": "#e8e8e8" if dark else "#1b1b1b",
        "head": "#353535" if dark else "#e4e7eb",
        "head_hover": "#404040" if dark else "#d6dadf",
        "sel": "#1f6aa5",
        "warn": "#e0a040" if dark else "#a85d00",
        "bad": "#ff6b6b" if dark else "#c02626",
        "good": "#4ade80" if dark else "#15803d",
        "muted": "#9aa3ad" if dark else "#5b6470",
        "link": "#60a5fa" if dark else "#1f6aa5",
        "code_bg": "#2f3238" if dark else "#eef0f3",
    }


def apply_theme(root) -> None:
    """Applica il tema corrente ai widget Tk/ttk non gestiti da CustomTkinter."""
    dark = is_dark()
    c = palette(dark)
    scale = ctk.ScalingTracker.get_widget_scaling(root)
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("Treeview", background=c["bg"], fieldbackground=c["bg"], foreground=c["fg"],
                    rowheight=int(25 * scale), borderwidth=0, font=(FONT, 10))
    style.configure("Treeview.Heading", background=c["head"], foreground=c["fg"], relief="flat",
                    borderwidth=0, font=(FONT, 10, "bold"), padding=(6, 4))
    style.map("Treeview", background=[("selected", c["sel"])], foreground=[("selected", "#ffffff")])
    style.map("Treeview.Heading", background=[("active", c["head_hover"])])
    style.layout("Treeview", [("Treeview.treearea", {"sticky": "nswe"})])
    for t in list(_TABLES):
        t.restyle(c)
    for lst in list(_LISTS):
        lst.restyle(c)
    for f in list(_FIGURES):
        f.rerender()
    for obj in list(_THEMED):
        obj.restyle(c)


# --------------------------------------------------------------------------- #
# Layout helper
# --------------------------------------------------------------------------- #
def section(parent, title: str, subtitle: str = "") -> ctk.CTkFrame:
    frame = ctk.CTkFrame(parent)
    frame.grid_columnconfigure(1, weight=1)
    ctk.CTkLabel(frame, text=title, font=(FONT, 15, "bold"), anchor="w").grid(
        row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=(10, 2 if subtitle else 6))
    if subtitle:
        ctk.CTkLabel(frame, text=subtitle, anchor="w", justify="left", text_color=MUTED,
                     wraplength=520).grid(row=1, column=0, columnspan=2, sticky="ew", padx=12,
                                          pady=(0, 6))
    return frame


def hint(parent, text: str, wrap: int = 520) -> ctk.CTkLabel:
    return ctk.CTkLabel(parent, text=text, anchor="w", justify="left", text_color=MUTED,
                        wraplength=wrap)


def option_row(parent, row: int, label: str, values: list, command=None,
               width: int = 220) -> ctk.CTkOptionMenu:
    lbl = ctk.CTkLabel(parent, text=label, anchor="w")
    lbl.grid(row=row, column=0, sticky="w", padx=12, pady=4)
    menu = ctk.CTkOptionMenu(parent, values=values or [NONE], command=command,
                             dynamic_resizing=False, width=width)
    menu.grid(row=row, column=1, sticky="ew", padx=12, pady=4)
    menu.label = lbl
    return menu


def entry_row(parent, row: int, label: str, default: str = "", placeholder: str = "",
              width: int = 220, **kw) -> ctk.CTkEntry:
    ctk.CTkLabel(parent, text=label, anchor="w").grid(row=row, column=0, sticky="w", padx=12, pady=4)
    entry = ctk.CTkEntry(parent, placeholder_text=placeholder, width=width, **kw)
    entry.grid(row=row, column=1, sticky="ew", padx=12, pady=4)
    if default:
        entry.insert(0, default)
    return entry


def set_options(menu: ctk.CTkOptionMenu, values: list, allow_none: bool = False) -> None:
    """Aggiorna i valori di un menu mantenendo la scelta corrente se ancora valida."""
    values = ([NONE] if allow_none else []) + list(values)
    if not values:
        values = [NONE]
    current = menu.get()
    menu.configure(values=values)
    menu.set(current if current in values else values[0])


def get_option(menu: ctk.CTkOptionMenu):
    value = menu.get()
    return None if value in (NONE, "") else value


def read_number(entry: ctk.CTkEntry, label: str, kind=float, minimum=None, maximum=None,
                default=None):
    from cleaner import DataError

    raw = entry.get().strip().replace(",", ".")
    if raw == "" and default is not None:
        return default
    try:
        value = kind(float(raw)) if kind is int else kind(raw)
    except ValueError as exc:
        raise DataError(f"«{label}» deve essere un numero (valore inserito: '{entry.get()}').") from exc
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        rng = f"tra {minimum} e {maximum}" if maximum is not None else f"almeno {minimum}"
        raise DataError(f"«{label}» deve essere {rng}.")
    return value


# --------------------------------------------------------------------------- #
# Tabella
# --------------------------------------------------------------------------- #
def format_cell(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "null"
        return f"{v:.6g}" if abs(v) < 1e15 else f"{v:.4e}"
    try:
        if pd.isna(v):
            return "null"
    except (TypeError, ValueError):
        pass
    text = str(v)
    return text if len(text) <= 120 else text[:117] + "…"


class DataTable(ctk.CTkFrame):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(self, show="headings", selectmode="browse")
        self.tree.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=(6, 0))
        ys = ctk.CTkScrollbar(self, command=self.tree.yview)
        xs = ctk.CTkScrollbar(self, orientation="horizontal", command=self.tree.xview)
        ys.grid(row=0, column=1, sticky="ns", pady=(6, 0))
        xs.grid(row=1, column=0, sticky="ew", padx=(6, 0), pady=(0, 4))
        self.tree.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        _TABLES.add(self)
        self.restyle(palette(is_dark()))

    def restyle(self, c: dict) -> None:
        self.tree.tag_configure("odd", background=c["alt"])
        self.tree.tag_configure("warn", foreground=c["warn"])
        self.tree.tag_configure("bad", foreground=c["bad"])
        self.tree.tag_configure("good", foreground=c["good"])

    def clear(self) -> None:
        self.tree.delete(*self.tree.get_children())
        self.tree.configure(columns=())

    def show(self, headers: list, rows, tags=None, anchors=None, widths=None) -> None:
        tree = self.tree
        tree.delete(*tree.get_children())
        ids = [f"c{i}" for i in range(len(headers))]
        tree.configure(columns=ids)
        rows = list(rows)
        for i, (cid, h) in enumerate(zip(ids, headers)):
            sample = [len(str(r[i])) for r in rows[:60]]
            width = widths[i] if widths else min(320, max(70, 8 * max([len(str(h)), *sample]) + 18))
            anchor = anchors[i] if anchors else "w"
            tree.heading(cid, text=str(h), anchor=anchor)
            tree.column(cid, width=width, minwidth=50, anchor=anchor, stretch=False)
        for j, row in enumerate(rows):
            t = list(tags[j]) if tags else []
            if j % 2:
                t.append("odd")
            tree.insert("", "end", values=list(row), tags=t)

    def show_df(self, df: pd.DataFrame, max_rows: int = 500) -> None:
        head = df.head(max_rows)
        numeric = [pd.api.types.is_numeric_dtype(head[c]) and not pd.api.types.is_bool_dtype(head[c])
                   for c in head.columns]
        rows = ([format_cell(v) for v in r] for r in head.itertuples(index=False, name=None))
        self.show([str(c) for c in head.columns], rows,
                  anchors=["e" if n else "w" for n in numeric])


# --------------------------------------------------------------------------- #
# Grafico matplotlib incorporato
# --------------------------------------------------------------------------- #
class FigurePanel(ctk.CTkFrame):
    """Mostra una figura matplotlib. `builder(dark) -> Figure` viene rieseguito al cambio tema."""

    def __init__(self, master, placeholder: str = "Il grafico apparirà qui.", **kw):
        super().__init__(master, **kw)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._builder = None
        self._canvas = None
        self._toolbar = None
        self._fig = None
        self._placeholder = ctk.CTkLabel(self, text=placeholder, text_color=MUTED)
        self._placeholder.grid(row=0, column=0)
        _FIGURES.add(self)

    def show(self, builder) -> None:
        fig = builder(is_dark())  # può sollevare eccezioni: le gestisce il chiamante
        self._builder = builder
        self._mount(fig)

    def rerender(self) -> None:
        if self._builder is None:
            return
        try:
            self._mount(self._builder(is_dark()))
        except Exception:  # noqa: BLE001 - il grafico precedente resta visibile
            pass

    def clear(self) -> None:
        self._builder = None
        self._unmount()
        self._placeholder.grid(row=0, column=0)

    def _unmount(self) -> None:
        # La figura va svuotata prima di distruggere la toolbar, che viene aggiornata da clear().
        if self._fig is not None:
            try:
                self._fig.clear()
            except tk.TclError:
                pass
            self._fig = None
        if self._toolbar is not None:
            self._toolbar.destroy()
            self._toolbar = None
        if self._canvas is not None:
            self._canvas.get_tk_widget().destroy()
            self._canvas = None
        # Gli oggetti Tk della toolbar formano cicli di riferimenti: vanno raccolti qui, nel
        # thread della GUI, e non da un thread di lavoro (Tk lo vieta e rallenterebbe l'app).
        gc.collect()

    def _mount(self, fig) -> None:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        self._unmount()
        self._placeholder.grid_remove()
        self._fig = fig
        self._canvas = FigureCanvasTkAgg(fig, master=self)
        widget = self._canvas.get_tk_widget()
        widget.configure(highlightthickness=0, bg=palette(is_dark())["bg"])
        widget.grid(row=0, column=0, sticky="nsew", padx=4, pady=(4, 0))
        self._toolbar = NavigationToolbar2Tk(self._canvas, self, pack_toolbar=False)
        bg = palette(is_dark())["head"]
        self._toolbar.configure(background=bg)
        for child in self._toolbar.winfo_children():
            try:
                child.configure(background=bg)
            except tk.TclError:
                pass
        self._toolbar.grid(row=1, column=0, sticky="ew", padx=4, pady=(0, 4))
        self._canvas.draw()


# --------------------------------------------------------------------------- #
# Elenco colonne a selezione multipla
# --------------------------------------------------------------------------- #
class ColumnList(ctk.CTkFrame):
    def __init__(self, master, title: str = "", buttons: bool = True, height: int = 8, **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._values: list[str] = []
        if title:
            ctk.CTkLabel(self, text=title, anchor="w").grid(row=0, column=0, columnspan=2, sticky="ew")
        box = ctk.CTkFrame(self)
        box.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=4)
        box.grid_columnconfigure(0, weight=1)
        box.grid_rowconfigure(0, weight=1)
        self.listbox = tk.Listbox(box, selectmode="multiple", activestyle="none", borderwidth=0,
                                  highlightthickness=0, exportselection=False, font=(FONT, 11),
                                  height=height)
        self.listbox.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=6)
        sb = ctk.CTkScrollbar(box, command=self.listbox.yview)
        sb.grid(row=0, column=1, sticky="ns", pady=6)
        self.listbox.configure(yscrollcommand=sb.set)
        self.listbox.bind("<<ListboxSelect>>", lambda _e: self._update_count())
        self.count = ctk.CTkLabel(self, text="0 selezionate", anchor="w", text_color=MUTED)
        self.count.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.buttons = []
        if buttons:
            b1 = ctk.CTkButton(self, text="Tutte", height=26, fg_color="gray40",
                               hover_color="gray30", command=lambda: self.select_all(True))
            b2 = ctk.CTkButton(self, text="Nessuna", height=26, fg_color="gray40",
                               hover_color="gray30", command=lambda: self.select_all(False))
            b1.grid(row=3, column=0, sticky="ew", padx=(0, 3), pady=(2, 0))
            b2.grid(row=3, column=1, sticky="ew", padx=(3, 0), pady=(2, 0))
            self.buttons = [b1, b2]
        _LISTS.add(self)
        self.restyle(palette(is_dark()))

    def restyle(self, c: dict) -> None:
        self.listbox.configure(bg=c["bg"], fg=c["fg"], selectbackground=c["sel"],
                               selectforeground="#ffffff", disabledforeground="gray50")

    def set_columns(self, values: list[str], display: list[str] | None = None) -> None:
        previous = set(self.selected())
        state = self.listbox.cget("state")
        self.listbox.configure(state="normal")
        self.listbox.delete(0, "end")
        self._values = list(values)
        for i, v in enumerate(values):
            self.listbox.insert("end", "  " + (display[i] if display else v))
            if v in previous:
                self.listbox.selection_set(i)
        self.listbox.configure(state=state)
        self._update_count()

    def selected(self) -> list[str]:
        return [self._values[i] for i in self.listbox.curselection() if i < len(self._values)]

    def select(self, values: list[str]) -> None:
        self.listbox.selection_clear(0, "end")
        for i, v in enumerate(self._values):
            if v in values:
                self.listbox.selection_set(i)
        self._update_count()

    def select_all(self, on: bool) -> None:
        if on:
            self.listbox.selection_set(0, "end")
        else:
            self.listbox.selection_clear(0, "end")
        self._update_count()

    def set_state(self, state: str) -> None:
        self.listbox.configure(state=state)
        for b in self.buttons:
            b.configure(state=state)

    def _update_count(self) -> None:
        n = len(self.listbox.curselection())
        self.count.configure(text=f"{n} selezionat{'a' if n == 1 else 'e'} su {len(self._values)}")


class TextPanel(ctk.CTkTextbox):
    """Casella di testo di sola lettura con font monospazio."""

    def __init__(self, master, mono: bool = True, **kw):
        super().__init__(master, font=(MONO, 13) if mono else (FONT, 13), wrap="word", **kw)
        self.configure(state="disabled")

    def set_text(self, text: str) -> None:
        self.configure(state="normal")
        self.delete("1.0", "end")
        self.insert("end", text)
        self.configure(state="disabled")

    def append(self, text: str) -> None:
        self.configure(state="normal")
        self.insert("end", text)
        self.see("end")
        self.configure(state="disabled")
