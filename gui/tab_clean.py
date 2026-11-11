"""Tab 1 — Pulizia & Validazione: panoramica, pulizia, conversione tipi, anteprima."""
from __future__ import annotations

import os
import tkinter as tk

import customtkinter as ctk

import cleaner
from cleaner import DataError
from gui import widgets as W

PREVIEW_ROWS = 500


class CleanTab:
    def __init__(self, app, parent):
        self.app = app
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        self.sub = ctk.CTkTabview(parent, height=10)
        self.sub.grid(row=0, column=0, sticky="nsew")
        self._build_overview(self.sub.add("Panoramica"))
        self._build_cleaning(self.sub.add("Pulizia"))
        self._build_types(self.sub.add("Tipi di dato"))
        self._build_preview(self.sub.add("Anteprima dati"))

    # ------------------------------------------------------------ Panoramica
    def _build_overview(self, tab):
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)
        cards = ctk.CTkFrame(tab, fg_color="transparent")
        cards.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.cards = {}
        specs = [("file", "Sorgente"), ("size", "Dimensione file"), ("rows", "Righe"),
                 ("cols", "Colonne"), ("mem", "Memoria"), ("nulls", "Celle nulle"),
                 ("dups", "Righe duplicate")]
        for i, (key, caption) in enumerate(specs):
            cards.grid_columnconfigure(i, weight=3 if key == "file" else 1, uniform="cards")
            card = ctk.CTkFrame(cards)
            card.grid(row=0, column=i, sticky="nsew", padx=3)
            ctk.CTkLabel(card, text=caption, font=(W.FONT, 12), text_color=W.MUTED).pack(
                anchor="w", padx=12, pady=(8, 0))
            value = ctk.CTkLabel(card, text="—", font=(W.FONT, 18, "bold"), anchor="w")
            value.pack(anchor="w", padx=12, pady=(0, 10), fill="x")
            self.cards[key] = value
        self.summary_table = W.DataTable(tab)
        self.summary_table.grid(row=1, column=0, sticky="nsew")
        W.hint(tab, "Righe evidenziate: arancione = valori mancanti fino al 50%, rosso = oltre "
                    "il 50%.", wrap=900).grid(row=2, column=0, sticky="w", pady=(4, 0))

    # --------------------------------------------------------------- Pulizia
    def _build_cleaning(self, tab):
        app = self.app
        tab.grid_columnconfigure(0, weight=2, uniform="c")
        tab.grid_columnconfigure(1, weight=3, uniform="c")
        tab.grid_rowconfigure(0, weight=1)

        left = W.section(tab, "Gestione colonne")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.grid_rowconfigure(2, weight=1)
        b = ctk.CTkButton(left, text="Normalizza nomi colonne (snake_case)",
                          command=lambda: app.apply_op("Normalizzazione nomi…",
                                                       cleaner.clean_column_names))
        b.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12, pady=(2, 8))
        self.col_list = W.ColumnList(left, "Selezione colonne (usata anche da «Applica a»):")
        self.col_list.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=12)
        b2 = ctk.CTkButton(left, text="Elimina colonne selezionate", fg_color="#b23b3b",
                           hover_color="#8f2f2f", command=app.guard(self._drop_selected))
        b2.grid(row=3, column=0, columnspan=2, sticky="ew", padx=12, pady=(6, 12))
        app.data_widgets += [b, self.col_list, b2]

        right = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        right.grid_columnconfigure(0, weight=1)

        sec = W.section(right, "Valori mancanti (null / NaN)")
        sec.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.null_info = W.hint(sec, "—")
        self.null_info.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12)
        self._strategy_by_label = {v: k for k, v in cleaner.MISSING_STRATEGIES.items()}
        self.strategy = W.option_row(sec, 2, "Strategia:", list(cleaner.MISSING_STRATEGIES.values()),
                                     command=lambda _v: self.on_controls(self._state()))
        self.fill_value = W.entry_row(sec, 3, "Valore:", placeholder="es. 0, N/D, Sconosciuto")
        ctk.CTkLabel(sec, text="Applica a:").grid(row=4, column=0, sticky="w", padx=12, pady=4)
        self.null_scope = ctk.CTkSegmentedButton(sec, values=["Tutte le colonne", "Colonne selezionate"])
        self.null_scope.set("Tutte le colonne")
        self.null_scope.grid(row=4, column=1, sticky="w", padx=12, pady=4)
        b = ctk.CTkButton(sec, text="Applica strategia", command=app.guard(self._apply_missing))
        b.grid(row=5, column=0, columnspan=2, sticky="ew", padx=12, pady=(6, 12))
        app.data_widgets += [self.strategy, self.null_scope, b]

        sec = W.section(right, "Righe duplicate")
        sec.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.dup_info = W.hint(sec, "—")
        self.dup_info.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12)
        ctk.CTkLabel(sec, text="Confronta:").grid(row=2, column=0, sticky="w", padx=12, pady=4)
        self.dup_scope = ctk.CTkSegmentedButton(sec, values=["Tutte le colonne", "Colonne selezionate"])
        self.dup_scope.set("Tutte le colonne")
        self.dup_scope.grid(row=2, column=1, sticky="w", padx=12, pady=4)
        b = ctk.CTkButton(sec, text="Rimuovi righe duplicate", command=app.guard(self._apply_dups))
        b.grid(row=3, column=0, columnspan=2, sticky="ew", padx=12, pady=(6, 12))
        app.data_widgets += [self.dup_scope, b]

        sec = W.section(right, "Trim dei testi",
                        "Rimuove gli spazi iniziali e finali da tutte le colonne di testo.")
        sec.grid(row=2, column=0, sticky="ew")
        self.collapse_var = tk.BooleanVar(value=True)
        self.empty_null_var = tk.BooleanVar(value=False)
        c1 = ctk.CTkCheckBox(sec, text="Riduci gli spazi multipli interni a uno solo",
                             variable=self.collapse_var)
        c2 = ctk.CTkCheckBox(sec, text="Converti le celle rimaste vuote in valori nulli",
                             variable=self.empty_null_var)
        c1.grid(row=2, column=0, columnspan=2, sticky="w", padx=12, pady=(4, 4))
        c2.grid(row=3, column=0, columnspan=2, sticky="w", padx=12, pady=4)
        b = ctk.CTkButton(sec, text="Rimuovi spazi superflui", command=app.guard(self._apply_trim))
        b.grid(row=4, column=0, columnspan=2, sticky="ew", padx=12, pady=(8, 12))
        app.data_widgets += [c1, c2, b]

    def _scope(self, widget):
        if widget.get() != "Colonne selezionate":
            return None
        cols = self.col_list.selected()
        if not cols:
            raise DataError("Hai scelto «Colonne selezionate» ma nessuna colonna è selezionata "
                            "nell'elenco «Gestione colonne».")
        return self.app.cols(cols)

    def _drop_selected(self):
        cols = self.app.cols(self.col_list.selected())
        if not cols:
            raise DataError("Seleziona almeno una colonna nell'elenco.")
        if self.app.confirm(f"Eliminare {len(cols)} colonna/e?\n\n{cleaner.names(cols, 10)}"):
            self.app.apply_op("Eliminazione colonne…", lambda df: cleaner.drop_columns(df, cols))

    def _apply_missing(self):
        strategy = self._strategy_by_label[self.strategy.get()]
        cols, value = self._scope(self.null_scope), self.fill_value.get()
        self.app.apply_op("Gestione valori mancanti…",
                          lambda df: cleaner.handle_missing(df, strategy, cols, value))

    def _apply_dups(self):
        cols = self._scope(self.dup_scope)
        self.app.apply_op("Ricerca duplicati…", lambda df: cleaner.remove_duplicates(df, cols))

    def _apply_trim(self):
        collapse, empty = self.collapse_var.get(), self.empty_null_var.get()
        self.app.apply_op("Pulizia testi…", lambda df: cleaner.trim_text(df, collapse, empty))

    # ----------------------------------------------------------- Tipi di dato
    def _build_types(self, tab):
        app = self.app
        tab.grid_columnconfigure(0, weight=2, uniform="t")
        tab.grid_columnconfigure(1, weight=3, uniform="t")
        tab.grid_rowconfigure(0, weight=1)

        left = W.section(tab, "Colonne da convertire")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.grid_rowconfigure(1, weight=1)
        self.type_list = W.ColumnList(left, buttons=True)
        self.type_list.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=12, pady=(0, 12))
        self.type_list.listbox.bind("<<ListboxSelect>>", lambda _e: (
            self.type_list._update_count(), self._preview_cast(silent=True)), add=True)
        app.data_widgets.append(self.type_list)

        right = ctk.CTkFrame(tab, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)
        sec = W.section(right, "Conversione forzata del tipo (cast)")
        sec.grid(row=0, column=0, sticky="ew")
        self.cast_target = W.option_row(sec, 2, "Converti in:", list(cleaner.CAST_TARGETS),
                                        command=lambda _v: self._preview_cast(silent=True))
        self.cast_fmt = W.entry_row(sec, 3, "Formato data:",
                                    placeholder="es. %d/%m/%Y  —  vuoto = riconoscimento automatico")
        self.decimal_comma = tk.BooleanVar(value=False)
        self.dayfirst = tk.BooleanVar(value=True)
        self.strict = tk.BooleanVar(value=False)
        checks = [
            ctk.CTkCheckBox(sec, text="Numeri in formato italiano (1.234,56)",
                            variable=self.decimal_comma),
            ctk.CTkCheckBox(sec, text="Date con il giorno prima del mese (gg/mm/aaaa)",
                            variable=self.dayfirst),
            ctk.CTkCheckBox(sec, text="Modalità rigorosa: interrompi se un valore non è convertibile",
                            variable=self.strict),
        ]
        for i, c in enumerate(checks):
            c.grid(row=4 + i, column=0, columnspan=2, sticky="w", padx=12, pady=3)
        bar = ctk.CTkFrame(sec, fg_color="transparent")
        bar.grid(row=7, column=0, columnspan=2, sticky="ew", padx=12, pady=(8, 4))
        bar.grid_columnconfigure((0, 1), weight=1)
        b1 = ctk.CTkButton(bar, text="Anteprima conversione", fg_color="gray40",
                           hover_color="gray30", command=app.guard(self._preview_cast))
        b2 = ctk.CTkButton(bar, text="Converti colonne selezionate", command=app.guard(self._cast))
        b1.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        b2.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        W.hint(sec, "Codici formato: %d giorno, %m mese, %Y anno (4 cifre), %y anno (2 cifre), "
                    "%H ore, %M minuti, %S secondi, %b mese abbreviato (Jan).  "
                    "Con la modalità non rigorosa i valori non convertibili diventano null.").grid(
            row=8, column=0, columnspan=2, sticky="ew", padx=12, pady=(4, 12))
        app.data_widgets += [self.cast_target, self.cast_fmt, *checks, b1, b2]

        prev = W.section(right, "Anteprima (prime 200 righe della prima colonna selezionata)")
        prev.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        prev.grid_rowconfigure(1, weight=1)
        prev.grid_columnconfigure(0, weight=1)
        self.cast_preview = W.DataTable(prev)
        self.cast_preview.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=12, pady=(0, 6))
        self.cast_info = W.hint(prev, "")
        self.cast_info.grid(row=2, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 10))

    def _cast_args(self):
        cols = self.app.cols(self.type_list.selected())
        if not cols:
            raise DataError("Seleziona almeno una colonna da convertire.")
        return cols, dict(target=cleaner.CAST_TARGETS[self.cast_target.get()],
                          fmt=self.cast_fmt.get().strip(), strict=self.strict.get(),
                          decimal_comma=self.decimal_comma.get(), dayfirst=self.dayfirst.get())

    def _preview_cast(self, silent: bool = False):
        if self.app.df is None:
            return
        try:
            cols, kw = self._cast_args()
            col = cols[0]
            head = self.app.df[[col]].head(200)
            res = cleaner.cast_columns(head, [col], **{**kw, "strict": False})
        except DataError as exc:
            if silent:
                self.cast_info.configure(text=str(exc).splitlines()[0])
                self.cast_preview.clear()
                return
            raise
        before, after = head[col], res.df[col]
        bad = after.isna() & before.notna()
        rows = [(W.format_cell(a), W.format_cell(b), "✗ non convertibile" if x else "")
                for a, b, x in zip(before, after, bad)]
        self.cast_preview.show([f"{col} (originale: {before.dtype})", f"convertito ({after.dtype})", ""],
                               rows, tags=[("bad",) if x else () for x in bad])
        self.cast_info.configure(text=f"{int(bad.sum())} valori su {len(head)} non convertibili "
                                      "nell'anteprima." if bad.any()
                                 else "Tutti i valori dell'anteprima sono convertibili.")

    def _cast(self):
        cols, kw = self._cast_args()
        self.app.apply_op("Conversione dei tipi…", lambda df: cleaner.cast_columns(df, cols, **kw),
                          on_done=lambda _r: self._preview_cast(silent=True))

    # ---------------------------------------------------------- Anteprima
    def _build_preview(self, tab):
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)
        self.preview_info = ctk.CTkLabel(tab, text="Nessun dato da mostrare.", anchor="w")
        self.preview_info.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.preview = W.DataTable(tab)
        self.preview.grid(row=1, column=0, sticky="nsew")

    # -------------------------------------------------------- Aggiornamento
    def _state(self):
        return "normal" if self.app.df is not None and not self.app.busy else "disabled"

    def on_controls(self, state: str):
        fill_on = state == "normal" and self._strategy_by_label[self.strategy.get()] == "value"
        self.fill_value.configure(state="normal" if fill_on else "disabled")

    def on_data_changed(self):
        app, s, df = self.app, self.app.summary, self.app.df
        name = os.path.basename(app.source) if app.source else "—"
        self.cards["file"].configure(text=name if len(name) <= 34 else name[:31] + "…")
        self.cards["size"].configure(text=cleaner.fmt_bytes(app.file_size) if app.file_size else "—")
        self.cards["rows"].configure(text=cleaner.fmt_int(s["rows"]))
        self.cards["cols"].configure(text=cleaner.fmt_int(s["cols"]))
        self.cards["mem"].configure(text=cleaner.fmt_bytes(s["memory"]))
        self.cards["nulls"].configure(text=cleaner.fmt_pct(s["null_pct"]))
        self.cards["dups"].configure(text="?" if s["duplicates"] is None else cleaner.fmt_int(s["duplicates"]))

        rows, tags = [], []
        for c in s["columns"]:
            rows.append((str(c["name"]), c["label"], c["dtype"], cleaner.fmt_int(c["non_null"]),
                         cleaner.fmt_int(c["nulls"]), cleaner.fmt_pct(c["pct"]),
                         "—" if c["unique"] is None else cleaner.fmt_int(c["unique"])))
            tags.append(("bad",) if c["pct"] > 50 else ("warn",) if c["nulls"] else ())
        self.summary_table.show(["Colonna", "Tipo", "dtype pandas", "Non nulli", "Nulli", "% nulli",
                                 "Valori unici"], rows, tags=tags,
                                anchors=["w", "w", "w", "e", "e", "e", "e"],
                                widths=[280, 110, 120, 100, 90, 90, 110])

        labels = [str(c["name"]) for c in s["columns"]]
        display = [f"{c['name']}   ·  {c['label']}  ·  {cleaner.fmt_pct(c['pct'])} null"
                   for c in s["columns"]]
        self.col_list.set_columns(labels, display)
        self.type_list.set_columns(labels, [f"{c['name']}   ·  {c['label']} ({c['dtype']})"
                                            for c in s["columns"]])
        self.null_info.configure(text=f"{cleaner.fmt_int(s['total_nulls'])} valori mancanti nel "
                                      f"dataset ({cleaner.fmt_pct(s['null_pct'])} delle celle)")
        dups = s["duplicates"]
        self.dup_info.configure(text="Duplicati non calcolabili" if dups is None else
                                f"{cleaner.fmt_int(dups)} righe duplicate rilevate (tutte le colonne)")

        self.preview.show_df(df, PREVIEW_ROWS)
        shown = min(len(df), PREVIEW_ROWS)
        self.preview_info.configure(
            text=(f"Mostrate le prime {cleaner.fmt_int(shown)} righe su {cleaner.fmt_int(len(df))}"
                  if len(df) > shown else f"Mostrate tutte le {cleaner.fmt_int(shown)} righe")
            + ". I valori mancanti sono indicati come «null».")
