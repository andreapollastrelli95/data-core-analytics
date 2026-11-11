"""Tab 3 — Insights & Statistica: descrittive, correlazioni, outlier, test d'ipotesi, grafici."""
from __future__ import annotations

import customtkinter as ctk

import cleaner
import stats_engine as st
from cleaner import DataError
from gui import widgets as W

PLOT_IMPORTS = ("import matplotlib.pyplot as plt", "import seaborn as sns")


def _fmt(v):
    return W.format_cell(v)


class InsightsTab:
    def __init__(self, app, parent):
        self.app = app
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        self.sub = ctk.CTkTabview(parent, height=10)
        self.sub.grid(row=0, column=0, sticky="nsew")
        self._build_describe(self.sub.add("Descrittive"))
        self._build_corr(self.sub.add("Correlazioni"))
        self._build_outliers(self.sub.add("Outlier"))
        self._build_tests(self.sub.add("Test d'ipotesi"))
        self._build_charts(self.sub.add("Grafici"))

    # ----------------------------------------------------------- Descrittive
    def _build_describe(self, tab):
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(2, weight=3)
        tab.grid_rowconfigure(4, weight=2)
        bar = ctk.CTkFrame(tab, fg_color="transparent")
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        b = ctk.CTkButton(bar, text="Calcola statistiche descrittive", command=self.app.guard(self._describe))
        b.pack(side="left")
        self.desc_info = W.hint(bar, "Percentili, skewness, kurtosis, coefficiente di variazione e "
                                     "occupazione di memoria per colonna.", wrap=800)
        self.desc_info.pack(side="left", padx=12)
        self.app.data_widgets.append(b)
        ctk.CTkLabel(tab, text="Variabili numeriche", font=(W.FONT, 14, "bold"), anchor="w").grid(
            row=1, column=0, sticky="ew")
        self.num_table = W.DataTable(tab)
        self.num_table.grid(row=2, column=0, sticky="nsew", pady=(2, 8))
        ctk.CTkLabel(tab, text="Variabili categoriche / testuali / date", font=(W.FONT, 14, "bold"),
                     anchor="w").grid(row=3, column=0, sticky="ew")
        self.cat_table = W.DataTable(tab)
        self.cat_table.grid(row=4, column=0, sticky="nsew", pady=(2, 0))

    def _describe(self):
        if not self.app.require_data():
            return
        df = self.app.df

        def done(res):
            num, cat = res
            self._show_frame(self.num_table, num.round(4))
            self._show_frame(self.cat_table, cat.round(2))
            mem = df.memory_usage(deep=True).sum()
            self.desc_info.configure(text=f"{len(num)} colonne numeriche, {len(cat)} non numeriche · "
                                          f"memoria totale {cleaner.fmt_bytes(mem)}")
            self.app.record("Statistiche descrittive", st.DESCRIBE_CODE)

        self.app.run_task("Calcolo statistiche…",
                          lambda: (st.describe_numeric(df), st.describe_categorical(df)), done)

    @staticmethod
    def _show_frame(table, frame):
        if frame.empty:
            table.show(["Nessuna colonna"], [])
            return
        table.show_df(frame, 2000)

    # ---------------------------------------------------------- Correlazioni
    def _build_corr(self, tab):
        tab.grid_columnconfigure(0, weight=3)
        tab.grid_columnconfigure(1, weight=2)
        tab.grid_rowconfigure(1, weight=1)
        bar = ctk.CTkFrame(tab, fg_color="transparent")
        bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ctk.CTkLabel(bar, text="Metodo:").pack(side="left")
        self.corr_method = ctk.CTkSegmentedButton(bar, values=["Pearson", "Spearman", "Kendall"])
        self.corr_method.set("Pearson")
        self.corr_method.pack(side="left", padx=8)
        b = ctk.CTkButton(bar, text="Calcola matrice di correlazione", command=self.app.guard(self._corr))
        b.pack(side="left", padx=8)
        W.hint(bar, "Pearson: relazioni lineari · Spearman/Kendall: relazioni monotone, robuste "
                    "agli outlier.", wrap=600).pack(side="left", padx=8)
        self.app.data_widgets += [self.corr_method, b]
        self.corr_fig = W.FigurePanel(tab, "Premi «Calcola» per generare la heatmap.")
        self.corr_fig.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        right = ctk.CTkFrame(tab, fg_color="transparent")
        right.grid(row=1, column=1, sticky="nsew")
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(right, text="Coppie più correlate (con p-value)", font=(W.FONT, 14, "bold"),
                     anchor="w").grid(row=0, column=0, sticky="ew")
        self.corr_table = W.DataTable(right)
        self.corr_table.grid(row=1, column=0, sticky="nsew")

    def _corr(self):
        if not self.app.require_data():
            return
        method = self.corr_method.get().lower()
        df = self.app.df

        def done(res):
            corr, pairs = res
            self.corr_fig.show(lambda dark: st.correlation_figure(corr, method, dark))
            rows, tags = [], []
            for _, r in pairs.iterrows():
                rows.append((r["variabile A"], r["variabile B"], f"{r['coefficiente']:.3f}",
                             _fmt(r["p-value"]), cleaner.fmt_int(r["n"]), r["forza"]))
                tags.append(("good",) if abs(r["coefficiente"]) >= 0.6 and r["p-value"] < 0.05 else ())
            self.corr_table.show(["A", "B", "coeff.", "p-value", "n", "forza"], rows, tags=tags,
                                 anchors=["w", "w", "e", "e", "e", "w"])
            self.app.record(f"Correlazioni ({method})", st.correlation_code(method))

        self.app.run_task("Calcolo correlazioni…", lambda: st.correlation(df, method), done)

    # --------------------------------------------------------------- Outlier
    def _build_outliers(self, tab):
        app = self.app
        tab.grid_columnconfigure(0, weight=1, uniform="o")
        tab.grid_columnconfigure(1, weight=3, uniform="o")
        tab.grid_rowconfigure(0, weight=1)
        left = W.section(tab, "Rilevamento outlier")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.grid_rowconfigure(1, weight=1)
        self.out_cols = W.ColumnList(left, "Colonne numeriche (nessuna = tutte):", height=6)
        self.out_cols.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=12)
        ctk.CTkLabel(left, text="Metodo:").grid(row=2, column=0, sticky="w", padx=12, pady=4)
        self.out_method = ctk.CTkSegmentedButton(left, values=["IQR", "Z-score"],
                                                 command=self._on_out_method)
        self.out_method.set("IQR")
        self.out_method.grid(row=2, column=1, sticky="w", padx=12, pady=4)
        self.out_thr = W.entry_row(left, 3, "Soglia:", "1.5", width=80)
        self.out_help = W.hint(left, "IQR: fuori da [Q1 − k·IQR, Q3 + k·IQR] (k = 1,5 standard, 3 = estremi)",
                               wrap=280)
        self.out_help.grid(row=4, column=0, columnspan=2, sticky="ew", padx=12)
        b1 = ctk.CTkButton(left, text="Analizza", command=app.guard(self._outliers))
        b2 = ctk.CTkButton(left, text="Rimuovi righe con outlier", fg_color="#b23b3b",
                           hover_color="#8f2f2f", command=app.guard(lambda: self._outlier_op(st.remove_outliers)))
        b3 = ctk.CTkButton(left, text="Limita ai confini (winsorizza)", fg_color="gray40",
                           hover_color="gray30", command=app.guard(lambda: self._outlier_op(st.clip_outliers)))
        for i, b in enumerate((b1, b2, b3)):
            b.grid(row=5 + i, column=0, columnspan=2, sticky="ew", padx=12, pady=(6 if i == 0 else 3, 3))
        ctk.CTkFrame(left, height=8, fg_color="transparent").grid(row=8, column=0)
        app.data_widgets += [self.out_cols, self.out_method, self.out_thr, b1, b2, b3]

        right = ctk.CTkFrame(tab, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=2)
        right.grid_rowconfigure(1, weight=3)
        self.out_table = W.DataTable(right)
        self.out_table.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        self.out_fig = W.FigurePanel(right, "Il boxplot degli outlier apparirà qui.")
        self.out_fig.grid(row=1, column=0, sticky="nsew")

    def _on_out_method(self, value):
        self.out_thr.delete(0, "end")
        self.out_thr.insert(0, "1.5" if value == "IQR" else "3")
        self.out_help.configure(text="IQR: fuori da [Q1 − k·IQR, Q3 + k·IQR] (k = 1,5 standard, 3 = estremi)"
                                if value == "IQR" else
                                "Z-score: |valore − media| / dev.std oltre la soglia (3 = standard)")

    def _out_args(self):
        cols = self.app.cols(self.out_cols.selected()) or None
        method = "iqr" if self.out_method.get() == "IQR" else "zscore"
        thr = W.read_number(self.out_thr, "Soglia", float, minimum=0.1, maximum=20)
        return cols, method, thr

    def _outliers(self):
        if not self.app.require_data():
            return
        cols, method, thr = self._out_args()
        df = self.app.df

        def done(res):
            summary, mask = res
            rows, tags = [], []
            for _, r in summary.iterrows():
                rows.append((r["colonna"], cleaner.fmt_int(r["outlier"]), cleaner.fmt_pct(r["% righe"], 2),
                             _fmt(r["limite inferiore"]), _fmt(r["limite superiore"]),
                             _fmt(r["min osservato"]), _fmt(r["max osservato"])))
                tags.append(("bad",) if r["% righe"] > 5 else ("warn",) if r["outlier"] else ())
            self.out_table.show(["Colonna", "Outlier", "% righe", "Limite inf.", "Limite sup.",
                                 "Min", "Max"], rows, tags=tags,
                                anchors=["w", "e", "e", "e", "e", "e", "e"])
            self.out_fig.show(lambda dark: st.outlier_figure(df, cols, method, thr, dark))
            self.app.status.configure(text=f"{cleaner.fmt_int(int(mask.sum()))} righe contengono "
                                           "almeno un outlier.")

        self.app.run_task("Ricerca outlier…", lambda: st.detect_outliers(df, cols, method, thr), done)

    def _outlier_op(self, fn):
        cols, method, thr = self._out_args()
        self.app.apply_op("Trattamento outlier…", lambda df: fn(df, cols, method, thr),
                          on_done=lambda _r: self._outliers())

    # ------------------------------------------------------------ Test d'ipotesi
    def _build_tests(self, tab):
        app = self.app
        tab.grid_columnconfigure(0, weight=1, uniform="h")
        tab.grid_columnconfigure(1, weight=2, uniform="h")
        tab.grid_rowconfigure(0, weight=1)
        left = W.section(tab, "Configurazione del test")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.test_kind = W.option_row(left, 2, "Test:", list(st.TESTS), command=self._on_test)
        self.test_v1 = W.option_row(left, 3, "Variabile 1:", [])
        self.test_v2 = W.option_row(left, 4, "Variabile 2:", [])
        self.test_alpha = W.option_row(left, 5, "Livello α:", ["0.05", "0.01", "0.10"])
        b = ctk.CTkButton(left, text="Esegui test", command=app.guard(self._run_test))
        b.grid(row=6, column=0, columnspan=2, sticky="ew", padx=12, pady=(10, 6))
        W.hint(left, "H0 = ipotesi nulla. Se p-value < α si rifiuta H0 (risultato statisticamente "
                     "significativo). Ogni test parametrico è affiancato dalla sua versione non "
                     "parametrica.", wrap=300).grid(row=7, column=0, columnspan=2, sticky="ew",
                                                    padx=12, pady=(0, 12))
        app.data_widgets += [self.test_kind, self.test_v1, self.test_v2, self.test_alpha, b]
        self.test_out = W.TextPanel(tab)
        self.test_out.grid(row=0, column=1, sticky="nsew")
        self.test_out.set_text("I risultati del test appariranno qui.")
        self._on_test(self.test_kind.get())

    def _on_test(self, label):
        test = st.TESTS[label]
        l1, l2 = st.TEST_NEEDS[test]
        self.test_v1.label.configure(text=l1 + ":")
        self.test_v2.label.configure(text=(l2 or "—") + ":")
        self._fill_test_options()

    def _fill_test_options(self):
        test = st.TESTS[self.test_kind.get()]
        num, allc = self.app.column_labels("numeric"), self.app.column_labels()
        first = allc if test == "chi2" else num
        second = allc if test in ("ttest", "anova", "chi2") else num
        W.set_options(self.test_v1, first)
        W.set_options(self.test_v2, second, allow_none=True)
        if st.TEST_NEEDS[test][1] is None:
            self.test_v2.set(W.NONE)
            self.test_v2.configure(state="disabled")
        elif self.app.df is not None and not self.app.busy:
            self.test_v2.configure(state="normal")

    def _run_test(self):
        if not self.app.require_data():
            return
        test = st.TESTS[self.test_kind.get()]
        v1, v2 = self.app.col(W.get_option(self.test_v1)), self.app.col(W.get_option(self.test_v2))
        alpha = float(self.test_alpha.get())
        df = self.app.df

        def done(text):
            self.test_out.set_text(text)
            self.app.record(f"Test: {self.test_kind.get()}", st.test_code(test, v1, v2, alpha),
                            ("from scipy import stats",))

        self.app.run_task("Esecuzione del test…", lambda: st.run_test(df, test, v1, v2, alpha), done)

    # --------------------------------------------------------------- Grafici
    def _build_charts(self, tab):
        app = self.app
        tab.grid_columnconfigure(0, weight=1, uniform="p")
        tab.grid_columnconfigure(1, weight=3, uniform="p")
        tab.grid_rowconfigure(0, weight=1)
        left = W.section(tab, "Grafico")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.chart_kind = W.option_row(left, 2, "Tipo:", list(st.CHARTS), command=self._on_chart, width=170)
        self.chart_x = W.option_row(left, 3, "X:", [], width=170)
        self.chart_y = W.option_row(left, 4, "Y:", [], width=170)
        self.chart_hue = W.option_row(left, 5, "Colore per:", [], width=170)
        self.chart_bins = W.entry_row(left, 6, "Classi (bins):", "30", width=170)
        b = ctk.CTkButton(left, text="Genera grafico", command=app.guard(self._chart))
        b.grid(row=7, column=0, columnspan=2, sticky="ew", padx=12, pady=(10, 6))
        W.hint(left, "Usa la barra sotto al grafico per zoom, spostamento e salvataggio in PNG/SVG/PDF.",
               wrap=280).grid(row=8, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 12))
        app.data_widgets += [self.chart_kind, self.chart_x, b]
        self.chart_fig = W.FigurePanel(tab, "Scegli il tipo di grafico e premi «Genera grafico».")
        self.chart_fig.grid(row=0, column=1, sticky="nsew")
        self._on_chart(self.chart_kind.get())

    def _on_chart(self, label):
        kind = st.CHARTS[label]
        need_x, need_y, hue_ok = st.CHART_NEEDS[kind]
        self.chart_x.label.configure(text=(need_x.split(":")[0] if need_x else "X") + ":")
        self.chart_y.label.configure(text=(need_y.split(":")[0] if need_y else "Y") + ":")
        self._fill_chart_options()

    def _fill_chart_options(self):
        kind = st.CHARTS[self.chart_kind.get()]
        need_x, need_y, hue_ok = st.CHART_NEEDS[kind]
        num, cat, allc = (self.app.column_labels("numeric"), self.app.column_labels("categorical"),
                          self.app.column_labels())
        W.set_options(self.chart_x, (cat or allc) if kind == "count" else num)
        W.set_options(self.chart_y, allc if kind == "box" else num, allow_none=kind == "box")
        W.set_options(self.chart_hue, allc, allow_none=True)
        enabled = self.app.df is not None and not self.app.busy
        for w, on in ((self.chart_x, need_x), (self.chart_y, need_y), (self.chart_hue, hue_ok),
                      (self.chart_bins, kind == "hist")):
            w.configure(state="normal" if on and enabled else "disabled")

    def _chart(self):
        if not self.app.require_data():
            return
        kind = st.CHARTS[self.chart_kind.get()]
        need_x, need_y, hue_ok = st.CHART_NEEDS[kind]
        x = self.app.col(W.get_option(self.chart_x)) if need_x else None
        y = self.app.col(W.get_option(self.chart_y)) if need_y else None
        hue = self.app.col(W.get_option(self.chart_hue)) if hue_ok else None
        if need_x and x is None:
            raise DataError("Seleziona la variabile X.")
        if kind == "scatter" and y is None:
            raise DataError("Seleziona la variabile Y.")
        bins = W.read_number(self.chart_bins, "Classi", int, minimum=2, maximum=500, default=30)
        df = self.app.df
        self.chart_fig.show(lambda dark: st.chart_figure(df, kind, x, y, hue, bins, dark))
        self.app.record(f"Grafico: {self.chart_kind.get()}", st.chart_code(kind, x, y, hue, bins),
                        PLOT_IMPORTS)

    # -------------------------------------------------------- Aggiornamento
    def on_controls(self, state: str):
        self._fill_chart_options()
        self._fill_test_options()

    def on_data_changed(self):
        self.out_cols.set_columns(self.app.column_labels("numeric"))
        self._fill_test_options()
        self._fill_chart_options()
