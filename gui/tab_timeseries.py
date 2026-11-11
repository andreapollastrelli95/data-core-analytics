"""Tab 4 — Time Series & Forecasting: aggregazione temporale, scomposizione, previsione."""
from __future__ import annotations

import customtkinter as ctk

import cleaner
import ts_engine as ts
from cleaner import DataError
from gui import widgets as W

TS_IMPORTS = ("import matplotlib.pyplot as plt",
              "from statsmodels.tsa.seasonal import seasonal_decompose",
              "from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing")


class TimeSeriesTab:
    def __init__(self, app, parent):
        self.app = app
        self.forecast_df = None
        self._candidates: list = []
        parent.grid_columnconfigure(0, weight=1, uniform="ts")
        parent.grid_columnconfigure(1, weight=3, uniform="ts")
        parent.grid_rowconfigure(0, weight=1)

        left = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.grid_columnconfigure(0, weight=1)

        sec = W.section(left, "1. Serie temporale")
        sec.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.date_col = W.option_row(sec, 2, "Colonna data:", [], width=150)
        self.value_col = W.option_row(sec, 3, "Valore:", [], width=150)
        self.freq = W.option_row(sec, 4, "Intervallo:", list(ts.FREQS), width=150,
                                 command=lambda _v: self._default_season())
        self.freq.set("Mensile")
        self.agg = W.option_row(sec, 5, "Aggregazione:", list(ts.AGGS), width=150)
        self.roll = W.entry_row(sec, 6, "Media mobile:", "3", width=150)
        self.date_info = W.hint(sec, "", wrap=260)
        self.date_info.grid(row=7, column=0, columnspan=2, sticky="ew", padx=12)
        b1 = ctk.CTkButton(sec, text="Visualizza serie", command=app.guard(self._show_series))
        b2 = ctk.CTkButton(sec, text="Applica aggregazione al dataset", fg_color="gray40",
                           hover_color="gray30", command=app.guard(self._apply_resample))
        b1.grid(row=8, column=0, columnspan=2, sticky="ew", padx=12, pady=(6, 3))
        b2.grid(row=9, column=0, columnspan=2, sticky="ew", padx=12, pady=(3, 12))

        sec2 = W.section(left, "2. Trend e stagionalità")
        sec2.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        ctk.CTkLabel(sec2, text="Modello:").grid(row=2, column=0, sticky="w", padx=12, pady=4)
        self.dec_model = ctk.CTkSegmentedButton(sec2, values=["Additivo", "Moltiplicativo"])
        self.dec_model.set("Additivo")
        self.dec_model.grid(row=2, column=1, sticky="w", padx=12, pady=4)
        self.period = W.entry_row(sec2, 3, "Periodo stagionale:", "12", width=150)
        b3 = ctk.CTkButton(sec2, text="Scomponi la serie", command=app.guard(self._decompose))
        b3.grid(row=4, column=0, columnspan=2, sticky="ew", padx=12, pady=(6, 12))

        sec3 = W.section(left, "3. Previsione")
        sec3.grid(row=2, column=0, sticky="ew")
        self.method = W.option_row(sec3, 2, "Metodo:", list(ts.METHODS), width=150)
        self.method.set("Holt-Winters (trend + stagionalità)")
        self.horizon = W.entry_row(sec3, 3, "Periodi futuri:", "12", width=150)
        self.window = W.entry_row(sec3, 4, "Finestra media mob.:", "3", width=150)
        b4 = ctk.CTkButton(sec3, text="Calcola previsione", command=app.guard(self._forecast))
        self.btn_export = ctk.CTkButton(sec3, text="Esporta previsioni…", fg_color="gray40",
                                        hover_color="gray30", command=app.guard(self._export))
        b4.grid(row=5, column=0, columnspan=2, sticky="ew", padx=12, pady=(6, 3))
        self.btn_export.grid(row=6, column=0, columnspan=2, sticky="ew", padx=12, pady=(3, 6))
        W.hint(sec3, "La qualità è stimata su un periodo di validazione finale (holdout): il modello "
                     "viene addestrato sul resto della serie e confrontato con i valori reali.",
               wrap=260).grid(row=7, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 12))
        app.data_widgets += [self.date_col, self.value_col, self.freq, self.agg, self.roll, b1, b2,
                             self.dec_model, self.period, b3, self.method, self.horizon,
                             self.window, b4]

        right = ctk.CTkFrame(parent, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=3)
        right.grid_rowconfigure(1, weight=2)
        self.fig = W.FigurePanel(right, "Seleziona colonna data e valore, poi «Visualizza serie».")
        self.fig.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        bottom = ctk.CTkFrame(right, fg_color="transparent")
        bottom.grid(row=1, column=0, sticky="nsew")
        bottom.grid_columnconfigure(0, weight=2)
        bottom.grid_columnconfigure(1, weight=1)
        bottom.grid_rowconfigure(0, weight=1)
        self.table = W.DataTable(bottom)
        self.table.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.info = W.TextPanel(bottom)
        self.info.grid(row=0, column=1, sticky="nsew")

    # ------------------------------------------------------------- Helpers
    def _default_season(self):
        freq = ts.FREQS[self.freq.get()]
        self.period.delete(0, "end")
        self.period.insert(0, str(max(2, ts.SEASON[freq])))

    def _series_args(self):
        date = self.app.col(W.get_option(self.date_col))
        if date is None:
            raise DataError("Nessuna colonna temporale disponibile.\n\nConverti una colonna in "
                            "Data/ora dalla scheda «Pulizia & Validazione › Tipi di dato».")
        value_label = self.value_col.get()
        value = ts.COUNT_ROWS if value_label == ts.COUNT_ROWS else self.app.col(value_label)
        return date, value, ts.FREQS[self.freq.get()], ts.AGGS[self.agg.get()]

    def _record_series(self, date, value, freq, agg, extra_desc, extra_code=""):
        code = ts.series_code(date, value, freq, agg) + (f"\n{extra_code}" if extra_code else "")
        self.app.record(f"Serie temporale: {extra_desc}", code, TS_IMPORTS)

    def _info_text(self, info: dict) -> str:
        lines = [f"Periodi: {cleaner.fmt_int(info['periods'])}",
                 f"Dal {info['start']:%d/%m/%Y} al {info['end']:%d/%m/%Y}"]
        if info["gaps"]:
            lines.append(f"Periodi senza dati: {info['gaps']}")
        if info["dropped_dates"]:
            lines.append(f"Righe con data non valida ignorate: {cleaner.fmt_int(info['dropped_dates'])}")
        return "\n".join(lines)

    # ----------------------------------------------------------- Azioni
    def _show_series(self):
        if not self.app.require_data():
            return
        args = self._series_args()
        roll = W.read_number(self.roll, "Media mobile", int, minimum=0, maximum=1000, default=0)
        df = self.app.df

        def done(res):
            series, info = res
            self.fig.show(lambda dark: ts.series_figure(series, dark, roll))
            frame = series.rename("valore").rename_axis("periodo").reset_index()
            self.table.show_df(frame, 2000)
            self.info.set_text(self._info_text(info))
            self._record_series(*args, "visualizzazione",
                                "serie.plot(figsize=(9, 4))\nplt.show()")

        self.app.run_task("Aggregazione della serie…", lambda: ts.build_series(df, *args), done)

    def _apply_resample(self):
        date, value, freq, agg = self._series_args()
        if value == ts.COUNT_ROWS:
            raise DataError("Per applicare l'aggregazione al dataset scegli una colonna valore numerica.")
        cols = [value]
        self.app.apply_op("Aggregazione temporale…",
                          lambda df: ts.resample_dataset(df, date, cols, freq, agg))

    def _decompose(self):
        if not self.app.require_data():
            return
        args = self._series_args()
        period = W.read_number(self.period, "Periodo stagionale", int, minimum=2, maximum=1000)
        model = "additive" if self.dec_model.get() == "Additivo" else "multiplicative"
        df = self.app.df

        def task():
            series, info = ts.build_series(df, *args)
            return ts.decompose(series, model, period), info

        def done(res):
            result, info = res
            self.fig.show(lambda dark: ts.decomposition_figure(result, dark))
            frame = result.observed.rename("osservato").to_frame()
            frame["trend"] = result.trend
            frame["stagionalità"] = result.seasonal
            frame["residuo"] = result.resid
            self.table.show_df(frame.rename_axis("periodo").reset_index(), 2000)
            trend, season = ts.seasonal_strength(result)
            self.info.set_text(self._info_text(info) + "\n\n"
                               f"Forza del trend:        {trend:.2f}\n"
                               f"Forza stagionalità:     {season:.2f}\n"
                               "(0 = assente, 1 = dominante)")
            self._record_series(*args, f"scomposizione {model}", ts.decompose_code(model, period))

        self.app.run_task("Scomposizione della serie…", task, done)

    def _forecast(self):
        if not self.app.require_data():
            return
        args = self._series_args()
        method_label = self.method.get()
        method = ts.METHODS[method_label]
        horizon = W.read_number(self.horizon, "Periodi futuri", int, minimum=1, maximum=1000)
        window = W.read_number(self.window, "Finestra media mobile", int, minimum=1, maximum=1000)
        season = W.read_number(self.period, "Periodo stagionale", int, minimum=2, maximum=1000)
        df = self.app.df

        def task():
            series, info = ts.build_series(df, *args)
            return ts.forecast(series, method, horizon, window, season), info

        def done(res):
            result, info = res
            self.forecast_df = result["forecast"]
            self.fig.show(lambda dark: ts.forecast_figure(result, method_label, dark))
            self.table.show_df(self.forecast_df, 2000)
            m = result["metrics"]
            text = self._info_text(info) + f"\n\nModello: {method_label}\n"
            if m:
                text += (f"\nValidazione sugli ultimi {m['periodi di validazione']} periodi:\n"
                         f"  MAE   {m['MAE']:.4g}\n  RMSE  {m['RMSE']:.4g}\n"
                         f"  MAPE  {m['MAPE %']:.2f}%")
            else:
                text += "\nSerie troppo corta per la validazione con questo modello."
            self.info.set_text(text)
            self.btn_export.configure(state="normal")
            self._record_series(*args, f"previsione {method_label}",
                                ts.forecast_code(method, horizon, window, season))

        self.app.run_task("Stima del modello di previsione…", task, done)

    def _export(self):
        if self.forecast_df is None:
            raise DataError("Calcola prima una previsione.")
        path = self.app.ask_save("Esporta previsioni", "previsioni", {"CSV": ".csv", "Excel": ".xlsx"})
        if not path:
            return
        frame = self.forecast_df
        self.app.run_task("Esportazione previsioni…", lambda: cleaner.export(frame, path),
                          lambda _n: self.app.info(f"Previsioni salvate in:\n{path}"))

    # -------------------------------------------------------- Aggiornamento
    def on_controls(self, state: str):
        self.btn_export.configure(state=state if self.forecast_df is not None else "disabled")

    def on_data_changed(self):
        df = self.app.df
        self._candidates = [str(c) for c in ts.datetime_candidates(df)]
        W.set_options(self.date_col, self._candidates)
        numeric = self.app.column_labels("numeric")
        first_time = self.value_col.get() == W.NONE
        W.set_options(self.value_col, [ts.COUNT_ROWS, *numeric])
        if first_time and numeric:
            self.value_col.set(numeric[0])
        self.date_info.configure(
            text=(f"Colonne temporali rilevate: {len(self._candidates)}" if self._candidates else
                  "Nessuna colonna data rilevata: convertine una in «Tipi di dato»."))
