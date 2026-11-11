"""Tab 5 — ML Studio: modelli supervisionati no-code con scikit-learn."""
from __future__ import annotations

import customtkinter as ctk

import ml_engine as ml
from cleaner import DataError
from gui import widgets as W

TASK_CHOICES = ["Automatico", "Regressione", "Classificazione"]


class MLTab:
    def __init__(self, app, parent):
        self.app = app
        parent.grid_columnconfigure(0, weight=1, uniform="ml")
        parent.grid_columnconfigure(1, weight=2, uniform="ml")
        parent.grid_rowconfigure(0, weight=1)

        left = W.section(parent, "Configurazione del modello")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.grid_rowconfigure(5, weight=1)
        self.target = W.option_row(left, 2, "Target (y):", [], command=lambda _v: self._on_target(),
                                   width=170)
        ctk.CTkLabel(left, text="Tipo di problema:").grid(row=3, column=0, sticky="w", padx=12, pady=4)
        self.task = ctk.CTkSegmentedButton(left, values=TASK_CHOICES, command=lambda _v: self._on_target())
        self.task.set("Automatico")
        self.task.grid(row=3, column=1, sticky="w", padx=12, pady=4)
        self.algorithm = W.option_row(left, 4, "Algoritmo:", [], width=170)
        self.features = W.ColumnList(left, "Feature (X):", height=6)
        self.features.grid(row=5, column=0, columnspan=2, sticky="nsew", padx=12, pady=(4, 0))

        params = ctk.CTkFrame(left, fg_color="transparent")
        params.grid(row=6, column=0, columnspan=2, sticky="ew")
        params.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(params, text="Quota di test:").grid(row=0, column=0, sticky="w", padx=12, pady=4)
        tf = ctk.CTkFrame(params, fg_color="transparent")
        tf.grid(row=0, column=1, sticky="ew", padx=12)
        tf.grid_columnconfigure(0, weight=1)
        self.test_size = ctk.CTkSlider(tf, from_=10, to=50, number_of_steps=8,
                                       command=lambda v: self.test_lbl.configure(text=f"{int(v)}%"))
        self.test_size.set(20)
        self.test_size.grid(row=0, column=0, sticky="ew")
        self.test_lbl = ctk.CTkLabel(tf, text="20%", width=40)
        self.test_lbl.grid(row=0, column=1)
        self.n_estimators = W.entry_row(params, 1, "Alberi (Random Forest):", "200", width=100)
        self.max_depth = W.entry_row(params, 2, "Profondità max (vuoto = libera):", width=100)
        self.seed = W.entry_row(params, 3, "Random state:", "42", width=100)

        bar = ctk.CTkFrame(left, fg_color="transparent")
        bar.grid(row=7, column=0, columnspan=2, sticky="ew", padx=12, pady=(8, 12))
        bar.grid_columnconfigure((0, 1), weight=1)
        b_train = ctk.CTkButton(bar, text="Addestra modello", height=36, font=(W.FONT, 14, "bold"),
                                command=app.guard(self._train))
        b_train.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        self.btn_pred = ctk.CTkButton(bar, text="Aggiungi predizioni al dataset", fg_color="gray40",
                                      hover_color="gray30", command=app.guard(self._predict))
        self.btn_save = ctk.CTkButton(bar, text="Salva modello…", fg_color="gray40",
                                      hover_color="gray30", command=app.guard(self._save_model))
        self.btn_pred.grid(row=1, column=0, sticky="ew", padx=(0, 3))
        self.btn_save.grid(row=1, column=1, sticky="ew", padx=(3, 0))
        app.data_widgets += [self.target, self.task, self.algorithm, self.features, self.test_size,
                             self.n_estimators, self.max_depth, self.seed, b_train]

        right = ctk.CTkFrame(parent, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=2)
        right.grid_rowconfigure(1, weight=3)
        self.metrics = W.TextPanel(right)
        self.metrics.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        self.metrics.set_text("Scegli il target e le feature, poi premi «Addestra modello».\n\n"
                              "Pre-elaborazione automatica: imputazione dei valori mancanti "
                              "(mediana / moda), one-hot encoding delle variabili categoriche, "
                              "standardizzazione per i modelli lineari, conversione delle date "
                              "in numeri.")
        self.plots = ctk.CTkTabview(right, height=10)
        self.plots.grid(row=1, column=0, sticky="nsew")
        self.fig_imp = W.FigurePanel(self.plots.add("Feature importance"))
        self.fig_imp.pack(fill="both", expand=True)
        self.fig_eval = W.FigurePanel(self.plots.add("Valutazione"))
        self.fig_eval.pack(fill="both", expand=True)

    # ------------------------------------------------------------- Helpers
    def _task(self) -> str:
        choice = self.task.get()
        if choice == "Regressione":
            return "regression"
        if choice == "Classificazione":
            return "classification"
        target = W.get_option(self.target)
        if target is None or self.app.df is None:
            return "regression"
        return ml.suggest_task(self.app.df[self.app.col(target)])

    def _on_target(self):
        task = self._task()
        W.set_options(self.algorithm, list(ml.ALGORITHMS[task]))
        target = W.get_option(self.target)
        labels = [c for c in self.app.column_labels() if c != target]
        selected = [c for c in self.features.selected() if c != target]
        self.features.set_columns(labels)
        self.features.select(selected or labels)
        if self.task.get() == "Automatico":
            self.algorithm.label.configure(text=f"Algoritmo ({ml.TASK_LABELS[task].lower()}):")
        else:
            self.algorithm.label.configure(text="Algoritmo:")

    # ------------------------------------------------------------- Azioni
    def _train(self):
        if not self.app.require_data():
            return
        target = self.app.col(W.get_option(self.target))
        if target is None:
            raise DataError("Seleziona la variabile target.")
        features = self.app.cols(self.features.selected())
        task = self._task()
        algorithm = ml.ALGORITHMS[task].get(self.algorithm.get())
        if algorithm is None:
            raise DataError("Seleziona un algoritmo.")
        params = {"n_estimators": W.read_number(self.n_estimators, "Alberi", int, 10, 2000),
                  "max_depth": W.read_number(self.max_depth, "Profondità max", int, 1, 100, default=0) or None}
        seed = W.read_number(self.seed, "Random state", int, 0, 2**31 - 1)
        test_size = round(self.test_size.get()) / 100
        df = self.app.df

        def done(res):
            self.app.last_ml = res
            self.metrics.set_text(ml.metrics_text(res))
            self.fig_imp.show(lambda dark: ml.importance_figure(res, dark))
            self.fig_eval.show(lambda dark: ml.evaluation_figure(res, dark))
            main = next(iter(res.metrics.items()))
            self.app.log(f"Addestrato {res.algorithm_label} su '{res.target}': "
                         f"{main[0]} = {main[1]:.4f}")
            self.app.record(f"Modello {res.algorithm_label} (target: {res.target})",
                            ml.training_code(res), ml.ML_IMPORTS)
            self.on_controls("normal")

        self.app.run_task("Addestramento del modello…",
                          lambda: ml.train(df, target, features, task, algorithm, test_size, seed, params),
                          done)

    def _predict(self):
        res = self.app.last_ml
        if res is None:
            raise DataError("Addestra prima un modello.")
        self.app.apply_op("Calcolo predizioni…", lambda df: ml.add_predictions(df, res))

    def _save_model(self):
        res = self.app.last_ml
        if res is None:
            raise DataError("Addestra prima un modello.")
        path = self.app.ask_save("Salva modello", f"modello_{res.target}", {"Modello joblib": ".joblib"})
        if not path:
            return
        self.app.run_task("Salvataggio del modello…", lambda: ml.save_model(res, path),
                          lambda _r: self.app.info(f"Modello salvato in:\n{path}\n\n"
                                                   "Caricalo con joblib.load(percorso)['pipeline']."))

    # -------------------------------------------------------- Aggiornamento
    def on_controls(self, state: str):
        has = self.app.last_ml is not None
        self.btn_pred.configure(state=state if has else "disabled")
        self.btn_save.configure(state=state if has else "disabled")

    def on_data_changed(self):
        labels = self.app.column_labels()
        W.set_options(self.target, labels)
        self._on_target()
