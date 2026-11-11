"""Tab 2 — Trasformazione & SQL: merge/concat, group by, editor DuckDB, import da database."""
from __future__ import annotations

import os
import tkinter as tk

import customtkinter as ctk

import cleaner
import sql_engine as sqe
import transform_engine as te
from cleaner import DataError
from gui import widgets as W

SECONDARY_NAME = "df2"


class TransformTab:
    def __init__(self, app, parent):
        self.app = app
        self.sql_result = None
        self.sql_query = ""
        self.gb_preview = None
        self.key_pairs: list[tuple] = []
        self.metrics: list[tuple] = []
        self.sqlite_path = ""
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        self.sub = ctk.CTkTabview(parent, height=10)
        self.sub.grid(row=0, column=0, sticky="nsew")
        self._build_merge(self.sub.add("Merge & Concat"))
        self._build_groupby(self.sub.add("Group By"))
        self._build_sql(self.sub.add("Editor SQL"))
        self._build_db(self.sub.add("Importa da database"))

    # --------------------------------------------------------- Merge/Concat
    def _build_merge(self, tab):
        app = self.app
        tab.grid_columnconfigure(0, weight=2, uniform="m")
        tab.grid_columnconfigure(1, weight=3, uniform="m")
        tab.grid_rowconfigure(0, weight=1)
        left = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.grid_columnconfigure(0, weight=1)

        sec = W.section(left, "1. Dataset secondario",
                        "Carica il secondo dataset da unire al principale (file o database).")
        sec.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ctk.CTkButton(sec, text="Carica file secondario…", command=app.guard(self._load_secondary)
                      ).grid(row=2, column=0, columnspan=2, sticky="ew", padx=12, pady=4)
        self.sec_info = W.hint(sec, "Nessun dataset secondario caricato.")
        self.sec_info.grid(row=3, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 10))

        sec = W.section(left, "2. Join relazionale (merge)")
        sec.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.join_type = W.option_row(sec, 2, "Tipo di join:", list(te.JOIN_TYPES))
        self.left_key = W.option_row(sec, 3, "Chiave principale:", [])
        self.right_key = W.option_row(sec, 4, "Chiave secondaria:", [])
        bar = ctk.CTkFrame(sec, fg_color="transparent")
        bar.grid(row=5, column=0, columnspan=2, sticky="ew", padx=12, pady=4)
        bar.grid_columnconfigure((0, 1), weight=1)
        b_add = ctk.CTkButton(bar, text="Aggiungi coppia di chiavi", command=app.guard(self._add_pair))
        b_clear = ctk.CTkButton(bar, text="Svuota", fg_color="gray40", hover_color="gray30",
                                command=self._clear_pairs)
        b_add.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        b_clear.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.pairs_info = W.hint(sec, "Chiavi: nessuna (se vuoto si usa la coppia selezionata sopra)")
        self.pairs_info.grid(row=6, column=0, columnspan=2, sticky="ew", padx=12)
        b_join = ctk.CTkButton(sec, text="Esegui join", command=app.guard(self._merge))
        b_join.grid(row=7, column=0, columnspan=2, sticky="ew", padx=12, pady=(6, 12))

        sec = W.section(left, "3. Accodamento verticale (concat)",
                        "Aggiunge le righe del secondario sotto al principale; le colonne vengono "
                        "allineate per nome.")
        sec.grid(row=2, column=0, sticky="ew")
        self.add_source = tk.BooleanVar(value=False)
        c = ctk.CTkCheckBox(sec, text="Aggiungi colonna «origine»", variable=self.add_source)
        c.grid(row=2, column=0, columnspan=2, sticky="w", padx=12, pady=4)
        b_cat = ctk.CTkButton(sec, text="Accoda righe", command=app.guard(self._concat))
        b_cat.grid(row=3, column=0, columnspan=2, sticky="ew", padx=12, pady=(6, 12))
        self.merge_widgets = [self.join_type, self.left_key, self.right_key, b_add, b_clear,
                              b_join, c, b_cat]

        right = W.section(tab, "Anteprima del dataset secondario")
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        right.grid_rowconfigure(1, weight=1)
        self.sec_table = W.DataTable(right)
        self.sec_table.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=12, pady=(0, 12))

    def _load_secondary(self):
        path = self.app.ask_open("Seleziona il dataset secondario")
        if not path:
            return

        def task():
            sheets = cleaner.excel_sheets(path)
            return sheets

        def got_sheets(sheets):
            sheet = self.app.choose_sheet(sheets)
            if sheet is False:
                return
            self.app.run_task("Caricamento del dataset secondario…",
                              lambda: cleaner.load_file(path, sheet, var="df_dx"),
                              lambda res: self.set_secondary(res[0], os.path.basename(path), res[1]))

        self.app.run_task("Lettura del file…", task, got_sheets)

    def set_secondary(self, df, name: str, code: str):
        self.app.secondary = {"df": df, "name": name, "code": code}
        self.app.log(f"Caricato dataset secondario '{name}': {cleaner.fmt_int(len(df))} righe × "
                     f"{df.shape[1]} colonne")
        self.key_pairs.clear()
        self._refresh_secondary()

    def _refresh_secondary(self):
        sec = self.app.secondary
        if not sec:
            return
        df = sec["df"]
        self.sec_info.configure(text=f"{sec['name']}: {cleaner.fmt_int(len(df))} righe × "
                                     f"{df.shape[1]} colonne  (in SQL: tabella «{SECONDARY_NAME}»)")
        W.set_options(self.right_key, [str(c) for c in df.columns])
        self.sec_table.show_df(df, 300)
        self._update_pairs()
        self.on_controls("normal" if self.app.df is not None and not self.app.busy else "disabled")

    def _secondary_col(self, label):
        for c in self.app.secondary["df"].columns:
            if str(c) == label:
                return c
        raise DataError(f"Colonna '{label}' non trovata nel dataset secondario.")

    def _need_secondary(self):
        if not self.app.secondary:
            raise DataError("Carica prima un dataset secondario.")
        return self.app.secondary

    def _add_pair(self):
        self._need_secondary()
        pair = (self.left_key.get(), self.right_key.get())
        if W.NONE in pair:
            raise DataError("Seleziona entrambe le chiavi.")
        if pair not in self.key_pairs:
            self.key_pairs.append(pair)
        self._update_pairs()

    def _clear_pairs(self):
        self.key_pairs.clear()
        self._update_pairs()

    def _update_pairs(self):
        text = ", ".join(f"{a} = {b}" for a, b in self.key_pairs) or \
            "nessuna (verrà usata la coppia selezionata sopra)"
        self.pairs_info.configure(text=f"Chiavi: {text}")

    def _merge(self):
        sec = self._need_secondary()
        pairs = self.key_pairs or [(self.left_key.get(), self.right_key.get())]
        left_on = self.app.cols([a for a, _ in pairs])
        right_on = [self._secondary_col(b) for _, b in pairs]
        how = te.JOIN_TYPES[self.join_type.get()]
        right, code = sec["df"], sec["code"]
        self.app.apply_op("Join in corso…",
                          lambda df: te.merge(df, right, how, left_on, right_on, code))

    def _concat(self):
        sec = self._need_secondary()
        right, code, src = sec["df"], sec["code"], self.add_source.get()
        self.app.apply_op("Accodamento…", lambda df: te.concat(df, right, src, code))

    # -------------------------------------------------------------- Group By
    def _build_groupby(self, tab):
        app = self.app
        tab.grid_columnconfigure(0, weight=2, uniform="g")
        tab.grid_columnconfigure(1, weight=3, uniform="g")
        tab.grid_rowconfigure(0, weight=1)
        left = W.section(tab, "Raggruppamento")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.grid_rowconfigure(1, weight=1)
        self.gb_keys = W.ColumnList(left, "Colonne di raggruppamento:", height=6)
        self.gb_keys.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=12)
        self.gb_col = W.option_row(left, 2, "Metrica su:", [])
        self.gb_func = W.option_row(left, 3, "Funzione:", list(te.AGG_FUNCS))
        bar = ctk.CTkFrame(left, fg_color="transparent")
        bar.grid(row=4, column=0, columnspan=2, sticky="ew", padx=12, pady=4)
        bar.grid_columnconfigure((0, 1), weight=1)
        b1 = ctk.CTkButton(bar, text="Aggiungi metrica", command=app.guard(self._add_metric))
        b2 = ctk.CTkButton(bar, text="Svuota metriche", fg_color="gray40", hover_color="gray30",
                           command=self._clear_metrics)
        b1.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        b2.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.metrics_info = W.hint(left, "Metriche: nessuna")
        self.metrics_info.grid(row=5, column=0, columnspan=2, sticky="ew", padx=12)
        self.count_rows = tk.BooleanVar(value=True)
        self.sort_desc = tk.BooleanVar(value=True)
        c1 = ctk.CTkCheckBox(left, text="Includi il conteggio delle righe (n_righe)",
                             variable=self.count_rows)
        c2 = ctk.CTkCheckBox(left, text="Ordina per la prima metrica (decrescente)",
                             variable=self.sort_desc)
        c1.grid(row=6, column=0, columnspan=2, sticky="w", padx=12, pady=3)
        c2.grid(row=7, column=0, columnspan=2, sticky="w", padx=12, pady=3)
        bar = ctk.CTkFrame(left, fg_color="transparent")
        bar.grid(row=8, column=0, columnspan=2, sticky="ew", padx=12, pady=(6, 12))
        bar.grid_columnconfigure((0, 1), weight=1)
        b3 = ctk.CTkButton(bar, text="Calcola anteprima", fg_color="gray40", hover_color="gray30",
                           command=app.guard(self._gb_preview))
        b4 = ctk.CTkButton(bar, text="Applica come dataset", command=app.guard(self._gb_apply))
        b3.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        b4.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        app.data_widgets += [self.gb_keys, self.gb_col, self.gb_func, b1, b2, c1, c2, b3, b4]

        right = W.section(tab, "Risultato dell'aggregazione")
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        right.grid_rowconfigure(2, weight=1)
        self.gb_info = W.hint(right, "Premi «Calcola anteprima» per vedere il risultato senza "
                                     "modificare il dataset.")
        self.gb_info.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12)
        self.gb_table = W.DataTable(right)
        self.gb_table.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=12, pady=(4, 12))

    def _add_metric(self):
        col = W.get_option(self.gb_col)
        if col is None:
            raise DataError("Seleziona la colonna su cui calcolare la metrica.")
        m = (col, te.AGG_FUNCS[self.gb_func.get()])
        if m not in self.metrics:
            self.metrics.append(m)
        self._update_metrics()

    def _clear_metrics(self):
        self.metrics.clear()
        self._update_metrics()

    def _update_metrics(self):
        text = ", ".join(f"{f}({c})" for c, f in self.metrics) or "nessuna"
        self.metrics_info.configure(text=f"Metriche: {text}")

    def _gb_args(self):
        keys = self.app.cols(self.gb_keys.selected())
        metrics = [(self.app.col(c), f) for c, f in self.metrics]
        first = f"{metrics[0][0]}_{metrics[0][1]}" if metrics else "n_righe"
        return keys, metrics, self.count_rows.get(), first if self.sort_desc.get() else None

    def _gb_preview(self):
        if not self.app.require_data():
            return
        args, df = self._gb_args(), self.app.df

        def done(res):
            self.gb_table.show_df(res.df, 1000)
            self.gb_info.configure(text=res.message + "  (anteprima: dataset non modificato)")

        self.app.run_task("Aggregazione…", lambda: te.group_by(df, *args), done)

    def _gb_apply(self):
        args = self._gb_args()

        def done(res):
            self.gb_table.show_df(res.df, 1000)
            self.gb_info.configure(text=res.message + "  (applicato: usa «Annulla ultima» per tornare indietro)")

        self.app.apply_op("Aggregazione…", lambda df: te.group_by(df, *args), on_done=done)

    # -------------------------------------------------------------- Editor SQL
    def _build_sql(self, tab):
        app = self.app
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)
        top = W.section(tab, "Query SQL (DuckDB) sui dati in memoria",
                        "Tabelle disponibili: «df» = dataset principale, «df2» = dataset secondario. "
                        "Racchiudi tra \"doppi apici\" i nomi di colonna con spazi o maiuscole. "
                        "Esegui con Ctrl+Invio.")
        top.grid(row=0, column=0, sticky="ew")
        top.grid_columnconfigure(0, weight=1)
        self.sql_box = ctk.CTkTextbox(top, height=150, font=(W.MONO, 13), wrap="none")
        self.sql_box.grid(row=2, column=0, columnspan=2, sticky="ew", padx=12, pady=4)
        self.sql_box.insert("1.0", "SELECT *\nFROM df\nLIMIT 100")
        self.sql_box.bind("<Control-Return>", lambda _e: (app.guard(self._run_sql)(), "break")[1])
        bar = ctk.CTkFrame(top, fg_color="transparent")
        bar.grid(row=3, column=0, columnspan=2, sticky="ew", padx=12, pady=(4, 12))
        b1 = ctk.CTkButton(bar, text="Esegui query", width=150, command=app.guard(self._run_sql))
        self.btn_use_sql = ctk.CTkButton(bar, text="Usa risultato come dataset", width=210,
                                         command=app.guard(self._use_sql))
        b3 = ctk.CTkButton(bar, text="Esempi", width=90, fg_color="gray40", hover_color="gray30",
                           command=self._sql_examples)
        b1.pack(side="left")
        self.btn_use_sql.pack(side="left", padx=8)
        b3.pack(side="left")
        self.sql_info = W.hint(bar, "")
        self.sql_info.pack(side="left", padx=12)
        app.data_widgets += [b1, b3]

        res = W.section(tab, "Risultato")
        res.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        res.grid_rowconfigure(1, weight=1)
        res.grid_columnconfigure(0, weight=1)
        self.sql_table = W.DataTable(res)
        self.sql_table.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=12, pady=(0, 12))

    def _sql_examples(self):
        df = self.app.df
        cols = [str(c) for c in df.columns] if df is not None else ["colonna"]
        num = self.app.column_labels("numeric") or cols
        cat = self.app.column_labels("categorical") or cols
        q = (f'-- Aggregazione per categoria\nSELECT "{cat[0]}", COUNT(*) AS n, AVG("{num[0]}") AS media\n'
             f'FROM df\nGROUP BY 1\nORDER BY n DESC\nLIMIT 50;\n\n'
             f'-- Filtro\n-- SELECT * FROM df WHERE "{num[0]}" > 0 LIMIT 100;\n\n'
             f'-- Join con il secondario\n-- SELECT * FROM df JOIN df2 USING ("{cols[0]}");')
        self.sql_box.delete("1.0", "end")
        self.sql_box.insert("1.0", q)

    def _secondary_code(self) -> str:
        return self.app.secondary["code"] if self.app.secondary else ""

    def _tables(self) -> dict:
        tables = {"df": self.app.df}
        if self.app.secondary:
            tables[SECONDARY_NAME] = self.app.secondary["df"]
        return tables

    def _run_sql(self):
        if not self.app.require_data():
            return
        query = self.sql_box.get("1.0", "end")
        # Si esegue solo l'ultima istruzione non commentata se ce ne sono più di una.
        stmts = [s for s in (x.strip() for x in query.split(";"))
                 if s and not all(ln.strip().startswith("--") or not ln.strip() for ln in s.splitlines())]
        if not stmts:
            raise DataError("Scrivi una query SQL da eseguire.")
        query = stmts[0]
        tables = self._tables()

        def done(result):
            self.sql_result, self.sql_query = result, query
            self.sql_table.show_df(result, 1000)
            self.sql_info.configure(text=f"{cleaner.fmt_int(len(result))} righe × {result.shape[1]} "
                                         "colonne" + (" (mostrate 1.000)" if len(result) > 1000 else ""))
            self.app.record("Query SQL (DuckDB)",
                            sqe.query_code(query, list(tables), "risultato_sql", self._secondary_code()),
                            ("import duckdb",))
            self.on_controls("normal")

        self.app.run_task("Esecuzione query SQL…", lambda: sqe.run_query(query, tables), done)

    def _use_sql(self):
        if self.sql_result is None:
            raise DataError("Esegui prima una query.")
        result, query, tables = self.sql_result, self.sql_query, list(self._tables())
        sec_code = self._secondary_code()

        def op(_df):
            return cleaner.OpResult(result, f"Dataset sostituito dal risultato della query SQL "
                                            f"({cleaner.fmt_int(len(result))} righe)",
                                    code=sqe.query_code(query, tables, "df", sec_code),
                                    imports=("import duckdb",))

        self.app.apply_op("Applicazione del risultato SQL…", op)

    # ------------------------------------------------------- Import database
    def _build_db(self, tab):
        app = self.app
        tab.grid_columnconfigure((0, 1), weight=1, uniform="d")
        tab.grid_rowconfigure(0, weight=1)

        sec = W.section(tab, "SQLite", "Importa una tabella o il risultato di una query da un file "
                                       ".db / .sqlite.")
        sec.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        ctk.CTkButton(sec, text="Apri database SQLite…", command=app.guard(self._open_sqlite)).grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=12, pady=4)
        self.sqlite_info = W.hint(sec, "Nessun database aperto.")
        self.sqlite_info.grid(row=3, column=0, columnspan=2, sticky="ew", padx=12)
        self.sqlite_table = W.option_row(sec, 4, "Tabella:", [],
                                         command=lambda v: self._set_query(self.sqlite_query,
                                                                           sqe.table_query(v)))
        ctk.CTkLabel(sec, text="Query:").grid(row=5, column=0, sticky="nw", padx=12, pady=4)
        self.sqlite_query = ctk.CTkTextbox(sec, height=90, font=(W.MONO, 12))
        self.sqlite_query.grid(row=5, column=1, sticky="ew", padx=12, pady=4)
        self.sqlite_target = ctk.CTkSegmentedButton(sec, values=["Come principale", "Come secondario"])
        self.sqlite_target.set("Come principale")
        self.sqlite_target.grid(row=6, column=1, sticky="w", padx=12, pady=4)
        ctk.CTkButton(sec, text="Importa", command=app.guard(self._import_sqlite)).grid(
            row=7, column=0, columnspan=2, sticky="ew", padx=12, pady=(6, 12))

        sec = W.section(tab, "PostgreSQL", "La password non viene salvata né scritta nello script "
                                           "generato (usa la variabile d'ambiente PGPASSWORD).")
        sec.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self.pg = {
            "host": W.entry_row(sec, 2, "Host:", "localhost"),
            "port": W.entry_row(sec, 3, "Porta:", "5432"),
            "database": W.entry_row(sec, 4, "Database:", placeholder="nome database"),
            "user": W.entry_row(sec, 5, "Utente:", "postgres"),
            "password": W.entry_row(sec, 6, "Password:", show="•"),
        }
        ctk.CTkButton(sec, text="Connetti ed elenca tabelle", command=app.guard(self._pg_tables)).grid(
            row=7, column=0, columnspan=2, sticky="ew", padx=12, pady=4)
        self.pg_table = W.option_row(sec, 8, "Tabella:", [],
                                     command=lambda v: self._set_query(self.pg_query,
                                                                       sqe.pg_table_query(v) + " LIMIT 100000"))
        ctk.CTkLabel(sec, text="Query:").grid(row=9, column=0, sticky="nw", padx=12, pady=4)
        self.pg_query = ctk.CTkTextbox(sec, height=90, font=(W.MONO, 12))
        self.pg_query.grid(row=9, column=1, sticky="ew", padx=12, pady=4)
        self.pg_target = ctk.CTkSegmentedButton(sec, values=["Come principale", "Come secondario"])
        self.pg_target.set("Come principale")
        self.pg_target.grid(row=10, column=1, sticky="w", padx=12, pady=4)
        ctk.CTkButton(sec, text="Importa", command=app.guard(self._import_pg)).grid(
            row=11, column=0, columnspan=2, sticky="ew", padx=12, pady=(6, 12))

    @staticmethod
    def _set_query(box, text):
        box.delete("1.0", "end")
        box.insert("1.0", text)

    def _open_sqlite(self):
        path = self.app.ask_open("Seleziona un database SQLite",
                                 [("Database SQLite", "*.db *.sqlite *.sqlite3 *.db3"),
                                  ("Tutti i file", "*.*")])
        if not path:
            return

        def done(tables):
            self.sqlite_path = path
            self.sqlite_info.configure(text=f"{os.path.basename(path)}: {len(tables)} tabelle/viste")
            W.set_options(self.sqlite_table, tables)
            if tables:
                self._set_query(self.sqlite_query, sqe.table_query(tables[0]))

        self.app.run_task("Lettura del database…", lambda: sqe.sqlite_tables(path), done)

    def _import(self, label: str, as_primary: bool, loader, code_fn, imports):
        if as_primary and self.app.dirty and not self.app.confirm(
                "Ci sono modifiche non salvate al dataset principale.\nSostituirlo?", True):
            return

        def task():
            df = loader()
            if df.shape[1] == 0:
                raise DataError("La query non ha restituito colonne.")
            return df, (cleaner.summarize(df) if as_primary else None)

        def done(res):
            df, summary = res
            if as_primary:
                self.app.load_dataset(df, summary, label, f"Importato da {label}",
                                      code_fn("df"), imports)
            else:
                self.set_secondary(df, label, "\n".join([*imports, code_fn("df_dx")]))
                self.sub.set("Merge & Concat")

        self.app.run_task(f"Importazione da {label}…", task, done)

    def _import_sqlite(self):
        if not self.sqlite_path:
            raise DataError("Apri prima un database SQLite.")
        path, query = self.sqlite_path, self.sqlite_query.get("1.0", "end").strip()
        self._import(f"SQLite {os.path.basename(path)}", self.sqlite_target.get() == "Come principale",
                     lambda: sqe.read_sqlite(path, query),
                     lambda var: sqe.sqlite_code(path, query, var),
                     ("import sqlite3", "from contextlib import closing"))

    def _pg_args(self):
        return [self.pg[k].get().strip() for k in ("host", "port", "database", "user")] + \
            [self.pg["password"].get()]

    def _pg_tables(self):
        args = self._pg_args()

        def done(tables):
            W.set_options(self.pg_table, tables)
            if tables:
                self._set_query(self.pg_query, sqe.pg_table_query(tables[0]) + " LIMIT 100000")
            self.app.info(f"Connessione riuscita: {len(tables)} tabelle/viste trovate.")

        self.app.run_task("Connessione a PostgreSQL…", lambda: sqe.postgres_tables(*args), done)

    def _import_pg(self):
        args = self._pg_args()
        query = self.pg_query.get("1.0", "end").strip()
        host, port, db, user, _ = args
        self._import(f"PostgreSQL {db}@{host}", self.pg_target.get() == "Come principale",
                     lambda: sqe.read_postgres(*args, query),
                     lambda var: sqe.postgres_code(host, port, db, user, query, var),
                     ("import os", "from sqlalchemy import create_engine, text",
                      "from sqlalchemy.engine import URL"))

    # -------------------------------------------------------- Aggiornamento
    def on_controls(self, state: str):
        merge_state = state if self.app.secondary else "disabled"
        for w in self.merge_widgets:
            w.configure(state=merge_state)
        self.btn_use_sql.configure(state=state if self.sql_result is not None else "disabled")

    def on_data_changed(self):
        labels = self.app.column_labels()
        W.set_options(self.left_key, labels)
        W.set_options(self.gb_col, labels)
        self.gb_keys.set_columns(labels)
        valid = set(labels)
        self.metrics = [m for m in self.metrics if m[0] in valid]
        self.key_pairs = [p for p in self.key_pairs if p[0] in valid]
        self._update_metrics()
        self._update_pairs()
        self._refresh_secondary()
