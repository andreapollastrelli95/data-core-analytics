"""SQL Hub: query DuckDB sui DataFrame in memoria e importazione da SQLite/PostgreSQL."""
from __future__ import annotations

import os
import sqlite3
from contextlib import closing

import pandas as pd

from cleaner import DataError

MAX_RESULT_ROWS = 5_000_000


# --------------------------------------------------------------------------- #
# DuckDB
# --------------------------------------------------------------------------- #
def run_query(query: str, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Esegue una query DuckDB; ogni DataFrame in `tables` è interrogabile con il suo nome."""
    query = query.strip().rstrip(";").strip()
    if not query:
        raise DataError("Scrivi una query SQL da eseguire.")
    try:
        import duckdb
    except ImportError as exc:
        raise DataError("Per l'editor SQL serve il pacchetto 'duckdb'.") from exc

    con = duckdb.connect(":memory:")
    try:
        for name, df in tables.items():
            if df is not None:
                con.register(name, _duckdb_safe(df))
        try:
            rel = con.execute(query)
        except duckdb.Error as exc:
            raise DataError(f"Errore nella query SQL:\n\n{exc}") from exc
        if rel.description is None:
            raise DataError("La query non restituisce risultati. Usa un'istruzione SELECT.")
        result = rel.fetch_df()
    finally:
        con.close()
    if len(result) > MAX_RESULT_ROWS:
        raise DataError(f"Il risultato ha più di {MAX_RESULT_ROWS:,} righe: aggiungi un "
                        "filtro o una clausola LIMIT.")
    return result


def _duckdb_safe(df: pd.DataFrame) -> pd.DataFrame:
    """DuckDB non accetta colonne object con tipi misti: le converte in testo."""
    out = df.copy(deep=False)
    out.columns = [str(c) for c in out.columns]
    for col in out.columns:
        s = out[col]
        if s.dtype == object:
            kind = pd.api.types.infer_dtype(s, skipna=True)
            if kind not in ("string", "empty"):
                out[col] = s.map(lambda v: v if pd.isna(v) else str(v))
    return out


PY_VARS = {"df": "df", "df2": "df_dx"}  # nome tabella SQL -> variabile nello script generato


def query_code(query: str, tables: list[str], target: str = "df", secondary_code: str = "") -> str:
    reg = "\n".join(f'con.register("{t}", {PY_VARS.get(t, t)})' for t in tables)
    if secondary_code and "df2" in tables:
        reg = f"{secondary_code}\n{reg}"
    q = query.strip().rstrip(";").replace('"""', "'''")
    return (f'con = duckdb.connect()\n{reg}\n'
            f'{target} = con.execute("""\n{q}\n""").fetch_df()\ncon.close()')


# --------------------------------------------------------------------------- #
# SQLite
# --------------------------------------------------------------------------- #
def _check_sqlite(path: str) -> None:
    if not os.path.isfile(path):
        raise DataError(f"File non trovato:\n{path}")
    with open(path, "rb") as fh:
        if fh.read(16) != b"SQLite format 3\x00":
            raise DataError("Il file selezionato non è un database SQLite valido.")


def sqlite_tables(path: str) -> list[str]:
    _check_sqlite(path)
    try:
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as con:
            rows = con.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view') "
                               "AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
    except sqlite3.Error as exc:
        raise DataError(f"Impossibile leggere il database SQLite:\n{exc}") from exc
    return [r[0] for r in rows]


def read_sqlite(path: str, query: str) -> pd.DataFrame:
    _check_sqlite(path)
    if not query.strip():
        raise DataError("Indica una tabella o una query SQL.")
    try:
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as con:
            return pd.read_sql_query(query, con)
    except (sqlite3.Error, pd.errors.DatabaseError) as exc:
        raise DataError(f"Errore nella query SQLite:\n\n{exc}") from exc


def sqlite_code(path: str, query: str, var: str = "df") -> str:
    return (f"with sqlite3.connect({path!r}) as con:\n"
            f"    {var} = pd.read_sql_query({query!r}, con)")


def table_query(table: str) -> str:
    return 'SELECT * FROM "' + table.replace('"', '""') + '"'


# --------------------------------------------------------------------------- #
# PostgreSQL
# --------------------------------------------------------------------------- #
def _pg_engine(host: str, port: str, database: str, user: str, password: str):
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.engine import URL
    except ImportError as exc:
        raise DataError("Per PostgreSQL servono i pacchetti 'sqlalchemy' e 'psycopg2'.") from exc
    if not host or not database:
        raise DataError("Indica almeno host e nome del database.")
    try:
        port_n = int(port or 5432)
    except ValueError as exc:
        raise DataError("La porta deve essere un numero (di solito 5432).") from exc
    url = URL.create("postgresql+psycopg2", username=user or None, password=password or None,
                     host=host, port=port_n, database=database)
    return create_engine(url, connect_args={"connect_timeout": 8}, pool_pre_ping=True)


def _pg_error(exc: Exception) -> DataError:
    text = str(getattr(exc, "orig", exc) or exc).strip()
    return DataError(f"Errore PostgreSQL:\n\n{text}")


def postgres_tables(host, port, database, user, password) -> list[str]:
    from sqlalchemy import inspect
    from sqlalchemy.exc import SQLAlchemyError

    engine = _pg_engine(host, port, database, user, password)
    try:
        insp = inspect(engine)
        tables = []
        for schema in insp.get_schema_names():
            if schema in ("information_schema",) or schema.startswith("pg_"):
                continue
            tables += [f"{schema}.{t}" for t in insp.get_table_names(schema=schema)]
            tables += [f"{schema}.{v}" for v in insp.get_view_names(schema=schema)]
        return sorted(tables)
    except SQLAlchemyError as exc:
        raise _pg_error(exc) from exc
    finally:
        engine.dispose()


def read_postgres(host, port, database, user, password, query: str) -> pd.DataFrame:
    from sqlalchemy import text
    from sqlalchemy.exc import SQLAlchemyError

    if not query.strip():
        raise DataError("Indica una tabella o una query SQL.")
    engine = _pg_engine(host, port, database, user, password)
    try:
        with engine.connect() as con:
            return pd.read_sql_query(text(query), con)
    except SQLAlchemyError as exc:
        raise _pg_error(exc) from exc
    finally:
        engine.dispose()


def pg_table_query(qualified: str) -> str:
    schema, _, table = qualified.partition(".")
    return f'SELECT * FROM "{schema}"."{table}"' if table else f'SELECT * FROM "{schema}"'


def postgres_code(host, port, database, user, query: str, var: str = "df") -> str:
    # La password non viene mai scritta nello script: si legge da una variabile d'ambiente.
    return (
        "engine = create_engine(URL.create(\n"
        f"    'postgresql+psycopg2', username={user!r}, password=os.environ.get('PGPASSWORD'),\n"
        f"    host={host!r}, port={int(port or 5432)}, database={database!r}))\n"
        "with engine.connect() as con:\n"
        f"    {var} = pd.read_sql_query(text({query!r}), con)"
    )
