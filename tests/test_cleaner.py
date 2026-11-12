import pandas as pd
import pytest

import cleaner
from cleaner import DataError


def test_load_sample_and_summary(sales):
    s = cleaner.summarize(sales)
    assert (s["rows"], s["cols"]) == (612, 9)
    assert s["duplicates"] == 12
    assert s["total_nulls"] == int(sales.isna().sum().sum())
    by_name = {c["name"]: c for c in s["columns"]}
    assert by_name["Regione"]["nulls"] == 26


def test_load_missing_file():
    with pytest.raises(DataError):
        cleaner.load_file("non_esiste.csv")


def test_csv_with_semicolon_and_decimal_comma(tmp_path):
    path = tmp_path / "it.csv"
    path.write_text("nome;prezzo\nmela;1,50\npera;2,25\n", encoding="utf-8")
    df, code = cleaner.load_file(str(path))
    assert df["prezzo"].tolist() == [1.5, 2.25]
    assert "read_csv" in code


def test_remove_duplicates_and_original_untouched(sales):
    res = cleaner.remove_duplicates(sales)
    assert len(res.df) == 600 and len(sales) == 612
    assert "drop_duplicates" in res.code
    assert cleaner.remove_duplicates(res.df).changed is False


def test_handle_missing_strategies(sales):
    dropped = cleaner.handle_missing(sales, "drop")
    assert dropped.df.isna().sum().sum() == 0
    filled = cleaner.handle_missing(sales, "mean", ["Prezzo Unitario"])
    assert filled.df["Prezzo Unitario"].isna().sum() == 0
    assert filled.df["Prezzo Unitario"].mean() == pytest.approx(sales["Prezzo Unitario"].mean())


def test_snake_case_and_drop_columns(sales):
    res = cleaner.clean_column_names(sales)
    assert "prezzo_unitario" in res.df.columns and "sconto" in res.df.columns
    with pytest.raises(DataError):
        cleaner.drop_columns(sales, list(sales.columns))
    assert cleaner.drop_columns(sales, ["Canale"]).df.shape[1] == 8


def test_cast_datetime_and_strict(sales):
    res = cleaner.cast_columns(sales, ["Data Ordine"], "datetime")
    assert pd.api.types.is_datetime64_any_dtype(res.df["Data Ordine"])
    with pytest.raises(DataError):
        cleaner.cast_columns(sales, ["Regione"], "float", strict=True)


@pytest.mark.parametrize("ext", [".csv", ".xlsx", ".parquet"])
def test_export_roundtrip(sales, tmp_path, ext):
    path = str(tmp_path / f"out{ext}")
    cleaner.export(sales, path)
    back, _ = cleaner.load_file(path)
    assert back.shape == sales.shape
    assert back["Fatturato"].sum() == pytest.approx(sales["Fatturato"].sum())


def test_number_formatting():
    assert cleaner.fmt_int(1234567) == "1.234.567"
    assert cleaner.fmt_num(1234.5) == "1.234,50"
    assert cleaner.pick(1, "{n} riga", "{n} righe") == "1 riga"
