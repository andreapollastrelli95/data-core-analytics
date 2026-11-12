
import numpy as np
import pandas as pd

import cleaner
import codegen
import report_engine as rep
from codegen import Op
from conftest import SAMPLE


def test_lit_roundtrip():
    for value in ["a'b", 3, 2.5, [1, "x"], ("a",), {"k": 1}, np.int64(7)]:
        assert eval(codegen.lit(value)) == (value.item() if isinstance(value, np.generic) else value)
    assert codegen.lit(float("nan")) == "np.nan"


def test_generated_script_reproduces_the_data(sales):
    """Lo script esportato deve dare lo stesso risultato delle operazioni fatte nell'app."""
    _df, load_code = cleaner.load_file(SAMPLE)
    ops = [Op("load", "Caricato", load_code)]
    df = sales
    for step in (cleaner.remove_duplicates, cleaner.clean_column_names,
                 lambda d: cleaner.handle_missing(d, "median", ["prezzo_unitario"])):
        res = step(df)
        df = res.df
        ops.append(Op("transform", res.message, res.code, res.imports))
    script = codegen.build_script(ops)
    scope: dict = {}
    exec(compile(script, "pipeline.py", "exec"), scope)  # noqa: S102 - codice generato dall'app
    pd.testing.assert_frame_equal(scope["df"], df, check_dtype=False)


def test_script_can_exclude_analysis():
    ops = [Op("load", "x", "df = pd.DataFrame()"), Op("analysis", "stat", "print(1)")]
    assert "print(1)" not in codegen.build_script(ops, include_analysis=False)
    assert "print(1)" in codegen.build_script(ops)


def test_html_and_excel_reports(sales, tmp_path):
    summary = cleaner.summarize(sales)
    html_path, xlsx_path = str(tmp_path / "r.html"), str(tmp_path / "r.xlsx")
    rep.html_report(html_path, sales, summary, SAMPLE, ["[10:00] caricato"])
    text = open(html_path, encoding="utf-8").read()
    assert "Report di sintesi" in text and "data:image/png;base64" in text
    rep.excel_report(xlsx_path, sales, summary, SAMPLE, [])
    sheets = pd.ExcelFile(xlsx_path).sheet_names
    assert "Dati" in sheets and len(sheets) >= 5


def test_markdown_html_escapes_and_structures():
    out = rep.markdown_html("## Titolo\nTesto **forte** <script>\n\n- uno\n- due\n\n"
                            "| a | b |\n|---|---|\n| 1 | 2 |\n\n```\nx < 1\n```")
    assert "<h2>Titolo</h2>" in out and "<b>forte</b>" in out
    assert "<script>" not in out and "&lt;script&gt;" in out
    assert out.count("<li>") == 2 and "<th>a</th>" in out and "<td>2</td>" in out
    assert "<pre>x &lt; 1</pre>" in out
