"""Registro delle operazioni e generazione dello script Python/pandas equivalente."""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BASE_IMPORTS = ("import pandas as pd", "import numpy as np")


@dataclass(frozen=True)
class Op:
    kind: str            # "load" | "transform" | "analysis"
    description: str
    code: str
    imports: tuple = field(default_factory=tuple)


def lit(value) -> str:
    """Rappresentazione Python valida di un valore (anche numpy/pandas)."""
    if isinstance(value, (list, tuple)):
        inner = ", ".join(lit(v) for v in value)
        return f"[{inner}]" if isinstance(value, list) else f"({inner}{',' if len(value) == 1 else ''})"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{lit(k)}: {lit(v)}" for k, v in value.items()) + "}"
    if isinstance(value, (pd.Timestamp, _dt.datetime)):
        return f"pd.Timestamp({str(value)!r})"
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return "np.nan"
    return repr(value)


def build_script(ops: list[Op], include_analysis: bool = True) -> str:
    chosen = [op for op in ops if include_analysis or op.kind != "analysis"]
    imports = list(BASE_IMPORTS)
    for op in chosen:
        for imp in op.imports:
            if imp not in imports:
                imports.append(imp)
    lines = [
        '"""',
        "Script generato automaticamente da Data Core Analytics",
        f"il {_dt.datetime.now():%d/%m/%Y alle %H:%M}.",
        "",
        "Riproduce in pandas le operazioni eseguite nell'applicazione.",
        '"""',
        *sorted(imports, key=lambda s: (not s.startswith("import"), s)),
        "",
    ]
    if not chosen:
        lines += ["", "# Nessuna operazione registrata: carica un file per iniziare."]
    for i, op in enumerate(chosen, 1):
        tag = {"load": "Caricamento", "transform": "Trasformazione", "analysis": "Analisi"}[op.kind]
        lines += ["", f"# --- {i}. [{tag}] {op.description}", op.code.rstrip()]
    if any(op.kind == "load" for op in chosen):
        lines +=["", "print(df.shape)", "print(df.head())"]
    return "\n".join(lines) + "\n"
