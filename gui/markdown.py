"""Rendering Markdown essenziale in un widget tk.Text (titoli, grassetto, corsivo, codice,
blocchi di codice, elenchi, citazioni, tabelle, righe orizzontali, blocchi <think>)."""
from __future__ import annotations

import re
import tkinter as tk
import tkinter.font as tkfont

from gui.widgets import FONT, MONO

_INLINE = re.compile(r"(\*\*[^*\n]+\*\*|__[^_\n]+__|`[^`\n]+`|\*[^*\s][^*\n]*\*|\[[^\]\n]+\]\([^)\s]+\))")


def setup_tags(text: tk.Text, colors: dict) -> None:
    base = tkfont.Font(family=FONT, size=10)
    fonts = {
        "h1": tkfont.Font(family=FONT, size=15, weight="bold"),
        "h2": tkfont.Font(family=FONT, size=13, weight="bold"),
        "h3": tkfont.Font(family=FONT, size=11, weight="bold"),
        "bold": tkfont.Font(family=FONT, size=10, weight="bold"),
        "italic": tkfont.Font(family=FONT, size=10, slant="italic"),
        "code": tkfont.Font(family=MONO, size=10),
        "role": tkfont.Font(family=FONT, size=9, weight="bold"),
    }
    text._md_fonts = fonts  # evita che i font vengano raccolti dal garbage collector
    text.configure(font=base, bg=colors["bg"], fg=colors["fg"], insertbackground=colors["fg"],
                   selectbackground=colors["sel"], selectforeground="#ffffff")
    text.tag_configure("h1", font=fonts["h1"], spacing1=8, spacing3=4)
    text.tag_configure("h2", font=fonts["h2"], spacing1=6, spacing3=3)
    text.tag_configure("h3", font=fonts["h3"], spacing1=4, spacing3=2)
    text.tag_configure("bold", font=fonts["bold"])
    text.tag_configure("italic", font=fonts["italic"])
    text.tag_configure("code", font=fonts["code"], background=colors["code_bg"])
    text.tag_configure("codeblock", font=fonts["code"], background=colors["code_bg"],
                       lmargin1=12, lmargin2=12, rmargin=8, spacing1=1, spacing3=1)
    text.tag_configure("bullet", lmargin1=10, lmargin2=26)
    text.tag_configure("quote", lmargin1=14, lmargin2=14, foreground=colors["muted"],
                       font=fonts["italic"])
    text.tag_configure("think", lmargin1=14, lmargin2=14, foreground=colors["muted"],
                       font=fonts["italic"])
    text.tag_configure("link", foreground=colors["link"], underline=True)
    text.tag_configure("hr", foreground=colors["muted"])
    text.tag_configure("table", font=fonts["code"])
    text.tag_configure("role_user", font=fonts["role"], foreground=colors["link"], spacing1=10)
    text.tag_configure("role_ai", font=fonts["role"], foreground=colors["good"], spacing1=10)
    text.tag_configure("note", foreground=colors["warn"], font=fonts["italic"])
    text.tag_configure("user_msg", lmargin1=4, lmargin2=4)


def _inline(text: tk.Text, line: str, extra: tuple = ()) -> None:
    for part in _INLINE.split(line):
        if not part:
            continue
        if (part.startswith("**") and part.endswith("**")) or (part.startswith("__") and part.endswith("__")):
            text.insert("end", part[2:-2], ("bold", *extra))
        elif part.startswith("`") and part.endswith("`") and len(part) > 1:
            text.insert("end", part[1:-1], ("code", *extra))
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            text.insert("end", part[1:-1], ("italic", *extra))
        elif part.startswith("[") and "](" in part:
            label = part[1:part.index("](")]
            text.insert("end", label, ("link", *extra))
        else:
            text.insert("end", part, extra)


def render(text: tk.Text, md: str) -> None:
    """Appende `md` formattato alla fine del widget."""
    in_code = False
    in_think = False
    for raw in md.split("\n"):
        line = raw.rstrip("\r")
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            text.insert("end", line + "\n", ("codeblock",))
            continue
        if stripped.startswith("<think>"):
            in_think = True
            stripped = stripped[len("<think>"):]
            text.insert("end", "Ragionamento del modello:\n", ("think",))
            if not stripped:
                continue
        if in_think:
            end = "</think>" in stripped
            content = stripped.replace("</think>", "")
            if content:
                text.insert("end", content + "\n", ("think",))
            if end:
                in_think = False
                text.insert("end", "\n")
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)", stripped)
        if heading:
            level = min(len(heading.group(1)), 3)
            _inline(text, heading.group(2), (f"h{level}",))
            text.insert("end", "\n", (f"h{level}",))
        elif re.fullmatch(r"[-*_]{3,}", stripped):
            text.insert("end", "─" * 40 + "\n", ("hr",))
        elif re.match(r"^[-*+]\s+", stripped):
            indent = "    " * ((len(line) - len(line.lstrip())) // 2)
            text.insert("end", f"{indent}•  ", ("bullet",))
            _inline(text, re.sub(r"^[-*+]\s+", "", stripped), ("bullet",))
            text.insert("end", "\n", ("bullet",))
        elif re.match(r"^\d+[.)]\s+", stripped):
            num = re.match(r"^(\d+)[.)]\s+", stripped).group(1)
            text.insert("end", f"{num}.  ", ("bullet",))
            _inline(text, re.sub(r"^\d+[.)]\s+", "", stripped), ("bullet",))
            text.insert("end", "\n", ("bullet",))
        elif stripped.startswith(">"):
            _inline(text, stripped.lstrip("> "), ("quote",))
            text.insert("end", "\n", ("quote",))
        elif stripped.startswith("|") and stripped.endswith("|"):
            if re.fullmatch(r"\|[\s:|-]+\|", stripped):
                continue  # riga separatrice della tabella
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            text.insert("end", "  " + "  │  ".join(cells) + "\n", ("table",))
        else:
            _inline(text, line)
            text.insert("end", "\n")
