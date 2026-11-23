"""Float placement / structure fixes for both papers (idempotent).
- floats [t] -> [!t]; \\FloatBarrier before Experimental-setup, Results and Limitations sections so floats stay in their section
- full-width figure moved one section earlier (figure* can only appear at the top of a later page)
- balanced last page; smaller workflow figure in SentriMail
"""
import glob, os, re

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def move_block(s, begin_marker, label, before):
    """Cut the figure* environment containing `label` and paste it just before `before`."""
    if label not in s: return s
    i = s.rfind(begin_marker, 0, s.index(label)); j = s.index("\\end{figure*}", i) + len("\\end{figure*}\n")
    block = s[i:j]; s = s[:i] + s[j:]
    k = s.index(before); return s[:k] + block + "\n" + s[k:]


for paper, moves, barriers in [
    ("SentriMail_Paper", [("\\label{fig:arch}", "\\section{Problem Definition")],
     ["\\section{Implementation and Experimental Setup}", "\\section{Results and Discussion}", "\\section{Limitations"]),
    ("SentriFormer_Paper", [("\\label{fig:dev}", "\\subsection{Training recipe}")],
     ["\\section{Experimental Setup}", "\\section{Results and Discussion}", "\\section{Limitations"]),
]:
    p = os.path.join(BASE, paper, "main.tex"); s = open(p, encoding="utf-8").read()
    if "{placeins}" not in s:
        s = s.replace("\\usepackage{cite}", "\\usepackage{cite}\n\\usepackage{placeins}\n\\usepackage{balance}", 1)
    for label, before in moves:
        if s.find(label) > s.find(before):  # only move if it currently comes after the target point
            s = move_block(s, "\\begin{figure*}", label, before)
    for b in barriers:
        if "\\FloatBarrier\n" + b not in s: s = s.replace(b, "\\FloatBarrier\n" + b, 1)
    if "\\balance\n\\bibliographystyle" not in s: s = s.replace("\\bibliographystyle", "\\balance\n\\bibliographystyle", 1)
    s = re.sub(r"\\begin\{(figure\*?|table\*?)\}\[t\]", r"\\begin{\1}[!t]", s)
    if paper == "SentriMail_Paper":
        i = s.index("\\label{fig:flow}"); k = s.rfind("\\resizebox{\\columnwidth}{!}{%", 0, i)
        if k != -1 and s.rfind("\\begin{figure}", 0, i) < k:
            s = s[:k] + "\\resizebox{!}{0.34\\textheight}{%" + s[k + len("\\resizebox{\\columnwidth}{!}{%"):]
    open(p, "w", encoding="utf-8").write(s)
    for t in glob.glob(os.path.join(BASE, paper, "tables", "*.tex")):
        ts = open(t, encoding="utf-8").read(); ts2 = re.sub(r"\\begin\{(table\*?)\}\[t\]", r"\\begin{\1}[!t]", ts)
        if ts2 != ts: open(t, "w", encoding="utf-8").write(ts2)
    print("layout", paper)
