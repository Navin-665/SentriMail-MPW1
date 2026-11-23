"""Replace the author block with a single shared-affiliation IEEE block (both papers)."""
import os, re

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTH = r"""\author{\IEEEauthorblockN{Naveen Kumar N, Lohith M, Vansh Paryani, Yash Kumar and Prof.\ Syeeda Mujeebunnisa}
\IEEEauthorblockA{\textit{Department of Artificial Intelligence and Machine Learning}\\
\textit{BMS College of Engineering}, Bengaluru, India\\

"""
for paper in ["SentriFormer_Paper", "SentriMail_Paper"]:
    p = os.path.join(BASE, paper, "main.tex"); s = open(p, encoding="utf-8").read()
    s = re.sub(r"\\makeatletter\n\\newcommand\{\\linebreakand\}.*?\\makeatother\n", "", s, flags=re.S)
    a = s.index("\\author{"); b = s.index("\\maketitle")
    s = s[:a] + AUTH + s[b:]
    open(p, "w", encoding="utf-8").write(s); print("fixed", paper)
