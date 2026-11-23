"""Wrap each tabular in tables/*.tex in \\resizebox so it fits its column (idempotent)."""
import glob, sys, re
for p in glob.glob(sys.argv[1] + "/tables/*.tex"):
    s = open(p, encoding="utf-8").read()
    if "resizebox" in s: continue
    width = r"\textwidth" if "table*" in s else r"\columnwidth"
    s = re.sub(r"(\\begin\{tabular\})", r"\\resizebox{%s}{!}{%%\n\1" % width.replace("\\", "\\\\"), s, count=1)
    s = s.replace("\\end{tabular}", "\\end{tabular}}", 1)
    open(p, "w", encoding="utf-8").write(s); print("fit", p)
# font: TeX Gyre Termes (Times clone) under XeTeX
m = sys.argv[1] + "/main.tex"; s = open(m, encoding="utf-8").read()
if "fontspec" not in s:
    s = s.replace("\\IEEEoverridecommandlockouts\n", "\\IEEEoverridecommandlockouts\n\\usepackage{fontspec}\n"
                  "\\setmainfont{texgyretermes}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,"
                  "ItalicFont=*-italic,BoldItalicFont=*-bolditalic]\n", 1)
    open(m, "w", encoding="utf-8").write(s); print("font set")
