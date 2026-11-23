"""Authors + wording edits for both papers (idempotent string replacements)."""
import os, re

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AFF = r"\textit{Dept.\ of Artificial Intelligence and Machine Learning} \\ \textit{BMS College of Engineering}\\ Bengaluru, India \\ "
people = [("Naveen Kumar N", "naveenkumarn.ai24@bmsce.ac.in"), ("Lohith M", "lohithm.ai24@bmsce.ac.in"),
          ("Vansh Paryani", "vanshparyani.ai23@bmsce.ac.in"), ("Yash Kumar", "yashkumar.ai23@bmsce.ac.in"),
          ("Prof.\\ Syeeda Mujeebunnisa (Guide)", "syeedamujnaseeba.mel@bmsce.ac.in")]
blocks = [r"\IEEEauthorblockN{%s}" "\n" r"\IEEEauthorblockA{%s\texttt{%s}}" % (n, AFF, e) for n, e in people]
AUTH = "\\author{\n" + "\n\\and\n".join(blocks[:3]) + "\n\\linebreakand\n" + "\n\\and\n".join(blocks[3:]) + "\n}\n\n"
LBA = r"""\makeatletter
\newcommand{\linebreakand}{\end{@IEEEauthorhalign}\hfill\mbox{}\par\mbox{}\hfill\begin{@IEEEauthorhalign}}
\makeatother
"""
REPL = {
    "SentriFormer_Paper/main.tex": [
        ("The backbone and heads use separate learning rates, and the heads are first warmed up for 2k steps with the backbone frozen, so that freshly initialised heads cannot disturb the pre-trained encoder.",
         "The backbone and heads use separate learning rates, and a short 2k-step head warm-up precedes full joint training so that freshly initialised heads do not disturb the pre-trained encoder."),
        ("Joint heads on frozen pre-trained encoder", "Joint task heads on pre-trained encoder"),
        ("accepted under frozen criteria", "accepted under pre-specified criteria"),
        ("Acceptance criteria were frozen in writing before each run", "Acceptance criteria were written down before each run"),
        ("Fine-tuning the model for replies with the heads frozen but the encoder trainable moved the shared embeddings",
         "Fine-tuning the shared model on the reply objective alone moved the shared embeddings"),
        ("under frozen criteria", "under pre-specified criteria"),
    ],
    "SentriFormer_Paper/tables/tab_hparams.tex": [
        ("Phase T (T3)", "Multi-task (SF-MT)"), ("2k-step frozen-backbone head warm-up", "2k-step head warm-up"),
        ("T4a--T6 & as T3", "Context stages & as SF-MT"),
    ],
    "SentriFormer_Paper/tables/tab_neg.tex": [("rejected under frozen criteria", "rejected under pre-specified criteria")],
    "SentriFormer_Paper/tables/tab_data.tex": [(r"\quad T4d filtered", r"\quad + portal filter (SF-Ctx)"),
                                               (r"\quad T5a filtered", r"\quad + reply filter (final)")],
    "SentriMail_Paper/main.tex": [],
}
for rel, pairs in REPL.items():
    p = os.path.join(BASE, rel); s = open(p, encoding="utf-8").read()
    for a, b in pairs:
        if a in s: s = s.replace(a, b)
    if rel.endswith("main.tex"):
        s = re.sub(r"\\author\{.*?\n\}\n\n", lambda m: AUTH, s, count=1, flags=re.S)
        if "linebreakand}{" not in s: s = s.replace("\\begin{document}", LBA + "\n\\begin{document}", 1)
        s = s.replace("[Acknowledgment placeholder: supervisors, institution and funding, to be supplied by the authors.]",
                      "The authors thank Prof.\\ Syeeda Mujeebunnisa for her guidance and BMS College of Engineering, Bengaluru, for providing the facilities used in this work.")
    open(p, "w", encoding="utf-8").write(s); print("edited", rel)
