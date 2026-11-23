# IEEEtran conference format notes

## Sources

- IEEE, "Manuscript Templates for Conference Proceedings": https://www.ieee.org/conferences/publishing/templates.html. This page has the official LaTeX template ZIP, the LaTeX bibliography files ZIP, and template instructions, and it links to Overleaf.
- IEEE official templates on Overleaf: https://www.overleaf.com/gallery/tagged/ieee-official
- IEEEtran class on CTAN (v1.8b, maintainer Michael Shell): https://ctan.org/pkg/ieeetran. Includes `bare_conf.tex` and `IEEEtran_HOWTO.pdf` (https://mirrors.ctan.org/macros/latex/contrib/IEEEtran/IEEEtran_HOWTO.pdf).
- IEEEtran BibTeX style: https://ctan.org/pkg/ieeetran-bst (`IEEEtran.bst`, documented in `IEEEtran_bst_HOWTO.pdf`).

**Always check the target conference's call for papers.** Page limits, blind review and the PDF eXpress requirement are set by each conference, not by the template.

## Skeleton (from bare_conf.tex)

```latex
\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts          % only if \thanks is used in the author block
\usepackage{cite}                      % sorts/compresses [1]-[3]
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}                  % optional, cleaner tables
\usepackage{url}                       % or hyperref (load last); IEEE final PDFs usually without colored links

\begin{document}
\title{SentriFormer: ...}

\author{\IEEEauthorblockN{1\textsuperscript{st} Given Surname}
\IEEEauthorblockA{\textit{Dept. name} \\
\textit{Organization}\\
City, Country \\
email@address}
\and
\IEEEauthorblockN{2\textsuperscript{nd} Given Surname}
\IEEEauthorblockA{\textit{Dept.} \\ \textit{Org.} \\ City, Country \\ email}
}

\maketitle

\begin{abstract}
... (no citations, no math-heavy notation; typically 150--250 words)
\end{abstract}

\begin{IEEEkeywords}
transformer, customer support, multi-task learning, response generation
\end{IEEEkeywords}

\section{Introduction}
...
\section*{Acknowledgment}   % IEEE spelling, unnumbered
\bibliographystyle{IEEEtran}
\bibliography{references}
\end{document}
```

## Key conventions

- **Class option:** use `conference`. It gives two columns, 10 pt Times, US letter (add `a4paper` only if the conference asks for it), and no page numbers or headers, because IEEE adds them.
- **Title:** use title case, with no symbols or footnotes in the title. Do not use `\thanks` unless you also use `\IEEEoverridecommandlockouts`.
- **Authors:** use `\IEEEauthorblockN` for the name and `\IEEEauthorblockA` for affiliation and email, and separate authors with `\and`. For many authors, IEEE's template places them in rows. Remove the names for double-blind review if the conference requires it.
- **Keywords:** the `IEEEkeywords` environment goes right after the abstract.
- **Sections:** `\section`, `\subsection` and `\subsubsection` are numbered I, A, 1) automatically. Keep at most three levels.
- **Figures:** use `\begin{figure}[t]` or `[!t]` for one column, and `figure*` for full width (full-width floats go at the top of a page only). The caption goes **below** the figure and reads "Fig. 1." Refer to it in the text as "Fig.~\ref{...}". Use vector PDF or EPS, or raster at 300 dpi or more; fonts must be readable at column width (about 3.5 in, or 8.9 cm).
- **Tables:** the caption goes **above** the table, and tables are numbered with Roman numerals (TABLE I) automatically. Refer to them as "Table~\ref{...}". Avoid vertical rules; `booktabs` is fine.
- **Equations:** use `equation` or `align`, refer to them as "(1)" via `\eqref` or `(\ref{})`, and punctuate them as part of the sentence.
- **Citations:** write `\cite{key}`, which renders as [1]. With `cite` loaded, `\cite{a,b,c}` renders as [1]-[3]. Do not write "in [1]" as a subject; write "Li et al. [5] proposed ...".
- **Bibliography:** `\bibliographystyle{IEEEtran}` plus `\bibliography{references}`. Entries are numbered in order of citation. IEEEtran.bst prints `doi` and `url` fields, so keep one of them per entry, as our bib files do. Abbreviate venues IEEE-style if space is tight (e.g. "Proc." and "Conf."). The bib files already use "Proc.".
- **Footnotes:** sparingly. Do not put footnotes in the abstract.
- **PDF:** IEEE conferences usually require **IEEE PDF eXpress** validation: all fonts embedded, no Type 3 fonts, PDF/A-compatible. In Overleaf, compile with pdfLaTeX and avoid bitmap fonts.
- **Units and numbers:** use SI units, a leading zero (0.25), and en-dashes for ranges in the bib (`--`).

## Page limits (typical, not set by IEEE)

- Most IEEE conferences allow **4 to 6 pages** including references for a regular paper. Many allow 1 or 2 extra pages for a fee, and short papers are usually 2 to 4 pages.
- Some conferences count references separately, so check the CFP.
- Over-length papers are usually rejected without review, so budget about half to one column for references. 30 to 45 IEEE-style references take roughly 0.75 to 1 page in two-column 8 pt.

## Fitting our reference lists

- Paper 1 has 43 bib entries and Paper 2 has 32. BibTeX prints only what you `\cite`, so cite about 25 to 30 in Paper 1 and 20 to 25 in Paper 2 to stay inside a 6-page limit.
- Our keys are plain `authorYEARword` (e.g., `vaswani2017attention`) and are shared across both files where the same work appears.
