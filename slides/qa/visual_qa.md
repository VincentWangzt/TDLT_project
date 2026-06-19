# Slide Visual QA

Artifact: `slides/tdlt_final_project.pdf`

Checks performed:

- Recompiled after the fresh-fit result update with `latexmk -xelatex -interaction=nonstopmode -halt-on-error`.
- Rendered all 76 pages with Poppler `pdftoppm` at 120 DPI.
- Built `slides/qa/contact_sheet.png` from the rendered pages.
- Inspected updated dense result tables, command slides, qualitative plots, appendix parameter/config slides, and references.

Result: no missing figures, clipped formulas, unreadable tables, or visibly broken slide layouts were found in the latest rendered PDF. The deck still contains explicit placeholders for GitHub URL, group members, and contribution details.
