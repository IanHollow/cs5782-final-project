# Report

This directory contains the final two-page project report and its source.

- `report.typ`: Typst source for the report.
- `114_dora_2page_report.pdf`: compiled final report PDF.
- `refs.bib`: bibliography entries used by the report.
- `charged-ieee.typ`: local report template adapted for the CS 4782 / CS 5782 two-page requirements.

Build from the repository root with:

```bash
typst compile --root . report/report.typ report/114_dora_2page_report.pdf
```
