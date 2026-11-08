# Submission Guide — Machine Intelligence Quotient (HIQ) Framework
**Author:** Dr. Jagadeesh Selvaraj  
**Affiliation:** NextGen Technologies Private Limited, Chennai, India  
**Email:** dr.jagadeeshselvaraj@nextgentpl.com  
**ORCID:** 0009-0007-2314-4057  
**Date:** November 2026

---

## Submission Options

### Option 1 — Journal of Open Source Software (JOSS)
**What to submit:** The GitHub repository containing the Python library + paper.md

**Steps:**
1. Create a public GitHub repo named `hiq-score`
2. Push the entire `python_library/` folder contents to the repo root
3. Place `joss_paper/paper.md` and `joss_paper/paper.bib` in a `paper/` subfolder
4. Tag a release: `git tag v0.1.0 && git push --tags`
5. Go to https://joss.theoj.org/papers/new
6. Enter the GitHub URL and confirm the paper path (`paper/paper.md`)
7. JOSS review: free, typically 2–8 weeks, issues a citable DOI

---

### Option 2 — MDPI *AI* Journal (full journal article)
**What to submit:** The main article (PDF + DOCX)

**Steps:**
1. Go to https://www.mdpi.com/journal/ai and click "Submit"
2. Upload `main_article/Machine_IQ_Journal_Article_v11.docx` as the manuscript
3. Select article type: **Article**
4. Suggested section: **AI Methods**
5. APC waiver available — apply during submission
6. Typical turnaround: 4–6 weeks

---

### Option 3 — arXiv preprint (immediate, establishes priority)
**What to submit:** PDF of the main article

**Steps:**
1. Go to https://arxiv.org/submit
2. Select primary category: **cs.AI**
3. Cross-list: **stat.AP** (statistics applications)
4. Upload `main_article/Machine_IQ_Journal_Article_v11.pdf`
5. Posts within 1 business day; free

**Recommended strategy:** Submit arXiv first (same day), then submit JOSS + a journal in parallel.

---

## File Inventory

```
submission_package/
│
├── main_article/
│   ├── Machine_IQ_Journal_Article_v11.pdf   ← Main paper (PDF, for arXiv / journal)
│   └── Machine_IQ_Journal_Article_v11.docx  ← Main paper (DOCX, for journal submission)
│
├── joss_paper/
│   ├── paper.md    ← JOSS submission paper (Markdown, goes in paper/ of GitHub repo)
│   └── paper.bib   ← BibTeX references for paper.md
│
├── python_library/
│   ├── hiq_score/          ← Python package source
│   │   ├── __init__.py
│   │   ├── core.py         ← HIQ formula
│   │   ├── norm.py         ← LOO-CV norm estimation
│   │   ├── reliability.py  ← SE, Spearman-Brown, min_questions
│   │   ├── validity.py     ← Contamination & drift corrections
│   │   └── meridian.py     ← 4-component Meridian decomposition
│   ├── tests/              ← 32 unit tests (all passing)
│   ├── pyproject.toml      ← Build config (pip-installable)
│   ├── README.md           ← Package documentation
│   ├── LICENSE             ← MIT licence
│   └── .github/workflows/ci.yml  ← GitHub Actions CI
│
└── supplementary/
    └── SUBMISSION_GUIDE.md  ← This file
```

---

## Key Numbers to Know

| Item | Value |
|---|---|
| HIQ formula | HIQ = 100 + 15(Φ − m) / s |
| Pilot norm mean (Halawi 2024) | m̂ = 0.1264 |
| Pilot norm SD (Halawi 2024) | ŝ = 0.0321 |
| Contamination formula | ΔHIQ = 408δ |
| Norm drift rate | 3.0–3.75 HIQ pts/year |
| LOO-CV range (Halawi 5-system) | 60.1 (Gemini-Pro) → 130.6 (GPT-4-1106) |
| ForecastBench norm mean | m̂ = 0.4728 |
| ForecastBench norm SD | ŝ = 0.0396 |
| Test suite | 32 tests, 0 failures |
| Python version requirement | ≥ 3.9 |
| External dependencies | None |
