# AllergenScan — Automated FoodEx2 Coding and Allergen Detection for Food Products

**Master's Thesis (TFM) · Alejandro Márquez González**

---

## Overview

AllergenScan is a system that automatically assigns **FoodEx2 food classification codes** to ingredient lists from the Open Food Facts (OFF) database and uses those codes to detect **EU regulated allergens**. The goal is to support food-safety workflows by replacing manual coding with a reproducible, confidence-scored pipeline and a Streamlit review interface.

---

## Problem Statement

Food product labelling requires accurate allergen declarations. Manual mapping of free-text ingredient lists to standardised classification systems (FoodEx2) is labour-intensive and error-prone. This project investigates whether a hybrid NLP + fuzzy-matching approach can automate that mapping at scale, quantify its own uncertainty, and flag products that need human review.

---

## System Architecture

```
Open Food Facts DB (DuckDB)
        │
        ▼
┌─────────────────────┐
│   ETL Pipeline      │  src/etl/
│  cleaning.py        │  — ingredient tag parsing, noise/E-code filtering,
│  build_*.py         │    matchability classification, FoodEx2 term export
└────────┬────────────┘
         │  products_ingredients.parquet
         │  foodex2_terms.parquet
         ▼
┌─────────────────────────────────────┐
│   Stratified Matcher                │  src/matching/
│   StratifiedMatcherOptimized        │  — RapidFuzz exact/reranked pass
│   run_matching_bge_small_chunked.py │  — BGE-small beam search cascade
└────────┬────────────────────────────┘
         │  matching_results_ingredients_*.parquet
         ▼
┌─────────────────────┐
│   AllergenDetector  │  allergenscan/
│   detector.py       │  — FoodEx2 → EU allergen mapping
│                     │  — Ambiguous ingredient pattern matching
└────────┬────────────┘
         ▼
┌─────────────────────┐
│   Streamlit App     │  streamlit_app/
│   app.py            │  — Pre-computed product browser
│   coding_assistant/ │  — Live ingredient coding + review UI
└─────────────────────┘
```

---

## Matching Pipeline

Ingredient matching uses a three-stage hybrid strategy:

1. **RapidFuzz exact pass** (`rf_exact`) — direct fuzzy string match against all FoodEx2 term variants (inverted word order, British/American spelling). Score threshold: 95/100.
2. **RapidFuzz + BGE reranking** (`rf_reranked`) — candidates above a lower RF threshold are re-ranked by cosine similarity with `BAAI/bge-small-en-v1.5` embeddings, weighted by an L2 branch prior.
3. **BGE beam search** (`beam`) — hierarchical cascade L2 → L3 → specific level using dense embeddings. Explores multiple branches per level instead of committing greedily.

A composite **path\_score** combines all signals:

```
path_score = 0.50 × BGE_score + 0.28 × (RF_score / 100) + 0.10 × lexical_Jaccard + 0.12 × L2_prior
```

Product-level confidence aggregates per-ingredient scores:

```
confidence = 0.4 × avg_path_score + 0.3 × HIGH_ratio + 0.2 × coverage + 0.1 × MEDIUM_ratio
```

---

## Project Structure

```
src/
  etl/
    cleaning.py                   — Token cleaning, validation, matchability filters
    build_products_ingredients.py — Expand OFF products → per-ingredient rows
    build_foodex2_terms.py        — Export FoodEx2 hierarchy + facets to parquet
  matching/
    stratified_matcher.py         — Core matching logic (RF + BGE cascade)
    run_matching_bge_small_chunked.py — Batch matching over full ingredient vocabulary

allergenscan/
  allergenscan/
    detector.py                   — Allergen detection from FoodEx2 codes
    comparator.py                 — Declared vs. detected allergen comparison
    models.py                     — Domain dataclasses and enums

streamlit_app/
  app.py                          — Main UI (product browser + live analysis)
  coding_assistant/
    ingredient_matcher.py         — Runtime matcher (cache-first + live fallback)
    product_analyzer.py           — Full analysis pipeline orchestration
    confidence_scorer.py          — Product-level confidence formula
    review_classifier.py          — AUTO_APPROVED / NEEDS_REVIEW / MANUAL_REQUIRED
    models.py                     — IngredientMatch, ProductAnalysis dataclasses
  utils/
    foodex2_info.py               — Hierarchy path, children, facet, allergen lookups
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Ingredient-level parquet (not product-level) | Avoids many-to-many join inflation when the same ingredient appears in thousands of products |
| `matchable = true` filter | Only English, non-noise, non-ancestor-only tags are sent to the matcher |
| Cache-first matching | Pre-computed results for ~28 K unique ingredients; live BGE inference only on cache miss |
| Confidence thresholds L2/L3/specific: 0.55 / 0.70 / 0.85 | Calibrated on stratified sample; lower thresholds fall back to a coarser level rather than forcing a wrong specific match |
| Branch penalty in RF reranking | Penalises RF candidates whose L2 BGE prior is weak, preventing cross-category RF hits |

---

## Tech Stack

| Component | Library / Tool |
|---|---|
| Database | DuckDB |
| Embeddings | `sentence-transformers` · `BAAI/bge-small-en-v1.5` |
| Fuzzy matching | RapidFuzz |
| Data manipulation | pandas · NumPy |
| UI | Streamlit |
| Source data | Open Food Facts (CC-BY-SA 4.0) · EFSA FoodEx2 |

---

## Running the Pipeline

```bash
# 1. Build ingredient and FoodEx2 term tables
python -m src.etl.build_products_ingredients
python -m src.etl.build_foodex2_terms

# 2. Run batch matching (produces matching_results_ingredients_*.parquet)
python -m src.matching.run_matching_bge_small_chunked

# 3. Launch the review interface
streamlit run streamlit_app/app.py
```

---

## Allergen Coverage

The system covers all **14 EU-regulated allergens** (Regulation EU 1169/2011): cereals containing gluten, crustaceans, eggs, fish, peanuts, soybeans, milk, tree nuts, celery, mustard, sesame, sulphites, lupin, and molluscs. Detection combines FoodEx2 code → allergen mapping with a pattern-based ambiguous-ingredient checker.
