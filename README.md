# Duplicate Question Pairs Detection

Detecting whether two questions are semantically identical, trained on the Quora Question Pairs dataset.

---

## Problem
Given two questions, classify whether they are duplicates or not.

---

## Approach
A systematic feature-building experiment comparing four feature sets.

| Step | Features | RF Accuracy | XGB Accuracy |
|-----|----------|-------------|--------------|
| 1 | Advanced features only | ~76% | ~76% |
| 2 | BoW + Advanced features | ~80% | ~80% |
| 3 | TF-IDF + Advanced features | ~80% | ~80% |
| 4 | Sentence Embeddings + Advanced features | ~82% | ~86% |

---

## Features

### Advanced 
- Length features: character/word counts, absolute differences
- Word overlap: common words, word share ratio
- Token ratios: cwc, csc, ctc (min/max), first/last word match
- Fuzzy scores: QRatio, partial ratio, token sort/set ratio

### Text Vector Representations
- **Bag of Words** (`CountVectorizer`, 3000 features)
- **TF-IDF** (`TfidfVectorizer`, 3000 features)
- **Sentence Embeddings** (`all-MiniLM-L6-v2` via sentence-transformers)
  - Cosine similarity
  - Absolute difference
  - Element-wise product

---

## Models
- **Random Forest** (`class_weight='balanced'`)
- **XGBoost** (`scale_pos_weight` for class imbalance)

---

## Dataset

**Quora Question Pairs**

Dataset link:  
https://www.kaggle.com/c/quora-question-pairs

- 100k sampled rows
- ~63% non-duplicate
- ~37% duplicate

---

## Key Learnings

- Hand-crafted features alone plateau at **~76% accuracy**
- **BoW** and **TF-IDF** both improve performance to **~80%**
- Sentence embeddings  push performance to **82–86%** by capturing semantic similarity

---
