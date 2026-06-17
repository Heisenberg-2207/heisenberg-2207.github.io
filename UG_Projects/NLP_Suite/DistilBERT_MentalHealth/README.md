# DistilBERT Mental Health Classification

> Independent project · IIT Madras · 2025

Fine-tuned DistilBERT for multi-class emotional state classification from mental health support forum text.

## Task

Classify social media / forum posts into **7 emotional states:**

| Class | Description |
|---|---|
| Normal | No mental health concern |
| Depression | Depressive episodes, hopelessness |
| Anxiety | Worry, panic, generalized anxiety |
| Suicidal Ideation | Thoughts of self-harm |
| Stress | Situational stress, overwhelm |
| Bipolar | Mood cycling indicators |
| Personality Disorder | BPD and related indicators |

## What's Here

```
DistilBERT_MentalHealth/
├── BERT.ipynb          ← Full training + evaluation notebook
└── Combined Data.csv   ← Merged dataset from multiple sources
```

## Approach

**Model:** `distilbert-base-uncased` (Hugging Face) with a classification head

**Why DistilBERT:**
- 40% fewer parameters than BERT-base
- 60% faster inference
- Retains 97% of BERT's language understanding
- Practical for deployment in resource-constrained health applications

**Class Imbalance:** Addressed via weighted cross-entropy loss. The dataset is heavily skewed toward "normal" — naive training would ignore minority classes entirely.

**Training:**
```
Optimizer: AdamW (lr=2e-5)
Epochs: 4
Batch size: 16
Max sequence length: 128
```

## Key Design Decisions

- Combined multiple open-source mental health Reddit datasets
- Removed duplicate and near-duplicate posts
- Applied weighted loss rather than resampling to preserve data distribution
- Evaluated per-class F1 to identify underperforming categories
