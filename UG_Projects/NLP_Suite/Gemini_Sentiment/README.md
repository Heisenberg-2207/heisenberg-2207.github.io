# Gemini Sentiment Classification

> Independent project · IIT Madras · 2025

Prompt-engineered sentiment classification pipeline using Google Gemini, benchmarked against the DistilBERT mental health classifier for contextual analysis.

## What's Here

```
Gemini_Sentiment/
├── GEMINI.ipynb        ← Full experiment notebook
└── amazon_alexa.tsv    ← Amazon Alexa review dataset (3,150 reviews)
```

## Task

Classify Amazon Alexa product reviews into sentiment categories:
- **Positive** — satisfied users
- **Negative** — dissatisfied users  
- **Mixed** — ambivalent feedback
- **Irrelevant** — off-topic content

## Approach

**Zero-shot prompting:**
```
Classify the sentiment of this review as positive, negative, mixed, or irrelevant.
Review: "Alexa is great for timers but terrible at recognizing accents."
```

**Few-shot prompting (3-shot):**
Provide 3 labeled examples per class before the target review. Dramatically improves performance on edge cases.

**Chain-of-thought:**
Asked the model to reason step-by-step before outputting a label. Improved accuracy on mixed-sentiment cases.

## Key Finding

Well-crafted 3-shot prompts with Gemini matched fine-tuned DistilBERT accuracy on this dataset — without any gradient updates or labeled training data.

This demonstrates the practical utility of large LLMs for low-resource classification tasks where annotation is expensive.

## Integration

Output was integrated into the DistilBERT mental health pipeline:
- Gemini pre-filters and labels incoming text by sentiment valence
- DistilBERT performs the fine-grained emotional state classification
- Combined pipeline handles both general sentiment and clinical categorization

## Technologies

`Python` · `Google Gemini API` · `Pandas` · `scikit-learn`
