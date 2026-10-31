# AI-ML-mini-project-
Drug sentiment review analysis using NLP classifiers and linear regression 

# Sentiment Classification of Patient Drug Reviews using NLP and Machine Learning

Mini-project for the **AI and ML in Healthcare** course.

**Author:** Mohammed Zaid Shaikh
**Institution:** Rizvi College of Engineering, Electronics and Computer Science Engineering

## Overview

Patients write thousands of online reviews about the medicines they take. These reviews contain useful information about effectiveness and side effects, but reading them manually does not scale. This project builds an NLP classifier that automatically labels a drug review as **negative**, **neutral** or **positive**.

## Dataset

- **Name:** Drug Review Dataset (Drugs.com), UCI ML Repository / Kaggle
- **Files:** `drugsComTrain_raw.tsv` (~161k reviews) and `drugsComTest_raw.tsv` (~54k reviews)
- **Columns used:** `review` (free text) and `rating` (1-10)
- The dataset files are not included in this repo because of their size. Download them from Kaggle by searching "Drug Review Dataset (Drugs.com)".

**Label mapping**

| Rating | Label |
|---|---|
| 1 - 4 | negative |
| 5 - 6 | neutral |
| 7 - 10 | positive |

**Class distribution (training set)**

| Class | Reviews |
|---|---|
| positive | 106,866 |
| negative | 40,075 |
| neutral | 14,356 |

The data is imbalanced: positive is the majority class and neutral is the smallest.

## Methodology

1. **Load data** from the train and test TSV files.
2. **Create labels** from ratings (mapping above).
3. **Clean text:** lowercase, remove HTML tags and the `&#039;` apostrophe artifact, keep letters only, remove stopwords (but keep "not", "no", "nor" since negations change sentiment).
4. **TF-IDF vectorization:** unigrams and bigrams, 50,000 features, `min_df=5`. The vocabulary is fit on the training data only to avoid data leakage.
5. **Model:** Logistic Regression with `class_weight="balanced"` to handle the class imbalance.
6. **Evaluation:** accuracy, precision, recall, F1-score and a confusion matrix on the test set.

## Tech Stack

Python 3, Google Colab, pandas, NLTK, scikit-learn, matplotlib

## Results

**Overall accuracy: 79.5%** on 53,766 test reviews.

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| negative | 0.74 | 0.79 | 0.76 | 13,497 |
| neutral | 0.37 | 0.64 | 0.47 | 4,829 |
| positive | 0.94 | 0.82 | 0.87 | 35,440 |
| **macro avg** | 0.68 | 0.75 | 0.70 | 53,766 |
| **weighted avg** | 0.84 | 0.79 | 0.81 | 53,766 |

**Classification report**

![Classification Report](classification_report.png)

**Confusion matrix**

![Confusion Matrix](confusion_matrix.png)

## Discussion

- **Positive** reviews are classified most reliably (precision 0.94, F1 0.87) because the wording is distinct and there are plenty of training samples.
- **Negative** reviews also perform well (F1 0.76).
- **Neutral** is the hardest class (F1 0.47). Mixed opinions overlap in vocabulary with both other classes, and neutral has the fewest samples. Class weighting raised neutral recall to 0.64, but at the cost of low precision (0.37): 3,579 positive reviews were predicted as neutral.
- **Limitations:** TF-IDF ignores word order and context (for example sarcasm), and a numeric rating is only a noisy proxy for the sentiment in the text.

## How to Run

1. Download the two dataset files from Kaggle.
2. Open `drug_review_sentiment.ipynb` in Google Colab (or Jupyter).
3. Upload `drugsComTrain_raw.tsv` and `drugsComTest_raw.tsv` to the same location as the notebook.
4. Run all cells (**Runtime → Run all**).

## Future Scope

- Fine-tune transformer models (DistilBERT, BioBERT, ClinicalBERT) and compare with this baseline.
- Add a regression model to predict the exact 1-10 rating.
- Use LIME or SHAP to explain which words drive each prediction.
- Extract side-effect mentions from reviews.
- Build a Streamlit web demo.

## References

1. Gräßer, F., Kallumadi, S., Malberg, H., Zaunseder, S. (2018). *Aspect-Based Sentiment Analysis of Drug Reviews Applying Cross-Domain and Cross-Data Learning.* Proc. International Conference on Digital Health (DH '18).
2. UCI Machine Learning Repository, *Drug Review Dataset (Drugs.com)*.
3. Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR, 12, 2825-2830.
4. Bird, Klein, Loper (2009). *Natural Language Processing with Python.* O'Reilly Media.
