# Credit Card Fraud Detection

Detecting fraudulent credit card transactions with a neural network, built as a hands-on project alongside the [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) (DeepLearning.AI + Stanford).

The focus is less on squeezing out the best score and more on doing each step the right way: splitting before preprocessing, choosing metrics that work with rare classes, diagnosing bias vs variance, and picking a decision threshold based on business cost.

## The problem

The [ULB Credit Card Fraud dataset](https://www.openml.org/d/42175) contains 284,807 transactions made by European cardholders over two days, of which only **492 are fraud (0.17%)**.

- `V1`..`V28`: PCA components (the original features are confidential).
- `Time`: seconds since the first transaction.
- `Amount`: transaction amount.
- `Class`: 1 = fraud, 0 = legit.

A model that never flags anything is 99.83% accurate. Accuracy is therefore useless here, and this project evaluates models on **precision, recall, F1 and PR-AUC** instead.

## Approach

1. **Split by time.** Train on the first 60% of transactions, validate on the next 20%, test on the last 20%. This mimics production (train on the past, predict the future) and is harsher than a random split. The test set is only used once, at the end.
2. **Preprocess with training statistics only.** `Amount` is log-transformed, `Time` becomes hour of day, and every feature is standardized with the mean and standard deviation of the training set.
3. **Baselines.** A lazy model that always predicts "legit", and logistic regression.
4. **Neural network** in Keras, with class weights to counter the imbalance. *(in progress)*
5. **Diagnosis.** Learning curves, J_train vs J_cv, regularization and error analysis. *(planned)*
6. **Cost-based threshold** and a FastAPI + Streamlit demo. *(planned)*

## Results (cv set)

| Model | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|
| Lazy (always "legit") | 0.000 | 0.000 | 0.000 | 0.001 |
| Logistic regression | 0.643 | 0.474 | 0.545 | 0.629 |
| Logistic regression (balanced weights) | 0.023 | 0.895 | 0.044 | 0.762 |
| Neural network | | | | |

Precision, recall and F1 use the default 0.5 threshold; PR-AUC is threshold-free. The cv set holds only 57 frauds, so each one moves recall by almost 2 points.

## Project structure

```
fraud-detection/
├── src/
│   ├── data.py        # loading and train/cv/test splits
│   ├── features.py    # feature engineering and train-only standardization
│   ├── train.py       # baselines and the neural network
│   └── evaluate.py    # metrics, confusion matrix, PR curves
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baselines.ipynb
│   └── 03_neural_network.ipynb
├── reports/figures/   # plots used in this README
└── tests/             # pipeline tests on synthetic data
```

## Running it

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

.venv/bin/python -m src.train   # downloads the data on first run, prints the baseline table
.venv/bin/pytest                # runs the tests
.venv/bin/jupyter lab           # opens the notebooks
```

## Limitations

- **Anonymized features.** `V1`..`V28` are PCA components, so error analysis cannot say *why* a fraud was missed in business terms.
- **Only two days of data.** Daily or weekly seasonality cannot be learned, and the cv and test sets are small (57 and 75 frauds).
- **Drift.** The fraud rate falls from 0.21% in train to 0.10% in cv. A model in production would need monitoring and periodic retraining.
