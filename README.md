# Credit Card Fraud Detection

Detecting fraudulent credit card transactions with a neural network, built as a hands-on project alongside the [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) (DeepLearning.AI + Stanford).

The focus is less on squeezing out the best score and more on doing each step the right way: splitting before preprocessing, choosing metrics that work with rare classes, diagnosing bias vs variance, analysing errors, and picking a decision threshold based on business cost.

## The problem

The [Sparkov dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) contains 1.85 million simulated card transactions made by 999 fictional customers at 693 merchants between January 2019 and December 2020. Only **9,651 are fraud (0.52%)**.

Unlike datasets with anonymized features, every column is readable: purchase amount, store category, time, cardholder age and location, store location, and the card number. That makes it possible to explain *why* the model gets a transaction wrong.

A model that never flags anything is 99.5% accurate. Accuracy is therefore useless here, and this project evaluates models on **precision, recall, F1 and PR-AUC** instead.

## What the data shows

- Fraudulent purchases are much larger (median 390 vs 47).
- Fraud concentrates at night: from 22h to 3h the fraud rate is 10 to 25 times higher than during the day.
- Online shopping and groceries carry about 10 times the fraud rate of restaurants.
- Fraud arrives in bursts: a compromised card suffers about 10 fraudulent purchases within roughly two days.

## Approach

1. **Split by time.** Train on 2019, validate (cv) on January to June 2020, and test on Kaggle's own test file (June to December 2020). This mimics production: train on the past, predict the future. The test set is only used once, at the end.
2. **Preprocess with training statistics only.** Features are built per transaction (log amount, hour, day of week, age, city size, home-to-store distance, store category, gender). Numeric features are standardized and categories one-hot encoded, both fitted on the training set.
3. **Baselines.** A lazy model that always predicts "legit", and logistic regression with and without class weights.
4. **Neural network** in Keras, with class weights to counter the imbalance. *(in progress)*
5. **Diagnosis and error analysis.** Learning curves, J_train vs J_cv, and a look at which frauds are missed and which legit purchases are flagged. *(in progress)*
6. **Cost-based threshold** and a FastAPI + Streamlit demo. *(planned)*

## Results (cv set)

| Model | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|
| Lazy (always "legit") | 0.000 | 0.000 | 0.000 | 0.006 |
| Logistic regression | 0.706 | 0.084 | 0.150 | 0.351 |
| Logistic regression (balanced weights) | 0.018 | 0.743 | 0.035 | 0.165 |
| Neural network | | | | |

Precision, recall and F1 use the default 0.5 threshold; PR-AUC is threshold-free.

Logistic regression is weak on this data because the fraud patterns are not linear. The risky hours (22h to 3h) sit at both ends of the 0-23 scale, so a single "hour" weight cannot capture them, and class weights cannot fix that.

## Project structure

```
fraud-detection/
├── src/
│   ├── data.py        # loading and train/cv/test splits
│   ├── features.py    # feature engineering, train-only scaling and encoding
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

# Needs a Kaggle API token (kaggle.com > Settings > API)
.venv/bin/kaggle datasets download -d kartik2112/fraud-detection -p data/sparkov --unzip

.venv/bin/python -m src.train   # prints the baseline table
.venv/bin/pytest                # runs the tests
.venv/bin/jupyter lab           # opens the notebooks
```

## Limitations

- **Simulated data.** The transactions come from a simulator, not a real bank. Patterns are cleaner and more regular than in real life (for example, every compromised card follows the same two-day burst), so scores here would not transfer to real data.
- **Some features carry no signal by construction.** The simulator places stores at random near the cardholder, so home-to-store distance does not help, even though it often does with real data.
- **Drift.** The monthly fraud rate moves between about 0.2% and 1%, so a model in production would need monitoring and periodic retraining.
- **Each transaction is scored on its own.** The model does not yet know what is normal for each card, which is the most promising direction for improvement.
