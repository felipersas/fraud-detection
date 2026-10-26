# Error analysis strategies

Training a model is a loop: choose an architecture, train it, diagnose it, and use the diagnosis to decide the next change. The first diagnostic is bias versus variance (learning curves, J_train vs J_cv), which tells you whether the model is too weak or fails to generalize. The second is **error analysis**, which tells you *which kinds of examples* the model gets wrong. Together they decide which idea deserves your time, out of the many that could be tried.

This document collects the error analysis strategies used in this project, starting from the recipe taught in the [Machine Learning Specialization](https://www.coursera.org/learn/advanced-learning-algorithms/lecture/FaPgS/error-analysis), and shows how each one played out on the fraud data. The code lives in [`src/error_analysis.py`](../src/error_analysis.py) and the analysis in [`notebooks/03_neural_network.ipynb`](../notebooks/03_neural_network.ipynb).

## Ground rules

- **Use the cv set, never the test set.** Every decision made by looking at errors adapts the model to the data you looked at. If that data is the test set, it stops being an honest estimate of performance on new data.
- **Fix the threshold first.** Which transactions count as errors depends on the decision threshold. Analyse at one threshold, and remember that the error map changes when the threshold does.
- **Look at both error types.** In fraud detection a missed fraud and a false alarm have different causes and different costs. Analyse each one separately.
- **Repeat after every change.** The error map moves each time the model changes. In this project, lowering the fraud weight from 400 to 10 turned a model that missed 25 frauds and raised about 25,000 false alarms into one that missed 402 and raised about 2,100, and the analysis had to be redone.

## Strategy 1: sample, tag, count

This is the course recipe, and the backbone of everything else.

1. Collect the cv examples the model got wrong. If there are many, **draw a random sample of about 100**. That is enough to estimate the share of each kind of error, and few enough to inspect carefully. Sample at random, never the first rows, which may all come from the same day or the same card.
2. **Tag each error** with the traits it shows. Tags may overlap: one error can be both "small purchase" and "daytime", so the counts can add up to more than 100.
3. **Count the tags and rank them.** Work on the largest groups first.
4. **Use the ceiling.** If a tag covers *k* of the 100 errors, fixing everything in that group removes at most *k*% of the errors. Before spending weeks on an idea, check its ceiling. In the course example, a detector for deliberate misspellings in spam turned out to address only a few percent of the errors.

With readable columns, tags can be written as rules (`amt < 100`, `hour in night_hours`) instead of being read one by one, which makes the analysis repeatable. `count_tags` in `src/error_analysis.py` does the counting.

**In this project**, on 100 sampled missed frauds:

| Tag | Missed frauds |
|---|---:|
| low-risk store category | 92 |
| card already in a burst (large purchase in the last 24 hours) | 92 |
| small purchase (under $100) | 72 |
| daytime (4h to 21h) | 23 |

And on 100 sampled false alarms:

| Tag | False alarms |
|---|---:|
| calm card history (no large purchase in the last 24 hours) | 85 |
| large purchase (over $200) | 80 |
| at night (22h to 3h) | 63 |
| high-risk store category | 63 |

## Strategy 2: compare against the right reference group

A trait shared by most errors is not necessarily useful. 92% of missed frauds are in low-risk store categories, but so are most legit purchases, so a model cannot use that trait to tell them apart. A trait is only actionable if it **differs between the errors and the examples they are confused with**:

- missed frauds should be compared with **caught frauds** (what makes these frauds harder?) and with **legit purchases** (why do they look legit?);
- false alarms should be compared with **legit purchases that were not flagged** (what makes these suspicious?) and with **caught frauds** (what do they share with fraud?).

**In this project**, recent card activity was the trait that split the groups sharply:

| Share with a large purchase on the same card in the previous 24 hours | |
|---|---:|
| missed frauds | 92% |
| caught frauds | 86% |
| false alarms | 16% |
| legit purchases, not flagged | 13% |

Missed frauds look like frauds on this trait, and false alarms look like legit purchases. A feature describing it can therefore fix both error types at once.

## Strategy 3: slice performance by readable features

Compute recall (for missed frauds) or the false alarm rate per slice of a readable feature: amount bucket, hour, store category, customer age. Slices where performance drops point to a pattern the model has not learned.

**In this project**, frauds under $100 were missed far more often than larger ones, and frauds during the day more often than at night. The slices also explained why logistic regression failed: fraud peaks at 22h-23h *and* 0h-3h, at both ends of the hour scale, which a single linear weight on `hour` cannot represent.

## Strategy 4: follow one entity over time

Statistics hide sequences. Pick an entity that produces many errors (a card, a customer, a device) and list its events in order, with the model's score next to each one.

**In this project**, a single card showed the whole story: two days of ordinary purchases, then a burst that opens with five large night-time purchases, all caught. As the burst goes on, the large purchases keep being caught, while the small purchases in low-risk stores mixed in between, and two large purchases made in the morning, are missed. The burst is obvious when reading the card's history, and invisible when each purchase is scored on its own.

## Strategy 5: look for information the model does not have

Some errors are not a modelling problem: the inputs simply do not contain the signal. Ask what a human investigator would check before deciding, and whether the model can see it.

**In this project**, an investigator would ask "is this normal for this card?". The model scores each transaction in isolation, so it cannot answer. Features built from the card's recent history (number of purchases, night purchases and large purchases in the previous 24 hours) supply that context. They must be computed **only from earlier transactions of the same card**, which is what a bank knows when a purchase arrives. Using later transactions, or past fraud labels that would not yet be known, leaks the answer into the features.

## Common pitfalls

- Analysing the test set instead of cv.
- Looking at five or ten errors: too few to estimate shares.
- Taking the first errors in the file instead of a random sample.
- Forcing each error into a single category.
- Choosing the most interesting idea instead of the one with the highest ceiling.
- Treating a trait common to the errors as actionable without checking it against the reference group.
- Forgetting that a new threshold or a new model changes which examples are errors.

## When error analysis works poorly

Error analysis relies on a person being able to look at an error and say why the model should have got it right. It works well when humans are good at the task, and poorly when they are not (predicting which ad someone will click, for instance).

It also needs **readable features**. This project started on the ULB credit card dataset, whose features are anonymized PCA components named `V1` to `V28`. The most an analysis could say there was "V14 looked normal", which suggests no next step. Switching to a dataset with readable columns made every finding above possible.
