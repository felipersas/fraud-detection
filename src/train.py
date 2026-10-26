"""Model definitions: two baselines and the neural network.

Run the baselines on the cv set:

    python -m src.train
"""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src.data import load_raw, split_by_time
from src.evaluate import confusion, evaluate, results_table
from src.features import Preprocessor

import tensorflow as tf


def train_lazy(X: np.ndarray, y: np.ndarray) -> DummyClassifier:
    """Lazy baseline that always predicts the majority class ("not fraud").

    It exists to show why accuracy is misleading: it scores ~99.8% accuracy
    while catching zero frauds.
    """
    return DummyClassifier(strategy="prior").fit(X, y)


def train_logistic(X: np.ndarray, y: np.ndarray, lambda_: float = 0.0, class_weight=None) -> LogisticRegression:
    """Logistic regression, the performance baseline every other model must beat.

    scikit-learn expresses L2 regularization as C = 1/λ, so a large C means
    weak regularization; lambda_=0 disables it entirely.
    class_weight={0: 1, 1: w} multiplies the cost of misclassifying a fraud by w.
    """
    C = 1.0 / lambda_ if lambda_ > 0 else np.inf
    model = LogisticRegression(C=C, class_weight=class_weight, max_iter=1000)
    return model.fit(X, y)


def build_neural_net(n_features: int, learning_rate: float = 1e-4):
    """Build and compile a small Keras network for binary classification.

    TODO: implement. A good starting point:
    - tf.keras.Sequential, starting with tf.keras.Input(shape=(n_features,))
    - two small Dense hidden layers with activation="relu" (e.g. 16 and 8 units)
    - a Dense(1, activation="sigmoid") output, so the network returns P(fraud)
    - compile with Adam(learning_rate=1e-3), BinaryCrossentropy() as the loss and
      tf.keras.metrics.AUC(curve="PR", name="pr_auc") as the tracked metric

    When fitting, pass class_weight={0: 1, 1: w} to up-weight frauds and
    validation_data=(X_cv, y_cv) to track J_train vs J_cv per epoch.
    For L2 regularization (λ), add kernel_regularizer=tf.keras.regularizers.L2(λ)
    to the Dense layers.
    """
    reg = tf.keras.regularizers.L2(0.001)  # Example regularization strength
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(n_features,)),
        tf.keras.layers.Dense(8, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dense(4, activation="relu", kernel_regularizer=reg),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc")]
    )
    return model


def train_neural_net(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cv: np.ndarray,
    y_cv: np.ndarray,
    fraud_weight: float,
    learning_rate: float = 1e-4,
    max_epochs: int = 100,
    patience: int | None = None,
    seed: int = 0,
    batch_size: int = 512,
    verbose: int = 0,
):
    """Build and train the network, returning (model, history).

    The cv set is passed as validation data so J_cv is measured after every
    epoch. It gets the same fraud weight as train, otherwise J_train (weighted
    by Keras) and J_cv (unweighted) would sit on different scales.

    With `patience`, training uses early stopping: it stops once J_cv has not
    improved for `patience` epochs in a row and rolls back to the weights of the
    best epoch. Without it, training runs for exactly `max_epochs`.
    """
    tf.keras.utils.set_random_seed(seed)  # seeds Python, NumPy and TensorFlow together
    model = build_neural_net(n_features=X_train.shape[1], learning_rate=learning_rate)
    callbacks = []
    if patience is not None:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
        )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_cv, y_cv, np.where(y_cv == 1, fraud_weight, 1.0)),
        class_weight={0: 1, 1: fraud_weight},
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )
    return model, history


def main():
    split = split_by_time(load_raw())
    print(split.summary(), "\n")

    prep = Preprocessor().fit(split.train)
    X_train, y_train = prep.transform(split.train)
    X_cv, y_cv = prep.transform(split.cv)

    models = {
        "lazy": train_lazy(X_train, y_train),
        "logistic": train_logistic(X_train, y_train),
    }
    results = {}
    for name, model in models.items():
        y_prob = model.predict_proba(X_cv)[:, 1]
        results[name] = evaluate(y_cv, y_prob)
        print(f"{name}\n{confusion(y_cv, y_prob)}\n")

    print(results_table(results))


if __name__ == "__main__":
    main()
