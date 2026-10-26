"""Helpers for error analysis on the cv set.

Error analysis follows the course recipe: take the cv examples the model got
wrong, draw a random sample of about 100, tag each one with the traits it
shows (a single example can carry several tags), count the tags, and work on
the biggest groups first. Fixing every example of a group can remove at most
that group's share of the errors, which puts a ceiling on each idea.
"""

import numpy as np
import pandas as pd

from src.data import TARGET, TIME

NIGHT_HOURS = [22, 23, 0, 1, 2, 3]  # the high-risk hours found in the EDA


def sample_errors(errors: pd.DataFrame, n: int = 100, seed: int = 0) -> pd.DataFrame:
    """Draw a random sample of errors, so the analysis is not biased toward the first rows."""
    return errors.sample(n=min(n, len(errors)), random_state=seed)


def count_tags(sample: pd.DataFrame, tags: dict[str, pd.Series]) -> pd.DataFrame:
    """Count how many sampled errors carry each tag, largest group first.

    `tags` maps a tag name to a boolean Series over the sample. Tags may overlap,
    so the shares can add up to more than 100%.
    """
    counts = pd.Series({name: int(mask.loc[sample.index].sum()) for name, mask in tags.items()})
    return (
        pd.DataFrame({"errors": counts, "share_%": (100 * counts / len(sample)).round(1)})
        .sort_values("errors", ascending=False)
    )


def recent_card_activity(df: pd.DataFrame, window: str = "24h", large_amount: float = 200) -> pd.DataFrame:
    """Describe what each card did in the `window` before every transaction.

    Only earlier transactions of the same card are counted, never the current
    one or anything after it, which is exactly what a bank knows at the moment
    a purchase arrives. Pass every available row (train and cv together) so the
    first cv days still see the card's history from the end of train.

    Returns, aligned with `df.index`:
    - tx_last_24h: purchases on the card in the window
    - night_tx_last_24h: of those, how many happened between 22h and 3h
    - large_tx_last_24h: of those, how many were above `large_amount`
    """
    events = pd.DataFrame(
        {
            "cc_num": df["cc_num"],
            "time": df[TIME],
            "tx": 1,
            "night_tx": df[TIME].dt.hour.isin(NIGHT_HOURS).astype(int),
            "large_tx": (df["amt"] > large_amount).astype(int),
        },
        index=df.index,
    ).sort_values(["cc_num", "time"], kind="stable")

    counts = ["tx", "night_tx", "large_tx"]
    # A time-based rolling window per card; the sums include the current row,
    # so subtracting it leaves only the card's earlier activity.
    rolled = events.set_index("time").groupby("cc_num", sort=True)[counts].rolling(window).sum()
    past = rolled.to_numpy() - events[counts].to_numpy()

    out = pd.DataFrame(past, index=events.index, columns=[f"{c}_last_{window}" for c in counts])
    return out.astype(int).reindex(df.index)
