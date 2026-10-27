"""Helpers for error analysis on the cv set.

Error analysis follows the course recipe: take the cv examples the model got
wrong, draw a random sample of about 100, tag each one with the traits it
shows (a single example can carry several tags), count the tags, and work on
the biggest groups first. Fixing every example of a group can remove at most
that group's share of the errors, which puts a ceiling on each idea.

The card-history counts used to tag errors come from fraud_core.history, the
same code the card-history model uses as features.
"""

import pandas as pd

# Re-exported so analysis code finds every helper in one place.
from fraud_core.history import NIGHT_HOURS, card_history  # noqa: F401


def sample_errors(errors: pd.DataFrame, n: int = 100, seed: int = 0) -> pd.DataFrame:
    """Draw a random sample of errors, so the analysis is not biased toward the first rows."""
    return errors.sample(n=min(n, len(errors)), random_state=seed)


def count_tags(sample: pd.DataFrame, tags: dict[str, pd.Series]) -> pd.DataFrame:
    """Count how many sampled errors carry each tag, largest group first.

    `tags` maps a tag name to a boolean Series over the sample. Tags may overlap,
    so the shares can add up to more than 100%.
    """
    counts = pd.Series({name: int(mask.loc[sample.index].sum()) for name, mask in tags.items()})
    return pd.DataFrame({"errors": counts, "share_%": (100 * counts / len(sample)).round(1)}).sort_values(
        "errors", ascending=False
    )
