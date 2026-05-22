from sherlock.annotate.sentiment import annotate_sentiment
from sherlock.annotate.irony import annotate_irony
from sherlock.annotate.pipeline import annotate, save_annotated

__all__ = [
    "annotate_sentiment",
    "annotate_irony",
    "annotate",
    "save_annotated",
]
