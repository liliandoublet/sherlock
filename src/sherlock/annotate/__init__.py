from sherlock.annotate.irony import annotate_irony
from sherlock.annotate.pipeline import annotate, save_annotated
from sherlock.annotate.sentiment import annotate_sentiment

__all__ = [
    "annotate",
    "annotate_irony",
    "annotate_sentiment",
    "save_annotated",
]
