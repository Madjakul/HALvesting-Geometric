# halph/models/__init__.py

from halph.models.modeling_link_prediction import LinkPrediction
from halph.models.modeling_lm_link_prediction import \
    LanguageModelLinkPrediction

__all__ = ["LinkPrediction", "LanguageModelLinkPrediction"]
