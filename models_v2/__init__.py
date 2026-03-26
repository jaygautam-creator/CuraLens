# models_v2 package
# CuraLens v2/v3 multi-modal model architectures
#
# Exports:
#   multimodal_model  — legacy oral model (4D metadata, backward-compat)
#   oral_model        — oral cancer v3 model (6D clinical metadata)
#   skin_model        — skin cancer model    (6D clinical metadata)

from .multimodal_model import build_multimodal_model, load_model, save_model
from .oral_model import build_oral_model, load_oral_model, save_oral_model
from .skin_model import build_skin_model, load_skin_model, save_skin_model

__all__ = [
    "build_multimodal_model", "load_model", "save_model",
    "build_oral_model",       "load_oral_model", "save_oral_model",
    "build_skin_model",       "load_skin_model", "save_skin_model",
]
