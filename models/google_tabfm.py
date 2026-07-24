import sys
import logging
from pathlib import Path

# Inject parent of models/ directory into sys.path to resolve models package correctly
parent_dir = str(Path(__file__).parent.parent.resolve())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.tabfm import TabFMClassifier, TabFMRegressor
from models.tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0

logger = logging.getLogger(__name__)

def get_classifier(device="cpu", **kwargs):
    """
    Returns an initialized Google TabFM Classifier using PyTorch backend.
    """
    logger.info("Initializing Google TabFM Classifier...")
    model = tabfm_v1_0_0.load("classification", device=device)
    return TabFMClassifier(model=model, **kwargs)

def get_regressor(device="cpu", **kwargs):
    """
    Returns an initialized Google TabFM Regressor using PyTorch backend.
    """
    logger.info("Initializing Google TabFM Regressor...")
    model = tabfm_v1_0_0.load("regression", device=device)
    return TabFMRegressor(model=model, **kwargs)
