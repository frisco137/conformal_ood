import sys
import logging
from pathlib import Path

# Inject models/ directory into sys.path to resolve local tabicl package if needed
models_dir = str(Path(__file__).parent.resolve())
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

from models.tabicl import TabICLClassifier, TabICLRegressor

logger = logging.getLogger(__name__)

def get_classifier(device="cpu", **kwargs):
    """
    Returns an initialized TabICL v2 Classifier.
    """
    logger.info("Initializing TabICL v2 Classifier...")
    # Map 'cpu' or 'cuda' device specifications
    return TabICLClassifier(device=device, **kwargs)

def get_regressor(device="cpu", **kwargs):
    """
    Returns an initialized TabICL v2 Regressor.
    """
    logger.info("Initializing TabICL v2 Regressor...")
    return TabICLRegressor(device=device, **kwargs)
