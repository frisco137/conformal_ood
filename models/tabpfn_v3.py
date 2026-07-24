import logging
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.constants import ModelVersion

logger = logging.getLogger(__name__)

def get_classifier(device="cpu", **kwargs):
    """
    Returns an initialized TabPFN v3 Classifier.
    """
    logger.info("Initializing TabPFN v3 Classifier...")
    return TabPFNClassifier.create_default_for_version(ModelVersion.V3, device=device, **kwargs)

def get_regressor(device="cpu", **kwargs):
    """
    Returns an initialized TabPFN v3 Regressor.
    """
    logger.info("Initializing TabPFN v3 Regressor...")
    return TabPFNRegressor.create_default_for_version(ModelVersion.V3, device=device, **kwargs)
