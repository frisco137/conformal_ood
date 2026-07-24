import sys
from pathlib import Path

# Add models/ directory to sys.path to ensure local tabicl imports resolve to models/tabicl
models_dir = str(Path(__file__).parent.resolve())
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

from .tabpfn_v3 import get_classifier as get_tabpfn_v3_classifier
from .tabpfn_v3 import get_regressor as get_tabpfn_v3_regressor

from .tabpfn_v2 import get_classifier as get_tabpfn_v2_classifier
from .tabpfn_v2 import get_regressor as get_tabpfn_v2_regressor

from .tabicl_v2 import get_classifier as get_tabicl_v2_classifier
from .tabicl_v2 import get_regressor as get_tabicl_v2_regressor

from .google_tabfm import get_classifier as get_google_tabfm_classifier
from .google_tabfm import get_regressor as get_google_tabfm_regressor

__all__ = [
    "get_tabpfn_v3_classifier",
    "get_tabpfn_v3_regressor",
    "get_tabpfn_v2_classifier",
    "get_tabpfn_v2_regressor",
    "get_tabicl_v2_classifier",
    "get_tabicl_v2_regressor",
    "get_google_tabfm_classifier",
    "get_google_tabfm_regressor",
]
