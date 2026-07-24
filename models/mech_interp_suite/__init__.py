"""Mechanistic Interpretability Suite for Tabular Foundation Models.

Provides hook-based activation extraction, architecture documentation,
and sanity checks for TabPFN v3, TabPFN v2, TabICL v2, and Google TabFM.
"""

from .utils import HookManager, generate_gaussian_data, print_activation_summary
