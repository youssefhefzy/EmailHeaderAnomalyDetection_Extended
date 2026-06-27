"""
feature_groups.py
-----------------
Definitions for feature groups G1–G8 used in the experiments.

Each group is a named subset of the full 94-feature space.  The function
:func:`get_feature_subset` returns the list of column names present in a
given DataFrame that belong to the requested group.

Groups
------
G1 : Presence/absence of key headers (binary flags)
G2 : Domain-matching features
G3 : Routing anomalies (Received-chain features)
G4 : Structural irregularities
G5 : Authentication results (SPF / DKIM / DMARC)
G6 : Top-30 features from permutation importance (original paper baseline)
G7 : G2 + G3 + G5 (routing + authentication + domain matching)
G8 : Top-5 features from G6 (fastest inference)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature-group definitions (column name prefixes / explicit lists)
# ---------------------------------------------------------------------------

# G1 — header presence flags
_G1_PREFIXES = ("has_",)

# G2 — domain matching
_G2_COLS = [
    "from_domain_len",
    "from_eq_reply_to_domain",
    "from_eq_return_path_domain",
    "from_eq_mid_domain",
    "from_eq_first_received_domain",
    "reply_to_present_and_differs",
    "return_path_present_and_differs",
    "to_domain_eq_from_domain",
    "mid_domain_eq_from_domain",
    "envelope_to_eq_from_domain",
    "reply_to_subdomain_of_from",
]

# G3 — routing anomalies
_G3_COLS = [
    "n_received_hops",
    "missing_received",
    "malformed_received_hops",
    "n_unique_hop_ips",
    "n_private_ips_in_received",
    "tz_mismatch_in_received",
    "n_unique_tz_offsets",
    "has_unusual_tz",
    "n_hop_domains",
    "date_vs_received_anomaly",
    "excessive_hops",
]

# G4 — structural irregularities
_G4_COLS = [
    "duplicate_headers",
    "subject_non_ascii",
    "from_non_ascii",
    "malformed_date",
    "subject_len",
    "subject_has_re",
    "subject_all_caps",
    "subject_exclaim_count",
    "subject_question_count",
    "has_mime_version",
    "is_base64_encoded",
    "is_quoted_printable",
    "mid_malformed",
    "mid_len",
    "is_multipart",
    "is_html_only",
    "total_header_count",
]

# G5 — authentication
_G5_COLS = [
    "spf_result",
    "dkim_result",
    "dmarc_result",
    "dkim_signature_present",
    "dkim_signature_valid_format",
    "auth_results_present",
    "all_auth_pass",
    "any_auth_fail",
]

# G6 — top-30 features from the original paper (permutation importance order).
# These are placeholders based on Table VI of Beaman & Isah (2021); they are
# overridden at runtime if permutation-importance scores are available.
_G6_TOP30 = [
    # Domain / identity (most important in original paper)
    "from_eq_return_path_domain",
    "from_eq_reply_to_domain",
    "reply_to_present_and_differs",
    "return_path_present_and_differs",
    "from_eq_mid_domain",
    "mid_malformed",
    "mid_domain_eq_from_domain",
    # Routing
    "n_received_hops",
    "missing_received",
    "malformed_received_hops",
    "tz_mismatch_in_received",
    "date_vs_received_anomaly",
    "n_unique_hop_ips",
    "has_unusual_tz",
    # Authentication
    "spf_result",
    "dkim_result",
    "dmarc_result",
    "any_auth_fail",
    "dkim_signature_present",
    "auth_results_present",
    # Structural
    "duplicate_headers",
    "subject_non_ascii",
    "malformed_date",
    "is_html_only",
    "total_header_count",
    # Presence flags
    "has_dkim_signature",
    "has_authentication_results",
    "has_reply_to",
    "has_return_path",
    "has_message_id",
]

# G7 = G2 + G3 + G5
_G7_COLS = _G2_COLS + _G3_COLS + _G5_COLS

# G8 — top-5 fastest-inference features (subset of G6)
_G8_TOP5 = [
    "from_eq_return_path_domain",
    "from_eq_reply_to_domain",
    "reply_to_present_and_differs",
    "n_received_hops",
    "spf_result",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_feature_subset(
    df: pd.DataFrame,
    group_name: str,
    importance_scores: Optional[Dict[str, float]] = None,
) -> List[str]:
    """
    Return the list of column names in *df* that belong to the named group.

    Parameters
    ----------
    df:
        The full feature DataFrame (must contain a ``label`` column).
    group_name:
        One of ``'G1'``, ``'G2'``, …, ``'G8'``.
    importance_scores:
        Optional dict mapping feature name → permutation importance score.
        If provided, G6 and G8 are built from these scores rather than the
        hard-coded defaults.

    Returns
    -------
    list[str]
        Column names available in *df* that belong to the group.
        Empty list if the group has no valid columns.

    Raises
    ------
    ValueError
        If *group_name* is not one of G1–G8.
    """
    group_name = group_name.upper()
    label_col = "label"
    available = [c for c in df.columns if c != label_col]

    if group_name == "G1":
        cols = _cols_by_prefixes(available, _G1_PREFIXES)

    elif group_name == "G2":
        cols = _intersect(available, _G2_COLS)

    elif group_name == "G3":
        cols = _intersect(available, _G3_COLS)

    elif group_name == "G4":
        cols = _intersect(available, _G4_COLS)

    elif group_name == "G5":
        cols = _intersect(available, _G5_COLS)

    elif group_name == "G6":
        if importance_scores:
            sorted_feats = sorted(
                importance_scores, key=importance_scores.get, reverse=True
            )
            top30 = sorted_feats[:30]
        else:
            top30 = _G6_TOP30
        cols = _intersect(available, top30)

    elif group_name == "G7":
        cols = _intersect(available, _G7_COLS)

    elif group_name == "G8":
        if importance_scores:
            sorted_feats = sorted(
                importance_scores, key=importance_scores.get, reverse=True
            )
            top5 = sorted_feats[:5]
        else:
            top5 = _G8_TOP5
        cols = _intersect(available, top5)

    else:
        raise ValueError(
            f"Unknown feature group '{group_name}'. "
            "Valid groups: G1, G2, G3, G4, G5, G6, G7, G8."
        )

    if not cols:
        logger.warning(
            "Feature group %s has no valid columns in the DataFrame. "
            "Check that the dataset was built with extract_headers.py.",
            group_name,
        )
    else:
        logger.debug("Group %s → %d features: %s", group_name, len(cols), cols)

    return cols


def get_all_group_names() -> List[str]:
    """Return the list of all supported feature group names."""
    return ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"]


def describe_groups() -> Dict[str, str]:
    """Return a short description of each group."""
    return {
        "G1": "Presence/absence of key headers (binary flags)",
        "G2": "Domain-matching fields (From ↔ Reply-To ↔ Return-Path ↔ MsgID)",
        "G3": "Routing anomalies (Received-chain hops, timezone mismatches)",
        "G4": "Structural irregularities (duplicates, non-ASCII, malformed Date)",
        "G5": "Authentication results (SPF / DKIM / DMARC pass/fail)",
        "G6": "Top-30 features from permutation importance (original paper baseline)",
        "G7": "Combined: G2 + G3 + G5 (routing + authentication + domain matching)",
        "G8": "Top-5 features from G6 (fastest-inference subset)",
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _cols_by_prefixes(available: List[str], prefixes: tuple) -> List[str]:
    """Return columns that start with any of the given prefixes."""
    return [c for c in available if any(c.startswith(p) for p in prefixes)]


def _intersect(available: List[str], wanted: List[str]) -> List[str]:
    """Return the subset of *wanted* that is present in *available*, preserving order."""
    avail_set = set(available)
    return [c for c in wanted if c in avail_set]
