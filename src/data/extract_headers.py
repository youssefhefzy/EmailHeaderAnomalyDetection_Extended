"""
extract_headers.py
------------------
Parse raw email files (RFC-2822 / mbox) and extract the 94 header features
described in Table I of Beaman & Isah (2021).

Parsing logic for Received-header chains is adapted from:
    https://github.com/kregg34/EmailHeaderAnomalyDetection
    (MIT licence, Craig Beaman 2021)

Usage
-----
    from src.data.extract_headers import EmailHeaderExtractor
    extractor = EmailHeaderExtractor()
    features = extractor.extract_from_file("path/to/email.eml")
"""

from __future__ import annotations

import email
import email.policy
import hashlib
import logging
import re
import unicodedata
from datetime import datetime, timezone
from email import message_from_bytes, message_from_string
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / regex
# ---------------------------------------------------------------------------

_RE_IP = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)
_RE_DOMAIN = re.compile(
    r"(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)"
    r"+[a-zA-Z]{2,}"
)
_RE_EMAIL_ADDR = re.compile(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}")
_RE_TZ = re.compile(r"[+-]\d{4}")
_RE_RECEIVED_FROM = re.compile(
    r"from\s+(\S+).*?by\s+(\S+)", re.IGNORECASE | re.DOTALL
)

# Known legitimate timezone offsets (UTC ± integer hours)
_COMMON_TZ_OFFSETS = {f"+{h:02d}00" for h in range(0, 15)} | \
                     {f"-{h:02d}00" for h in range(0, 15)}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _safe_lower(value: Optional[str]) -> str:
    """Return lowercase string or empty string if None."""
    return (value or "").lower().strip()


def _extract_domain(addr: str) -> str:
    """Extract domain from an email address or URL string."""
    addr = _safe_lower(addr)
    if "@" in addr:
        return addr.split("@", 1)[-1].strip(">").strip()
    m = _RE_DOMAIN.search(addr)
    return m.group(0) if m else ""


def _normalize_domain(domain: str) -> str:
    """Strip www. prefix and trailing dots."""
    d = domain.lower().strip(". ")
    if d.startswith("www."):
        d = d[4:]
    return d


def _has_non_ascii(text: str) -> int:
    """Return 1 if the string contains non-ASCII characters, else 0."""
    return int(any(ord(c) > 127 for c in (text or "")))


def _count_unicode_categories(text: str, category_prefix: str) -> int:
    """Count characters matching a Unicode category prefix (e.g. 'C' for control)."""
    return sum(1 for c in (text or "")
               if unicodedata.category(c).startswith(category_prefix))


def _parse_date(date_str: str) -> Optional[datetime]:
    """Parse RFC-2822 date string; return None on failure."""
    if not date_str:
        return None
    from email.utils import parsedate_to_datetime
    try:
        return parsedate_to_datetime(date_str)
    except Exception:
        return None


def _extract_tz_offset(date_str: str) -> Optional[str]:
    """Extract timezone offset string (+0530, -0700, etc.) from a Date header."""
    m = _RE_TZ.search(date_str or "")
    return m.group(0) if m else None


# ---------------------------------------------------------------------------
# Core extractor
# ---------------------------------------------------------------------------

class EmailHeaderExtractor:
    """
    Extract the 94 email-header features used by Beaman & Isah (2021).

    Each call to :meth:`extract_from_message` / :meth:`extract_from_file`
    returns a flat dict of feature_name → value, suitable for building a
    pandas DataFrame row.
    """

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract_from_file(self, path: Path | str) -> Dict[str, Any]:
        """
        Parse an email file and return its header features.

        Parameters
        ----------
        path:
            Path to the raw email file (.eml or plain RFC-2822 text).

        Returns
        -------
        dict
            Feature dictionary (94 keys + optional metadata).
        """
        path = Path(path)
        try:
            raw = path.read_bytes()
            msg = message_from_bytes(raw, policy=email.policy.compat32)
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", path, exc)
            return self._empty_features()
        feats = self.extract_from_message(msg)
        feats["_source_file"] = str(path)
        return feats

    def extract_from_string(self, raw: str) -> Dict[str, Any]:
        """Parse an email from a raw string."""
        try:
            msg = message_from_string(raw, policy=email.policy.compat32)
        except Exception as exc:
            logger.warning("Failed to parse string email: %s", exc)
            return self._empty_features()
        return self.extract_from_message(msg)

    def extract_from_message(self, msg: email.message.Message) -> Dict[str, Any]:
        """
        Extract features from a parsed :class:`email.message.Message`.

        Returns
        -------
        dict
            Flat feature dict.
        """
        f: Dict[str, Any] = {}

        # ── Presence / absence of key headers (G1) ──────────────────────
        f.update(self._header_presence_features(msg))

        # ── Domain-matching features (G2) ───────────────────────────────
        f.update(self._domain_matching_features(msg))

        # ── Routing / Received chain features (G3) ──────────────────────
        f.update(self._routing_features(msg))

        # ── Structural irregularity features (G4) ───────────────────────
        f.update(self._structural_features(msg))

        # ── Authentication features (G5) ────────────────────────────────
        f.update(self._authentication_features(msg))

        # ── Miscellaneous numerical / content features ───────────────────
        f.update(self._misc_features(msg))

        return f

    # ------------------------------------------------------------------
    # G1 — Header presence / absence
    # ------------------------------------------------------------------

    def _header_presence_features(self, msg: email.message.Message) -> Dict[str, Any]:
        """
        Binary flags (0/1) indicating whether key headers are present.

        Feature names: has_<header_name_snake_case>
        """
        key_headers = [
            "Received", "Message-ID", "Date", "From", "To", "Reply-To",
            "Return-Path", "MIME-Version", "Content-Type",
            "Content-Transfer-Encoding", "Subject",
            "X-Mailer", "X-Originating-IP", "X-Spam-Status",
            "DKIM-Signature", "DomainKey-Signature",
            "Authentication-Results", "ARC-Authentication-Results",
            "Delivered-To", "Envelope-To", "Errors-To",
            "List-Unsubscribe", "Precedence", "X-Priority",
        ]
        features: Dict[str, Any] = {}
        for h in key_headers:
            snake = "has_" + h.lower().replace("-", "_")
            features[snake] = int(msg[h] is not None)

        # Derived presence features
        features["has_spf"] = int(
            self._find_auth_result(msg, "spf") is not None
        )
        features["has_dkim"] = int(
            self._find_auth_result(msg, "dkim") is not None
            or msg["DKIM-Signature"] is not None
        )
        features["has_dmarc"] = int(
            self._find_auth_result(msg, "dmarc") is not None
        )
        features["has_x_originating_ip"] = int(
            msg["X-Originating-IP"] is not None
        )
        return features

    # ------------------------------------------------------------------
    # G2 — Domain matching
    # ------------------------------------------------------------------

    def _domain_matching_features(self, msg: email.message.Message) -> Dict[str, Any]:
        """
        Boolean (0/1) and integer features comparing domains across headers.
        """
        from_domain = _normalize_domain(_extract_domain(msg.get("From", "")))
        reply_to_domain = _normalize_domain(_extract_domain(msg.get("Reply-To", "")))
        return_path_domain = _normalize_domain(_extract_domain(msg.get("Return-Path", "")))
        envelope_to_domain = _normalize_domain(_extract_domain(msg.get("Envelope-To", "")))
        to_domain = _normalize_domain(_extract_domain(msg.get("To", "")))

        # Message-ID domain
        mid = msg.get("Message-ID", "") or ""
        mid_domain = _normalize_domain(_extract_domain(mid)) if "@" in mid else ""

        # Received-chain first-hop domain
        received_list = msg.get_all("Received") or []
        first_received_domain = ""
        if received_list:
            m = _RE_RECEIVED_FROM.search(received_list[-1])
            if m:
                first_received_domain = _normalize_domain(m.group(1))

        features: Dict[str, Any] = {
            "from_domain_len": len(from_domain),
            "from_eq_reply_to_domain": int(
                bool(from_domain and reply_to_domain)
                and from_domain == reply_to_domain
            ),
            "from_eq_return_path_domain": int(
                bool(from_domain and return_path_domain)
                and from_domain == return_path_domain
            ),
            "from_eq_mid_domain": int(
                bool(from_domain and mid_domain)
                and from_domain == mid_domain
            ),
            "from_eq_first_received_domain": int(
                bool(from_domain and first_received_domain)
                and from_domain == first_received_domain
            ),
            "reply_to_present_and_differs": int(
                bool(reply_to_domain) and reply_to_domain != from_domain
            ),
            "return_path_present_and_differs": int(
                bool(return_path_domain) and return_path_domain != from_domain
            ),
            "to_domain_eq_from_domain": int(
                bool(to_domain and from_domain) and to_domain == from_domain
            ),
            "mid_domain_eq_from_domain": int(
                bool(mid_domain and from_domain) and mid_domain == from_domain
            ),
            "envelope_to_eq_from_domain": int(
                bool(envelope_to_domain and from_domain)
                and envelope_to_domain == from_domain
            ),
            # Subdomain check
            "reply_to_subdomain_of_from": int(
                bool(from_domain and reply_to_domain)
                and reply_to_domain.endswith("." + from_domain)
            ),
        }
        return features

    # ------------------------------------------------------------------
    # G3 — Routing anomalies
    # ------------------------------------------------------------------

    def _routing_features(self, msg: email.message.Message) -> Dict[str, Any]:
        """
        Features derived from the Received-header chain.
        """
        received_list = msg.get_all("Received") or []
        n_hops = len(received_list)

        # Parse each hop
        hop_ips: List[str] = []
        hop_domains: List[str] = []
        hop_tz_offsets: List[str] = []
        malformed_hops = 0

        for r in received_list:
            m = _RE_RECEIVED_FROM.search(r)
            if m:
                hop_domains.append(m.group(1))
                hop_domains.append(m.group(2))
            else:
                malformed_hops += 1

            for ip in _RE_IP.findall(r):
                hop_ips.append(ip)

            tz = _extract_tz_offset(r)
            if tz:
                hop_tz_offsets.append(tz)

        # Timezone consistency
        unique_tz = len(set(hop_tz_offsets))
        tz_mismatch = int(unique_tz > 1) if hop_tz_offsets else 0
        unusual_tz = int(
            any(tz not in _COMMON_TZ_OFFSETS for tz in hop_tz_offsets)
        )

        # IP diversity
        private_ips = sum(1 for ip in hop_ips if self._is_private_ip(ip))
        unique_ips = len(set(hop_ips))

        # Date anomaly: Date header vs first Received timestamp
        date_str = msg.get("Date", "")
        date_anomaly = self._compute_date_anomaly(date_str, received_list)

        features: Dict[str, Any] = {
            "n_received_hops": n_hops,
            "missing_received": int(n_hops == 0),
            "malformed_received_hops": malformed_hops,
            "n_unique_hop_ips": unique_ips,
            "n_private_ips_in_received": private_ips,
            "tz_mismatch_in_received": tz_mismatch,
            "n_unique_tz_offsets": unique_tz,
            "has_unusual_tz": unusual_tz,
            "n_hop_domains": len(set(hop_domains)),
            "date_vs_received_anomaly": date_anomaly,
            "excessive_hops": int(n_hops > 10),
        }
        return features

    def _is_private_ip(self, ip: str) -> bool:
        """Return True if IP is in RFC-1918 private range."""
        parts = list(map(int, ip.split(".")))
        if parts[0] == 10:
            return True
        if parts[0] == 172 and 16 <= parts[1] <= 31:
            return True
        if parts[0] == 192 and parts[1] == 168:
            return True
        return False

    def _compute_date_anomaly(
        self, date_str: str, received_list: List[str]
    ) -> int:
        """
        Return 1 if the Date header is more than 24 h from the first
        Received timestamp, indicating a forgery attempt.
        """
        if not date_str or not received_list:
            return 0
        msg_dt = _parse_date(date_str)
        if msg_dt is None:
            return 0

        # First Received (bottom of list = oldest)
        last_received = received_list[-1]
        # Try to find a date within the Received header (after semicolon)
        if ";" in last_received:
            rec_date_str = last_received.split(";", 1)[-1].strip()
            rec_dt = _parse_date(rec_date_str)
            if rec_dt is not None:
                delta_seconds = abs((msg_dt - rec_dt).total_seconds())
                return int(delta_seconds > 86400)  # 24 hours
        return 0

    # ------------------------------------------------------------------
    # G4 — Structural irregularities
    # ------------------------------------------------------------------

    def _structural_features(self, msg: email.message.Message) -> Dict[str, Any]:
        """
        Features based on header structure, encoding, and formatting.
        """
        # Duplicate headers
        all_keys = [k.lower() for k in msg.keys()]
        dup_headers = len(all_keys) - len(set(all_keys))

        subject = msg.get("Subject", "") or ""
        from_val = msg.get("From", "") or ""
        date_val = msg.get("Date", "") or ""

        # Non-ASCII
        subject_non_ascii = _has_non_ascii(subject)
        from_non_ascii = _has_non_ascii(from_val)

        # Malformed Date
        malformed_date = int(_parse_date(date_val) is None and bool(date_val))

        # Subject features
        subject_len = len(subject)
        subject_has_re = int(subject.lower().startswith("re:"))
        subject_all_caps = int(subject.isupper() and len(subject) > 3)
        subject_exclaim = subject.count("!")
        subject_question = subject.count("?")

        # Encoding anomalies
        cte = _safe_lower(msg.get("Content-Transfer-Encoding", ""))
        mime_ver = msg.get("MIME-Version", "")
        has_mime = int(bool(mime_ver))
        is_base64 = int("base64" in cte)
        is_qp = int("quoted-printable" in cte)

        # Message-ID format
        mid = msg.get("Message-ID", "") or ""
        mid_malformed = int(bool(mid) and not re.match(r"<[^>]+@[^>]+>", mid.strip()))
        mid_len = len(mid)

        # Content-Type
        ct = _safe_lower(msg.get("Content-Type", ""))
        is_multipart = int("multipart" in ct)
        is_html_only = int("text/html" in ct and "multipart" not in ct)

        # Total header count
        total_headers = len(all_keys)

        features: Dict[str, Any] = {
            "duplicate_headers": dup_headers,
            "subject_non_ascii": subject_non_ascii,
            "from_non_ascii": from_non_ascii,
            "malformed_date": malformed_date,
            "subject_len": subject_len,
            "subject_has_re": subject_has_re,
            "subject_all_caps": subject_all_caps,
            "subject_exclaim_count": subject_exclaim,
            "subject_question_count": subject_question,
            "has_mime_version": has_mime,
            "is_base64_encoded": is_base64,
            "is_quoted_printable": is_qp,
            "mid_malformed": mid_malformed,
            "mid_len": mid_len,
            "is_multipart": is_multipart,
            "is_html_only": is_html_only,
            "total_header_count": total_headers,
        }
        return features

    # ------------------------------------------------------------------
    # G5 — Authentication
    # ------------------------------------------------------------------

    def _authentication_features(self, msg: email.message.Message) -> Dict[str, Any]:
        """
        SPF, DKIM, DMARC pass/fail/neutral results.
        Encoded as: pass=1, fail=-1, neutral/none/unknown=0.
        """
        spf_result = self._find_auth_result(msg, "spf")
        dkim_result = self._find_auth_result(msg, "dkim")
        dmarc_result = self._find_auth_result(msg, "dmarc")

        def encode(r: Optional[str]) -> int:
            if r is None:
                return 0
            r = r.lower()
            if "pass" in r:
                return 1
            if "fail" in r or "reject" in r:
                return -1
            return 0

        # DKIM-Signature header presence and basic validation
        dkim_sig = msg.get("DKIM-Signature", "") or ""
        dkim_sig_valid_format = int(
            bool(dkim_sig) and "v=" in dkim_sig and "d=" in dkim_sig
        )

        features: Dict[str, Any] = {
            "spf_result": encode(spf_result),
            "dkim_result": encode(dkim_result),
            "dmarc_result": encode(dmarc_result),
            "dkim_signature_present": int(bool(dkim_sig)),
            "dkim_signature_valid_format": dkim_sig_valid_format,
            "auth_results_present": int(msg.get("Authentication-Results") is not None),
            # Combined: all three pass
            "all_auth_pass": int(
                encode(spf_result) == 1
                and encode(dkim_result) == 1
                and encode(dmarc_result) == 1
            ),
            # Any authentication failure
            "any_auth_fail": int(
                encode(spf_result) == -1
                or encode(dkim_result) == -1
                or encode(dmarc_result) == -1
            ),
        }
        return features

    def _find_auth_result(
        self, msg: email.message.Message, protocol: str
    ) -> Optional[str]:
        """
        Search Authentication-Results and ARC-Authentication-Results headers
        for the given protocol result (spf/dkim/dmarc).
        """
        for header_name in ("Authentication-Results", "ARC-Authentication-Results"):
            values = msg.get_all(header_name) or []
            for v in values:
                v_lower = v.lower()
                if protocol in v_lower:
                    # Find the result token after the protocol name
                    pattern = rf"{protocol}\s*=\s*(\S+)"
                    m = re.search(pattern, v_lower)
                    if m:
                        return m.group(1).rstrip(";,")
        return None

    # ------------------------------------------------------------------
    # Miscellaneous features
    # ------------------------------------------------------------------

    def _misc_features(self, msg: email.message.Message) -> Dict[str, Any]:
        """
        Additional features: X-Mailer, priority, list-management headers.
        """
        x_mailer = msg.get("X-Mailer", "") or ""
        x_priority = msg.get("X-Priority", "") or ""
        precedence = _safe_lower(msg.get("Precedence", ""))
        list_unsub = msg.get("List-Unsubscribe", "") or ""

        # Priority parsing
        try:
            priority_val = int(re.search(r"\d", x_priority).group())
        except (AttributeError, ValueError):
            priority_val = 3  # normal

        features: Dict[str, Any] = {
            "has_x_mailer": int(bool(x_mailer)),
            "x_mailer_len": len(x_mailer),
            "x_priority_value": priority_val,
            "is_bulk_precedence": int(precedence in ("bulk", "list", "junk")),
            "has_list_unsubscribe": int(bool(list_unsub)),
            "list_unsubscribe_len": len(list_unsub),
            # Sender count
            "n_to_addresses": len(_RE_EMAIL_ADDR.findall(msg.get("To", "") or "")),
            "n_cc_addresses": len(_RE_EMAIL_ADDR.findall(msg.get("Cc", "") or "")),
            "n_bcc_addresses": len(_RE_EMAIL_ADDR.findall(msg.get("Bcc", "") or "")),
            # X-Spam
            "has_x_spam_status": int(msg.get("X-Spam-Status") is not None),
            "x_spam_score": self._parse_spam_score(msg.get("X-Spam-Status", "")),
        }
        return features

    def _parse_spam_score(self, spam_status: str) -> float:
        """Extract numerical spam score from X-Spam-Status header."""
        if not spam_status:
            return 0.0
        m = re.search(r"score=([\d.+-]+)", spam_status, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        return 0.0

    # ------------------------------------------------------------------
    # Empty feature dict (fallback on parse failure)
    # ------------------------------------------------------------------

    def _empty_features(self) -> Dict[str, Any]:
        """Return a zero-filled feature dict for emails that cannot be parsed."""
        dummy_msg = message_from_string("")
        return self.extract_from_message(dummy_msg)


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def extract_features_from_directory(
    directory: Path | str,
    label: int,
    extractor: Optional[EmailHeaderExtractor] = None,
    max_emails: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Walk a directory of raw email files and extract features for each.

    Parameters
    ----------
    directory:
        Root directory containing .eml / plaintext email files.
    label:
        Integer class label (0 = ham, 1 = spam, 2 = phishing).
    extractor:
        Optional pre-created :class:`EmailHeaderExtractor`. Created if None.
    max_emails:
        If given, stop after this many emails.

    Returns
    -------
    list of dict
        One feature dict per email (with 'label' key appended).
    """
    directory = Path(directory)
    if extractor is None:
        extractor = EmailHeaderExtractor()

    rows: List[Dict[str, Any]] = []
    files = list(directory.rglob("*"))
    files = [f for f in files if f.is_file()]

    if max_emails is not None:
        files = files[:max_emails]

    for path in files:
        try:
            feats = extractor.extract_from_file(path)
            feats["label"] = label
            rows.append(feats)
        except Exception as exc:
            logger.debug("Skipping %s: %s", path, exc)

    logger.info(
        "Extracted %d feature rows from %s (label=%d)",
        len(rows), directory, label
    )
    return rows
