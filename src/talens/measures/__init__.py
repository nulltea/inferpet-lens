"""Information-theoretic measures — the three complementary lenses.

* :func:`~talens.measures.vinfo.v_information` — V-usable info / PVI
  (usable-info axis; what a bounded adversary can extract).
* :func:`~talens.measures.mdl.online_code_length` — MDL online-coding +
  Surplus Description Length (code-length / complexity axis).
* :func:`~talens.measures.club.club_mi_upper_bound` — CLUB MI upper
  bound (brackets leakage from above).
"""

from __future__ import annotations

from .club import club_mi_upper_bound
from .mdl import online_code_length, online_code_length_retrieval
from .vinfo import v_information, v_information_retrieval

__all__ = [
    "v_information",              # class-probe family (row-split; resolution A)
    "v_information_retrieval",    # retrieval family (vocab-disjoint; resolution B)
    "online_code_length",
    "online_code_length_retrieval",
    "club_mi_upper_bound",
]
