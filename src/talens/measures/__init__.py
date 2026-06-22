"""Information-theoretic measures — the three complementary lenses.

* :func:`~talens.measures.vinfo.v_information` — V-usable info / PVI
  (usable-info axis; what a bounded adversary can extract).
* :func:`~talens.measures.mdl.online_code_length` — MDL online-coding +
  Surplus Description Length (code-length / complexity axis).
* :func:`~talens.measures.club.club_mi_upper_bound` — CLUB MI upper
  bound (brackets leakage from above).
* :func:`~talens.measures.pid.pid_mmi` — MMI partial-information
  decomposition of the QK/OV attention channel (which operand leaks).
"""

from __future__ import annotations

from .channel_error_bounds import fano_equivocation, union_bhattacharyya
from .club import club_mi_upper_bound
from .mdl import online_code_length, online_code_length_retrieval
from .pid import pid_mmi
from .vinfo import v_information, v_information_retrieval
from .vinfo_capacity import v_information_capacity

__all__ = [
    "v_information",              # class-probe family (row-split; resolution A)
    "v_information_capacity",     # capacity-matched class-probe (PCA/randproj/gauss/knn)
    "v_information_retrieval",    # retrieval family (vocab-disjoint; resolution B)
    "online_code_length",
    "online_code_length_retrieval",
    "club_mi_upper_bound",
    "pid_mmi",                    # MMI-PID of the QK/OV attention channel
    "union_bhattacharyya",        # geometry-only upper bound on BNN/MAP error
    "fano_equivocation",          # Fano lower bound via fresh-noise equivocation
]
