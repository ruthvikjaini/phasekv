"""Phase-aware, query-adaptive KV-cache compression.

Independent NumPy implementation. Baseline presses re-implement scoring
rules from published descriptions (see docs/RESEARCH.md); no code is
copied from NVIDIA/kvpress or any other repository.
"""

from phasekv.attention import AttentionContext, make_context
from phasekv.policies import POLICIES, apply_policy
from phasekv.adaptive import PhaseAdaptivePolicy
from phasekv.probes import evaluate

__version__ = "0.1.0"
__all__ = [
    "AttentionContext",
    "make_context",
    "POLICIES",
    "apply_policy",
    "PhaseAdaptivePolicy",
    "evaluate",
]
