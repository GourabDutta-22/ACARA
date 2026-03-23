"""
ARC — Adaptive Retrieval Controller
====================================
Matches the "Adaptive Retrieval Controller (ARC)" box from the flow diagram.

Responsibilities:
  - Adjust similarity_threshold dynamically (lower = broader recall)
  - Adjust top_k (number of chunks retrieved from Vector Memory)
  - Adjust chunk_size / chunk_overlap for the Dynamic Chunking Module
  - Provide the current params to any node that needs them
"""

import threading
import logging

logger = logging.getLogger(__name__)


class ARCController:
    """
    Adaptive Retrieval Controller.
    Thread-safe singleton that holds dynamic retrieval parameters
    and adjusts them based on pipeline feedback signals.
    """

    # Bounds
    MIN_THRESHOLD = 0.35
    MAX_THRESHOLD = 0.60
    MIN_TOP_K = 2
    MAX_TOP_K = 8
    MIN_CHUNK_SIZE = 300
    MAX_CHUNK_SIZE = 2000

    def __init__(
        self,
        similarity_threshold: float = 0.45,
        top_k: int = 3,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self._lock = threading.Lock()
        self._similarity_threshold = similarity_threshold
        self._top_k = top_k
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        # Track adjustment history for observability
        self._adjustment_count = 0

    # ------------------------------------------------------------------ #
    #  Adjustment callbacks                                                #
    # ------------------------------------------------------------------ #

    def adjust_on_weak_context(self):
        """
        Called by the Context Awareness Gate when context is WEAK.
        → lower threshold (broader recall), increase top_k.
        """
        with self._lock:
            old_thresh = self._similarity_threshold
            old_k = self._top_k

            # Lower threshold by 0.05, clamp to MIN
            self._similarity_threshold = max(
                self.MIN_THRESHOLD, self._similarity_threshold - 0.05
            )
            # Increase top_k by 1, clamp to MAX
            self._top_k = min(self.MAX_TOP_K, self._top_k + 1)
            self._adjustment_count += 1

            logger.info(
                "ARC [WEAK] threshold %.2f→%.2f  top_k %d→%d",
                old_thresh, self._similarity_threshold, old_k, self._top_k,
            )

    def adjust_on_strong_context(self):
        """
        Called by the Context Awareness Gate when context is STRONG.
        → tighten threshold slightly, restore top_k toward default.
        """
        with self._lock:
            old_thresh = self._similarity_threshold
            old_k = self._top_k

            # Nudge threshold up by 0.02, clamp to MAX
            self._similarity_threshold = min(
                self.MAX_THRESHOLD, self._similarity_threshold + 0.02
            )
            # Decay top_k back toward 3 (one step)
            if self._top_k > 3:
                self._top_k -= 1
            self._adjustment_count += 1

            logger.info(
                "ARC [STRONG] threshold %.2f→%.2f  top_k %d→%d",
                old_thresh, self._similarity_threshold, old_k, self._top_k,
            )

    def adjust_chunk_size(self, query: str):
        """
        Heuristic: longer / more complex queries benefit from smaller,
        more focused chunks; short queries can use larger chunks.
        """
        with self._lock:
            word_count = len(query.split())
            if word_count > 20:
                new_size = max(self.MIN_CHUNK_SIZE, self._chunk_size - 500)
                new_overlap = max(50, self._chunk_overlap - 25)
            elif word_count < 6:
                new_size = min(self.MAX_CHUNK_SIZE, self._chunk_size + 500)
                new_overlap = min(400, self._chunk_overlap + 25)
            else:
                return  # no change

            if new_size != self._chunk_size:
                logger.info(
                    "ARC chunk_size %d→%d (query words=%d)",
                    self._chunk_size, new_size, word_count,
                )
            self._chunk_size = new_size
            self._chunk_overlap = new_overlap

    def reset(self):
        """Resets the ARC parameters to their original defaults."""
        with self._lock:
            self._similarity_threshold = 0.45
            self._top_k = 3
            self._chunk_size = 1000
            self._chunk_overlap = 200
            self._adjustment_count = 0
            logger.info("ARC Controller reset to default parameters.")

    # ------------------------------------------------------------------ #
    #  Accessors                                                           #
    # ------------------------------------------------------------------ #

    @property
    def similarity_threshold(self) -> float:
        with self._lock:
            return self._similarity_threshold

    @property
    def top_k(self) -> int:
        with self._lock:
            return self._top_k

    @property
    def chunk_size(self) -> int:
        with self._lock:
            return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        with self._lock:
            return self._chunk_overlap

    def get_params(self) -> dict:
        with self._lock:
            return {
                "similarity_threshold": round(self._similarity_threshold, 3),
                "top_k": self._top_k,
                "chunk_size": self._chunk_size,
                "chunk_overlap": self._chunk_overlap,
                "adjustment_count": self._adjustment_count,
            }


# Global singleton — used across the entire backend
arc = ARCController()
