"""
CCM - Conformal Correlation Matrix

Based on: Computer Communications 247 (2026)

Key principles:
- Conformal prediction gives prediction SETS with guaranteed coverage
- Set size = uncertainty indicator
- CCM shows which classes are easily confusable
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class CCMResult:
    """CCM analysis result"""
    dimension: str
    prediction_set: List[str]
    set_size: int
    confidence_level: float
    confidence_category: str  # "HIGH", "MEDIUM", "LOW"
    needs_review: bool


class CCMAnalyzer:
    """
    Conformal Correlation Matrix analyzer.

    Simplified implementation:
    - Uses probability distribution from SSR
    - Prediction set = classes above threshold
    - Set size indicates uncertainty
    """

    def __init__(self, coverage_level: float = 0.90):
        self.coverage_level = coverage_level
        self.threshold = 0.15  # Include classes with >15% probability

    def predict_with_uncertainty(
        self,
        dimension: str,
        class_probabilities: Dict[str, float]
    ) -> CCMResult:
        """Create prediction set with uncertainty quantification."""
        # Sort by probability descending
        sorted_probs = sorted(
            class_probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Build prediction set
        prediction_set = []
        cumulative = 0.0

        for category, prob in sorted_probs:
            if cumulative < self.coverage_level or prob >= self.threshold:
                prediction_set.append(category)
                cumulative += prob

        if not prediction_set:
            prediction_set = [sorted_probs[0][0]]

        set_size = len(prediction_set)

        # Determine confidence category
        if set_size == 1:
            confidence_category = "HIGH"
            needs_review = False
        elif set_size == 2:
            confidence_category = "MEDIUM"
            needs_review = False
        else:
            confidence_category = "LOW"
            needs_review = True

        return CCMResult(
            dimension=dimension,
            prediction_set=prediction_set,
            set_size=set_size,
            confidence_level=self.coverage_level,
            confidence_category=confidence_category,
            needs_review=needs_review
        )


def analyze_ssr_with_ccm(
    ssr_results: Dict[str, 'SSRResult'],
    ccm: CCMAnalyzer
) -> Dict[str, CCMResult]:
    """Apply CCM to all SSR results."""
    ccm_results = {}

    for dimension, ssr_result in ssr_results.items():
        ccm_result = ccm.predict_with_uncertainty(
            dimension=dimension,
            class_probabilities=ssr_result.distribution
        )
        ccm_results[dimension] = ccm_result

    return ccm_results
