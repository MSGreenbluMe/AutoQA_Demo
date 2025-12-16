"""
SSR - Semantic Similarity Rating

Based on: arXiv:2510.08338 (PyMC Labs)

Key principles:
- NOT direct LLM number rating (1-5)
- Text elicitation → embedding → similarity → DISTRIBUTION
- Output is probability distribution, not single score
"""

import json
import re
from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class SSRResult:
    """SSR result - probability distribution"""
    dimension: str
    distribution: Dict[str, float]  # {"low": 0.1, "medium": 0.3, "high": 0.6}
    text_response: str              # Original text response (explainability)
    top_category: str
    confidence: float               # Max value in distribution


class SSRAnalyzer:
    """
    Semantic Similarity Rating analyzer.

    Instead of direct number request:
    1. LLM generates text description
    2. Text is embedded
    3. Compared to reference statements
    4. Result is distribution, not single number
    """

    def __init__(self, gemini_model, embedding_model=None):
        """
        Args:
            gemini_model: Gemini GenerativeModel for text generation
            embedding_model: SentenceTransformer for embeddings (optional)
        """
        self.model = gemini_model
        self.embedder = embedding_model
        self.generation_config = {
            "temperature": 0.3,
            "max_output_tokens": 512,
        }

        self.references = self._load_references()
        self.prompts = self._load_prompts()
        self._reference_embeddings = {}

    def _load_references(self) -> Dict[str, Dict[str, List[str]]]:
        """Reference statements for each dimension and category."""
        return {
            "empathy": {
                "low": [
                    "Agent showed no understanding of customer situation",
                    "Agent was dismissive and cold",
                    "No empathetic phrases used",
                    "Agent ignored customer emotions completely"
                ],
                "medium": [
                    "Agent acknowledged the issue but could show more understanding",
                    "Some empathy shown but not consistently",
                    "Moderate emotional awareness displayed",
                    "Basic acknowledgment of customer feelings"
                ],
                "high": [
                    "Agent deeply understood customer emotions",
                    "Excellent empathetic language throughout",
                    "Agent actively validated customer feelings",
                    "Strong emotional connection established"
                ]
            },
            "professionalism": {
                "low": [
                    "Agent was unprofessional and inappropriate",
                    "Poor conduct, used informal language incorrectly",
                    "Agent seemed careless and disengaged",
                    "Multiple protocol violations observed"
                ],
                "medium": [
                    "Agent maintained basic professional standards",
                    "Adequate conduct with room for improvement",
                    "Professional but not exceptional service",
                    "Met minimum requirements for professional behavior"
                ],
                "high": [
                    "Agent demonstrated exemplary professional conduct",
                    "Outstanding service delivery with proper etiquette",
                    "Model behavior for customer service excellence",
                    "Exceeded all professional standards"
                ]
            },
            "fcr_likelihood": {
                "unlikely": [
                    "Issue was not resolved, customer will likely call back",
                    "Problem remains unaddressed after this call",
                    "Customer left without solution to their issue",
                    "High probability of repeat contact needed"
                ],
                "possible": [
                    "Partial resolution achieved but may need follow-up",
                    "Some aspects resolved but customer may return",
                    "Uncertain if issue was fully addressed",
                    "Moderate chance of first call resolution"
                ],
                "likely": [
                    "Issue fully resolved in this interaction",
                    "Customer problem completely addressed",
                    "High confidence in first call resolution",
                    "Comprehensive solution provided, no follow-up needed"
                ]
            }
        }

    def _load_prompts(self) -> Dict[str, str]:
        """Prompts for text elicitation per dimension."""
        return {
            "empathy": """Na základe tohto prepisu hovoru, opíš v 2-3 vetách
ako agent preukázal (alebo nepreukázal) porozumenie emočného stavu zákazníka.
Zameraj sa na konkrétne správanie a frázy.""",

            "professionalism": """Na základe tohto prepisu hovoru, opíš v 2-3 vetách
profesionálne správanie agenta. Zváž jazyk, tón, dodržiavanie etikety
a celkovú kvalitu služby.""",

            "fcr_likelihood": """Na základe tohto prepisu hovoru, opíš v 2-3 vetách
či bol problém zákazníka úplne vyriešený. Zváž či zákazník bude pravdepodobne
musieť volať znova kvôli rovnakému problému."""
        }

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for texts."""
        if self.embedder is not None:
            return self.embedder.encode(texts)

        # Fallback: simple TF-IDF-like approach
        return self._simple_embeddings(texts)

    def _simple_embeddings(self, texts: List[str]) -> np.ndarray:
        """Simple word-based embeddings as fallback."""
        all_words = set()
        for text in texts:
            words = text.lower().split()
            all_words.update(words)

        vocab = {word: i for i, word in enumerate(sorted(all_words))}

        embeddings = np.zeros((len(texts), len(vocab)))
        for i, text in enumerate(texts):
            words = text.lower().split()
            for word in words:
                if word in vocab:
                    embeddings[i, vocab[word]] += 1
            norm = np.linalg.norm(embeddings[i])
            if norm > 0:
                embeddings[i] /= norm

        return embeddings

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def _compute_reference_embeddings(self, dimension: str):
        """Pre-compute embeddings for reference statements."""
        if dimension in self._reference_embeddings:
            return

        refs = self.references.get(dimension, {})
        self._reference_embeddings[dimension] = {}

        for category, statements in refs.items():
            embeddings = self._get_embeddings(statements)
            self._reference_embeddings[dimension][category] = embeddings

    def elicit_text_response(self, transcript: str, dimension: str) -> str:
        """Get text description from LLM instead of direct rating."""
        prompt_template = self.prompts.get(dimension, "Opíš kvalitu.")
        prompt = f"""{prompt_template}

PREPIS HOVORU:
{transcript}

Poskytni svoj opis (2-3 vety, žiadne čísla ani hodnotenia):"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            return response.text.strip()
        except Exception as e:
            return f"Analýza zlyhala: {str(e)}"

    def compute_distribution(
        self,
        text_response: str,
        dimension: str
    ) -> Dict[str, float]:
        """
        Compute probability distribution over categories.

        1. Embed text response
        2. Compare to reference embeddings
        3. Softmax to distribution
        """
        self._compute_reference_embeddings(dimension)

        # Embed response
        response_embedding = self._get_embeddings([text_response])[0]

        # Compute similarities
        similarities = {}
        for category, ref_embeddings in self._reference_embeddings[dimension].items():
            sims = []
            for ref_emb in ref_embeddings:
                sim = self._cosine_similarity(response_embedding, ref_emb)
                sims.append(sim)
            similarities[category] = np.mean(sims)

        # Softmax with temperature
        values = np.array(list(similarities.values()))
        temperature = 0.5
        exp_values = np.exp(values / temperature)
        softmax_values = exp_values / np.sum(exp_values)

        return {k: round(float(v), 3) for k, v in zip(similarities.keys(), softmax_values)}

    def analyze(self, transcript: str, dimension: str) -> SSRResult:
        """Main SSR analysis for one dimension."""
        if dimension not in self.references:
            raise ValueError(f"Unknown dimension: {dimension}")

        # 1. Elicit text response
        text_response = self.elicit_text_response(transcript, dimension)

        # 2. Compute distribution
        distribution = self.compute_distribution(text_response, dimension)

        # 3. Find top category
        top_category = max(distribution, key=distribution.get)
        confidence = distribution[top_category]

        return SSRResult(
            dimension=dimension,
            distribution=distribution,
            text_response=text_response,
            top_category=top_category,
            confidence=confidence
        )

    def analyze_all(self, transcript: str) -> Dict[str, SSRResult]:
        """Analyze all configured dimensions."""
        results = {}
        for dimension in self.references.keys():
            results[dimension] = self.analyze(transcript, dimension)
        return results
