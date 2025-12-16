"""
ERFC - Emotion Recognition and Forecasting in Conversation

Based on: arXiv:2509.18175 (Accenture)

Key principles:
- TURN-based (not per-utterance) - turn = both speakers completed exchange
- Speaker inter-dependency - agent emotion affects customer and vice versa
- AVD attributes (Activation, Valence, Dominance) - emotion intensity
- Forecasting - predict future emotion trajectory
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class EmotionCategory(Enum):
    HAPPY = "happy"
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"
    SAD = "sad"


@dataclass
class AVDAttributes:
    """Activation, Valence, Dominance - captures emotion intensity"""
    activation: float = 0.0   # -1 (calm) to +1 (excited)
    valence: float = 0.0      # -1 (negative) to +1 (positive)
    dominance: float = 0.0    # -1 (weak) to +1 (strong)


@dataclass
class EmotionState:
    category: str = "neutral"
    confidence: float = 0.5
    avd: AVDAttributes = field(default_factory=AVDAttributes)


@dataclass
class Turn:
    """Turn = both agent and customer completed their utterances"""
    turn_index: int
    agent_text: str
    customer_text: str
    agent_emotion: Optional[EmotionState] = None
    customer_emotion: Optional[EmotionState] = None
    timestamp_start: float = 0.0
    timestamp_end: float = 0.0


@dataclass
class InterventionPoint:
    turn_index: int
    intervention_type: str  # "positive_impact", "negative_impact", "missed_opportunity"
    description: str
    customer_change: float = 0.0
    agent_valence: float = 0.0


@dataclass
class ERFCResult:
    turns: List[Turn]
    forecasted_turns: List[Turn]
    customer_trajectory: str  # "improving", "declining", "stable", "volatile"
    agent_consistency: float  # 0-1
    intervention_points: List[InterventionPoint]


class ERFCAnalyzer:
    """
    ERFC implementation using existing Gemini model.

    Difference from basic sentiment:
    - Works with TURNS, not individual utterances
    - Extracts AVD attributes for intensity
    - Analyzes mutual dynamics between speakers
    - Predicts future emotion development
    """

    def __init__(self, gemini_model, generation_config: dict = None):
        """
        Args:
            gemini_model: Initialized genai.GenerativeModel instance
            generation_config: Optional config override
        """
        self.model = gemini_model
        self.generation_config = generation_config or {
            "temperature": 0.2,
            "max_output_tokens": 2048,
            "response_mime_type": "application/json",
        }

    def create_turns_from_segments(
        self,
        segments: List[Dict],
        swap_speakers: bool = False
    ) -> List[Turn]:
        """
        Convert transcript segments to turns.

        Turn is complete when:
        - Both agent and customer have at least one utterance
        - Captures mutual interaction
        """
        if not segments:
            return []

        turns = []
        current_agent = []
        current_customer = []
        turn_start = segments[0].get("start", 0)
        last_end = 0

        for seg in segments:
            speaker = seg.get("speaker", "")
            text = seg.get("text", "").strip()

            if not text:
                continue

            # Determine role (respect swap)
            is_agent = speaker == "Agent"
            if swap_speakers:
                is_agent = not is_agent

            if is_agent:
                current_agent.append(text)
            else:
                current_customer.append(text)

            last_end = seg.get("end", last_end)

            # Turn is complete when both have content
            if current_agent and current_customer:
                turns.append(Turn(
                    turn_index=len(turns),
                    agent_text=" ".join(current_agent),
                    customer_text=" ".join(current_customer),
                    timestamp_start=turn_start,
                    timestamp_end=last_end
                ))
                current_agent = []
                current_customer = []
                turn_start = last_end

        # Handle remaining incomplete turn
        if current_agent or current_customer:
            turns.append(Turn(
                turn_index=len(turns),
                agent_text=" ".join(current_agent) if current_agent else "[no response]",
                customer_text=" ".join(current_customer) if current_customer else "[no response]",
                timestamp_start=turn_start,
                timestamp_end=last_end
            ))

        return turns

    def analyze_turn_emotions(self, turns: List[Turn]) -> List[Turn]:
        """
        Analyze emotions for each turn WITH context from previous turns.
        """
        if not turns:
            return turns

        analyzed_turns = []

        for i, turn in enumerate(turns):
            # Context = last 3 turns
            context_start = max(0, i - 3)
            context_turns = analyzed_turns[context_start:]

            prompt = self._build_emotion_prompt(turn, context_turns)

            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config
                )
                emotions = self._parse_emotion_response(response.text)

                turn.agent_emotion = emotions.get("agent", EmotionState())
                turn.customer_emotion = emotions.get("customer", EmotionState())
            except Exception as e:
                # Fallback to neutral
                turn.agent_emotion = EmotionState(category="neutral", confidence=0.3)
                turn.customer_emotion = EmotionState(category="neutral", confidence=0.3)

            analyzed_turns.append(turn)

        return analyzed_turns

    def forecast_emotions(self, turns: List[Turn], horizon: int = 3) -> List[Turn]:
        """Predict emotions for next K turns based on trajectory."""
        if len(turns) < 3:
            return []

        prompt = self._build_forecast_prompt(turns, horizon)

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            return self._parse_forecast_response(response.text, len(turns), horizon)
        except Exception:
            return []

    def analyze_trajectory(self, turns: List[Turn]) -> str:
        """Determine overall customer emotion trend."""
        if len(turns) < 2:
            return "stable"

        valences = [t.customer_emotion.avd.valence
                   for t in turns if t.customer_emotion]

        if len(valences) < 2:
            return "stable"

        mid = len(valences) // 2
        first_avg = sum(valences[:mid]) / mid if mid > 0 else 0
        second_avg = sum(valences[mid:]) / (len(valences) - mid)

        diff = second_avg - first_avg
        variance = sum((v - sum(valences)/len(valences))**2 for v in valences) / len(valences)

        if diff > 0.25:
            return "improving"
        elif diff < -0.25:
            return "declining"
        elif variance > 0.25:
            return "volatile"
        return "stable"

    def identify_intervention_points(self, turns: List[Turn]) -> List[InterventionPoint]:
        """
        Identify moments where:
        - Agent's intervention helped/hurt
        - Agent could have intervened more empathetically
        """
        interventions = []

        for i in range(1, len(turns)):
            prev = turns[i - 1]
            curr = turns[i]

            if not (prev.customer_emotion and curr.customer_emotion):
                continue

            prev_val = prev.customer_emotion.avd.valence
            curr_val = curr.customer_emotion.avd.valence
            agent_val = curr.agent_emotion.avd.valence if curr.agent_emotion else 0

            change = curr_val - prev_val

            # Positive impact - customer improved
            if change > 0.3:
                interventions.append(InterventionPoint(
                    turn_index=i,
                    intervention_type="positive_impact",
                    description="Agentova odpoveď zlepšila náladu zákazníka",
                    customer_change=change,
                    agent_valence=agent_val
                ))

            # Negative impact - customer got worse
            elif change < -0.3:
                interventions.append(InterventionPoint(
                    turn_index=i,
                    intervention_type="negative_impact",
                    description="Nálada zákazníka sa zhoršila po odpovedi agenta",
                    customer_change=change,
                    agent_valence=agent_val
                ))

            # Missed opportunity - customer negative but agent not empathetic
            if curr_val < -0.2 and agent_val < 0.3:
                # Only add if not already added as negative_impact
                if not interventions or interventions[-1].turn_index != i:
                    interventions.append(InterventionPoint(
                        turn_index=i,
                        intervention_type="missed_opportunity",
                        description="Agent mohol prejaviť viac empatie",
                        customer_change=change,
                        agent_valence=agent_val
                    ))

        return interventions

    def calculate_agent_consistency(self, turns: List[Turn]) -> float:
        """
        Calculate how consistently agent maintained positive emotion.
        Ideal agent = high valence + low variance
        """
        valences = [t.agent_emotion.avd.valence
                   for t in turns if t.agent_emotion]

        if not valences:
            return 0.5

        mean_val = sum(valences) / len(valences)
        variance = sum((v - mean_val)**2 for v in valences) / len(valences)

        positivity = (mean_val + 1) / 2  # Normalize 0-1
        stability = max(0, 1 - variance)

        return round((positivity * 0.6 + stability * 0.4), 2)

    def analyze(
        self,
        segments: List[Dict],
        swap_speakers: bool = False
    ) -> ERFCResult:
        """Main method - complete ERFC analysis."""
        # 1. Create turns
        turns = self.create_turns_from_segments(segments, swap_speakers)

        if not turns:
            return ERFCResult(
                turns=[],
                forecasted_turns=[],
                customer_trajectory="stable",
                agent_consistency=0.5,
                intervention_points=[]
            )

        # 2. Analyze emotions
        turns = self.analyze_turn_emotions(turns)

        # 3. Forecast future
        forecasted = self.forecast_emotions(turns)

        # 4. Analyze trajectory
        trajectory = self.analyze_trajectory(turns)

        # 5. Find intervention points
        interventions = self.identify_intervention_points(turns)

        # 6. Calculate agent consistency
        consistency = self.calculate_agent_consistency(turns)

        return ERFCResult(
            turns=turns,
            forecasted_turns=forecasted,
            customer_trajectory=trajectory,
            agent_consistency=consistency,
            intervention_points=interventions
        )

    def _build_emotion_prompt(self, turn: Turn, context: List[Turn]) -> str:
        """Build prompt for emotion analysis."""
        context_text = ""
        if context:
            context_text = "Predošlé turny:\n"
            for t in context[-3:]:
                cust_em = t.customer_emotion.category if t.customer_emotion else "unknown"
                cust_val = t.customer_emotion.avd.valence if t.customer_emotion else 0
                context_text += f"Turn {t.turn_index}: Zákazník={cust_em} (valence={cust_val:.2f})\n"

        return f"""Analyzuj emócie v tomto úseku hovoru z call centra.

{context_text}

AKTUÁLNY TURN {turn.turn_index}:
Agent povedal: "{turn.agent_text}"
Zákazník povedal: "{turn.customer_text}"

Analyzuj emócie OBOCH - agenta aj zákazníka. Zváž:
1. Ako tón agenta ovplyvnil zákazníka
2. Emočnú trajektóriu z kontextu

Vráť JSON:
{{
  "agent": {{
    "category": "happy" alebo "neutral" alebo "frustrated" alebo "angry" alebo "sad",
    "activation": číslo od -1.0 (pokojný) do 1.0 (vzrušený),
    "valence": číslo od -1.0 (negatívny) do 1.0 (pozitívny),
    "dominance": číslo od -1.0 (slabý) do 1.0 (silný),
    "confidence": číslo od 0.0 do 1.0
  }},
  "customer": {{
    "category": "happy" alebo "neutral" alebo "frustrated" alebo "angry" alebo "sad",
    "activation": číslo od -1.0 do 1.0,
    "valence": číslo od -1.0 do 1.0,
    "dominance": číslo od -1.0 do 1.0,
    "confidence": číslo od 0.0 do 1.0
  }}
}}"""

    def _build_forecast_prompt(self, turns: List[Turn], horizon: int) -> str:
        """Build prompt for forecasting."""
        history = ""
        for t in turns[-5:]:
            cust = t.customer_emotion
            if cust:
                history += f"Turn {t.turn_index}: {cust.category}, valence={cust.avd.valence:.2f}\n"

        return f"""Na základe emočnej trajektórie v tomto hovore, predpovedaj nasledujúcich {horizon} turnov.

HISTÓRIA EMÓCIÍ:
{history}

Predpovedaj emóciu zákazníka pre nasledujúce turny, ak agent bude pokračovať rovnakým štýlom.

Vráť JSON pole:
[
  {{"turn": {len(turns)}, "category": "...", "valence": 0.0, "confidence": 0.0}},
  {{"turn": {len(turns)+1}, "category": "...", "valence": 0.0, "confidence": 0.0}},
  {{"turn": {len(turns)+2}, "category": "...", "valence": 0.0, "confidence": 0.0}}
]"""

    def _parse_emotion_response(self, response_text: str) -> Dict[str, EmotionState]:
        """Parse Gemini response to EmotionState objects."""
        try:
            text = response_text.strip()
            if text.startswith("```"):
                text = re.sub(r'^```\w*\n?', '', text)
                text = re.sub(r'\n?```$', '', text)

            data = json.loads(text)
            result = {}

            for role in ["agent", "customer"]:
                if role in data:
                    d = data[role]
                    result[role] = EmotionState(
                        category=d.get("category", "neutral"),
                        confidence=float(d.get("confidence", 0.5)),
                        avd=AVDAttributes(
                            activation=float(d.get("activation", 0)),
                            valence=float(d.get("valence", 0)),
                            dominance=float(d.get("dominance", 0))
                        )
                    )

            return result
        except Exception:
            return {}

    def _parse_forecast_response(
        self,
        response_text: str,
        current_turn: int,
        horizon: int
    ) -> List[Turn]:
        """Parse forecast response."""
        try:
            text = response_text.strip()
            if text.startswith("```"):
                text = re.sub(r'^```\w*\n?', '', text)
                text = re.sub(r'\n?```$', '', text)

            data = json.loads(text)

            forecasted = []
            for item in data[:horizon]:
                turn = Turn(
                    turn_index=item.get("turn", current_turn + len(forecasted)),
                    agent_text="[predikcia]",
                    customer_text="[predikcia]"
                )
                turn.customer_emotion = EmotionState(
                    category=item.get("category", "neutral"),
                    confidence=float(item.get("confidence", 0.5)),
                    avd=AVDAttributes(
                        valence=float(item.get("valence", 0))
                    )
                )
                forecasted.append(turn)

            return forecasted
        except Exception:
            return []
