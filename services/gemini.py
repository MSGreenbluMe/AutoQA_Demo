"""
Google Gemini Analysis Service

This module handles AI-powered call analysis using Google Gemini API
to calculate quality metrics for call center conversations.
"""

import json
from typing import Optional

import google.generativeai as genai

from utils.config import get_api_key


class GeminiAnalyzer:
    """Handle call quality analysis via Google Gemini API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Gemini analyzer.

        Args:
            api_key: Gemini API key. If not provided, reads from config.
        """
        self.api_key = api_key or get_api_key("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found. Set it in .env or Streamlit secrets.")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-2.5-pro-preview-05-06")
        self.generation_config = {
            "temperature": 0.3,
            "max_output_tokens": 4096,
        }

    def analyze_call(self, transcript: dict) -> dict:
        """Perform comprehensive call analysis.

        Args:
            transcript: Processed transcript dictionary with segments.

        Returns:
            Dictionary containing all calculated metrics.
        """
        full_text = transcript.get("full_text", "")
        segments = transcript.get("segments", [])
        duration = transcript.get("duration", 0)

        # Build formatted transcript for analysis
        formatted_transcript = self._format_transcript_for_analysis(segments)

        # Calculate all metrics
        metrics = {}

        # 1. Sentiment Analysis
        metrics["sentiment"] = self._analyze_sentiment(formatted_transcript)

        # 2. FCR Prediction
        metrics["fcr"] = self._predict_fcr(formatted_transcript)

        # 3. Speech Metrics (calculated locally from segments)
        metrics["speech"] = self._calculate_speech_metrics(segments, duration)

        # 4. Interruption Detection
        metrics["interruptions"] = self._detect_interruptions(segments)

        # 5. Empathy & Professionalism
        metrics["empathy_professionalism"] = self._analyze_empathy_professionalism(
            formatted_transcript
        )

        # 6. Call Duration
        metrics["duration"] = {
            "total_seconds": duration,
            "formatted": self._format_duration(duration),
        }

        # 7. Overall Quality Score
        metrics["overall_score"] = self._calculate_overall_score(metrics)

        return metrics

    def _format_transcript_for_analysis(self, segments: list) -> str:
        """Format transcript segments for Gemini analysis.

        Args:
            segments: List of transcript segments.

        Returns:
            Formatted transcript string.
        """
        lines = []
        for seg in segments:
            speaker = seg.get("speaker", "Unknown")
            text = seg.get("text", "")
            start = seg.get("start", 0)
            timestamp = self._format_duration(start)
            lines.append(f"[{timestamp}] {speaker}: {text}")
        return "\n".join(lines)

    def _analyze_sentiment(self, transcript: str) -> dict:
        """Analyze overall sentiment of the call.

        Args:
            transcript: Formatted transcript.

        Returns:
            Sentiment analysis results.
        """
        prompt = f"""Analyzuj sentiment nasledujúceho prepisu hovoru z call centra.

Poskyni:
1. overall_sentiment: "Positive", "Neutral", alebo "Negative"
2. sentiment_score: číslo od -1.0 (veľmi negatívne) do +1.0 (veľmi pozitívne)
3. sentiment_start: sentiment na začiatku hovoru ("Positive", "Neutral", "Negative")
4. sentiment_end: sentiment na konci hovoru ("Positive", "Neutral", "Negative")
5. explanation: stručné vysvetlenie v slovenčine (max 2 vety)

Odpovedz VÝHRADNE vo formáte JSON bez markdown formátovania:
{{"overall_sentiment": "...", "sentiment_score": 0.0, "sentiment_start": "...", "sentiment_end": "...", "explanation": "..."}}

Prepis:
{transcript}"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            result = self._parse_json_response(response.text)
            return result
        except Exception as e:
            return {
                "overall_sentiment": "Neutral",
                "sentiment_score": 0.0,
                "sentiment_start": "Neutral",
                "sentiment_end": "Neutral",
                "explanation": f"Analýza zlyhala: {str(e)}",
                "error": True
            }

    def _predict_fcr(self, transcript: str) -> dict:
        """Predict First Call Resolution status.

        Args:
            transcript: Formatted transcript.

        Returns:
            FCR prediction results.
        """
        prompt = f"""Na základe prepisu hovoru z call centra urči, či bol problém zákazníka vyriešený počas tohto hovoru (First Call Resolution).

Poskyni:
1. fcr_status: "Yes" (vyriešené), "No" (nevyriešené), alebo "Uncertain" (nejasné)
2. confidence: číslo 0-100 vyjadrujúce istotu predikcie
3. reasoning: stručné zdôvodnenie v slovenčine (max 2 vety)

Odpovedz VÝHRADNE vo formáte JSON bez markdown formátovania:
{{"fcr_status": "...", "confidence": 0, "reasoning": "..."}}

Prepis:
{transcript}"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            result = self._parse_json_response(response.text)
            return result
        except Exception as e:
            return {
                "fcr_status": "Uncertain",
                "confidence": 0,
                "reasoning": f"Analýza zlyhala: {str(e)}",
                "error": True
            }

    def _calculate_speech_metrics(self, segments: list, duration: float) -> dict:
        """Calculate speech-related metrics from segments.

        Args:
            segments: List of transcript segments.
            duration: Total call duration in seconds.

        Returns:
            Speech metrics dictionary.
        """
        agent_words = 0
        customer_words = 0
        agent_time = 0.0
        customer_time = 0.0

        for seg in segments:
            text = seg.get("text", "")
            word_count = len(text.split())
            segment_duration = seg.get("end", 0) - seg.get("start", 0)

            if seg.get("speaker") == "Agent":
                agent_words += word_count
                agent_time += segment_duration
            else:
                customer_words += word_count
                customer_time += segment_duration

        total_words = agent_words + customer_words
        total_speaking_time = agent_time + customer_time

        # Calculate words per minute
        if duration > 0:
            wpm_agent = (agent_words / (agent_time / 60)) if agent_time > 0 else 0
            wpm_customer = (customer_words / (customer_time / 60)) if customer_time > 0 else 0
        else:
            wpm_agent = 0
            wpm_customer = 0

        # Talk time ratio
        if total_speaking_time > 0:
            agent_ratio = (agent_time / total_speaking_time) * 100
            customer_ratio = (customer_time / total_speaking_time) * 100
        else:
            agent_ratio = 50
            customer_ratio = 50

        return {
            "total_words": total_words,
            "agent_words": agent_words,
            "customer_words": customer_words,
            "word_ratio": round(agent_words / customer_words, 2) if customer_words > 0 else 0,
            "agent_wpm": round(wpm_agent, 1),
            "customer_wpm": round(wpm_customer, 1),
            "agent_talk_time_percent": round(agent_ratio, 1),
            "customer_talk_time_percent": round(customer_ratio, 1),
            "agent_talk_time_seconds": round(agent_time, 1),
            "customer_talk_time_seconds": round(customer_time, 1),
        }

    def _detect_interruptions(self, segments: list) -> dict:
        """Detect interruptions in the conversation.

        Args:
            segments: List of transcript segments.

        Returns:
            Interruption detection results.
        """
        interruptions = 0
        overlaps = 0

        # Simple heuristic: if segments overlap in time
        for i in range(1, len(segments)):
            prev_seg = segments[i - 1]
            curr_seg = segments[i]

            prev_end = prev_seg.get("end", 0)
            curr_start = curr_seg.get("start", 0)

            # Check for overlap (interruption)
            if curr_start < prev_end:
                overlaps += 1
                # If agent interrupted customer
                if prev_seg.get("speaker") == "Zákazník" and curr_seg.get("speaker") == "Agent":
                    interruptions += 1

        # Score: 10 = no interruptions, 0 = many interruptions
        score = max(0, 10 - interruptions)

        return {
            "agent_interruptions": interruptions,
            "total_overlaps": overlaps,
            "score": score,
            "rating": self._get_rating(score),
        }

    def _analyze_empathy_professionalism(self, transcript: str) -> dict:
        """Analyze empathy and professionalism of the agent.

        Args:
            transcript: Formatted transcript.

        Returns:
            Empathy and professionalism scores.
        """
        prompt = f"""Analyzuj empatiu a profesionalitu agenta v nasledujúcom prepise hovoru z call centra.

Poskyni:
1. empathy_score: číslo 0-10 (0 = žiadna empatia, 10 = výnimočná empatia)
2. professionalism_score: číslo 0-10 (0 = neprofesionálne, 10 = vysoko profesionálne)
3. empathy_phrases: zoznam empatických fráz použitých agentom (max 5)
4. professionalism_phrases: zoznam profesionálnych fráz (max 5)
5. improvement_suggestions: zoznam návrhov na zlepšenie (max 3)

Odpovedz VÝHRADNE vo formáte JSON bez markdown formátovania:
{{"empathy_score": 0, "professionalism_score": 0, "empathy_phrases": [], "professionalism_phrases": [], "improvement_suggestions": []}}

Prepis:
{transcript}"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            result = self._parse_json_response(response.text)

            # Add ratings
            result["empathy_rating"] = self._get_rating(result.get("empathy_score", 5))
            result["professionalism_rating"] = self._get_rating(
                result.get("professionalism_score", 5)
            )

            return result
        except Exception as e:
            return {
                "empathy_score": 5,
                "professionalism_score": 5,
                "empathy_phrases": [],
                "professionalism_phrases": [],
                "improvement_suggestions": [f"Analýza zlyhala: {str(e)}"],
                "empathy_rating": "Priemerné",
                "professionalism_rating": "Priemerné",
                "error": True
            }

    def _calculate_overall_score(self, metrics: dict) -> dict:
        """Calculate overall quality score from all metrics.

        Args:
            metrics: Dictionary of all calculated metrics.

        Returns:
            Overall score and rating.
        """
        scores = []

        # Sentiment (convert -1 to 1 scale to 0-10)
        sentiment_score = metrics.get("sentiment", {}).get("sentiment_score", 0)
        sentiment_normalized = (sentiment_score + 1) * 5  # -1 to 1 -> 0 to 10
        scores.append(sentiment_normalized)

        # FCR (confidence as partial score)
        fcr = metrics.get("fcr", {})
        fcr_status = fcr.get("fcr_status", "Uncertain")
        fcr_confidence = fcr.get("confidence", 50) / 100
        if fcr_status == "Yes":
            fcr_score = 10 * fcr_confidence
        elif fcr_status == "No":
            fcr_score = 3 * fcr_confidence
        else:
            fcr_score = 5
        scores.append(fcr_score)

        # Interruptions
        interruption_score = metrics.get("interruptions", {}).get("score", 5)
        scores.append(interruption_score)

        # Empathy
        empathy_score = metrics.get("empathy_professionalism", {}).get("empathy_score", 5)
        scores.append(empathy_score)

        # Professionalism
        prof_score = metrics.get("empathy_professionalism", {}).get("professionalism_score", 5)
        scores.append(prof_score)

        # Calculate weighted average
        weights = [1.5, 2.0, 1.0, 1.5, 1.5]  # FCR weighted highest
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        total_weight = sum(weights)
        overall = weighted_sum / total_weight

        return {
            "score": round(overall, 1),
            "rating": self._get_rating(overall),
            "max_score": 10,
            "components": {
                "sentiment": round(sentiment_normalized, 1),
                "fcr": round(fcr_score, 1),
                "interruptions": interruption_score,
                "empathy": empathy_score,
                "professionalism": prof_score,
            }
        }

    def _get_rating(self, score: float) -> str:
        """Convert numeric score to rating label.

        Args:
            score: Numeric score (0-10).

        Returns:
            Rating label.
        """
        if score >= 8:
            return "Vynikajúce"
        elif score >= 5:
            return "Priemerné"
        else:
            return "Potrebuje zlepšenie"

    def _format_duration(self, seconds: float) -> str:
        """Format seconds to MM:SS format.

        Args:
            seconds: Duration in seconds.

        Returns:
            Formatted duration string.
        """
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def _parse_json_response(self, response_text: str) -> dict:
        """Parse JSON from Gemini response.

        Args:
            response_text: Raw response text from Gemini.

        Returns:
            Parsed JSON dictionary.
        """
        # Try to extract JSON from response
        text = response_text.strip()

        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        # Find JSON object
        start = text.find("{")
        end = text.rfind("}") + 1

        if start != -1 and end > start:
            json_str = text[start:end]
            return json.loads(json_str)

        raise ValueError(f"Could not parse JSON from response: {response_text[:200]}")
