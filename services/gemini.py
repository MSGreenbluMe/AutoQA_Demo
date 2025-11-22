"""
Google Gemini Analysis Service

This module handles AI-powered call analysis using Google Gemini API
to calculate quality metrics for call center conversations.
"""

import json
import re
from typing import Optional

import google.generativeai as genai

from utils.config import get_api_key


class GeminiAnalyzer:
    """Handle call quality analysis via Google Gemini API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Gemini analyzer."""
        self.api_key = api_key or get_api_key("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found. Set it in .env or Streamlit secrets.")

        genai.configure(api_key=self.api_key)
        # Gemini 2.5 Flash - 10 RPM free tier, best price-performance
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        self.generation_config = {
            "temperature": 0.3,
            "max_output_tokens": 4096,
            "response_mime_type": "application/json",  # Force valid JSON output
        }

    def analyze_call(self, transcript: dict) -> dict:
        """Perform comprehensive call analysis with SINGLE API call."""
        segments = transcript.get("segments", [])
        duration = transcript.get("duration", 0)

        # Build formatted transcript
        formatted_transcript = self._format_transcript_for_analysis(segments)

        # Calculate local metrics (no API needed)
        speech_metrics = self._calculate_speech_metrics(segments, duration)
        interruptions = self._detect_interruptions(segments)

        # Single combined API call for all AI analysis
        ai_metrics = self._analyze_all_metrics(formatted_transcript)

        # Build final metrics
        metrics = {
            "sentiment": ai_metrics.get("sentiment", self._default_sentiment()),
            "fcr": ai_metrics.get("fcr", self._default_fcr()),
            "speech": speech_metrics,
            "interruptions": interruptions,
            "empathy_professionalism": ai_metrics.get("empathy_professionalism", self._default_empathy()),
            "duration": {
                "total_seconds": duration,
                "formatted": self._format_duration(duration),
            },
        }

        # Calculate overall score
        metrics["overall_score"] = self._calculate_overall_score(metrics)

        return metrics

    def _analyze_all_metrics(self, transcript: str) -> dict:
        """Single API call to analyze all metrics at once."""
        prompt = f"""Analyzuj nasledujúci prepis hovoru z call centra a poskytni kompletnú analýzu.

Vráť JSON s presne touto štruktúrou (bez markdown formátovania):
{{
    "sentiment": {{
        "overall_sentiment": "Positive" alebo "Neutral" alebo "Negative",
        "sentiment_score": číslo od -1.0 do 1.0,
        "sentiment_start": "Positive" alebo "Neutral" alebo "Negative",
        "sentiment_end": "Positive" alebo "Neutral" alebo "Negative",
        "explanation": "stručné vysvetlenie v slovenčine (max 2 vety)"
    }},
    "fcr": {{
        "fcr_status": "Yes" alebo "No" alebo "Uncertain",
        "confidence": číslo 0-100,
        "reasoning": "stručné zdôvodnenie v slovenčine (max 2 vety)"
    }},
    "empathy_professionalism": {{
        "empathy_score": číslo 0-10,
        "professionalism_score": číslo 0-10,
        "empathy_phrases": ["max 5 empatických fráz agenta"],
        "professionalism_phrases": ["max 5 profesionálnych fráz"],
        "improvement_suggestions": ["max 3 návrhy na zlepšenie"]
    }}
}}

Prepis hovoru:
{transcript}"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            result = self._parse_json_response(response.text)

            # Add ratings to empathy_professionalism
            if "empathy_professionalism" in result:
                ep = result["empathy_professionalism"]
                ep["empathy_rating"] = self._get_rating(ep.get("empathy_score", 5))
                ep["professionalism_rating"] = self._get_rating(ep.get("professionalism_score", 5))

            return result
        except Exception as e:
            return {
                "sentiment": self._default_sentiment(str(e)),
                "fcr": self._default_fcr(str(e)),
                "empathy_professionalism": self._default_empathy(str(e)),
            }

    def _default_sentiment(self, error: str = None) -> dict:
        return {
            "overall_sentiment": "Neutral",
            "sentiment_score": 0.0,
            "sentiment_start": "Neutral",
            "sentiment_end": "Neutral",
            "explanation": f"Analýza zlyhala: {error}" if error else "Predvolené hodnoty",
            "error": bool(error)
        }

    def _default_fcr(self, error: str = None) -> dict:
        return {
            "fcr_status": "Uncertain",
            "confidence": 0,
            "reasoning": f"Analýza zlyhala: {error}" if error else "Predvolené hodnoty",
            "error": bool(error)
        }

    def _default_empathy(self, error: str = None) -> dict:
        return {
            "empathy_score": 5,
            "professionalism_score": 5,
            "empathy_phrases": [],
            "professionalism_phrases": [],
            "improvement_suggestions": [f"Analýza zlyhala: {error}"] if error else [],
            "empathy_rating": "Priemerné",
            "professionalism_rating": "Priemerné",
            "error": bool(error)
        }

    def _format_transcript_for_analysis(self, segments: list) -> str:
        """Format transcript segments for analysis."""
        lines = []
        for seg in segments:
            speaker = seg.get("speaker", "Unknown")
            text = seg.get("text", "")
            start = seg.get("start", 0)
            timestamp = self._format_duration(start)
            lines.append(f"[{timestamp}] {speaker}: {text}")
        return "\n".join(lines)

    def _calculate_speech_metrics(self, segments: list, duration: float) -> dict:
        """Calculate speech-related metrics from segments (local, no API)."""
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

        if duration > 0:
            wpm_agent = (agent_words / (agent_time / 60)) if agent_time > 0 else 0
            wpm_customer = (customer_words / (customer_time / 60)) if customer_time > 0 else 0
        else:
            wpm_agent = 0
            wpm_customer = 0

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
        """Detect interruptions in the conversation (local, no API)."""
        interruptions = 0
        overlaps = 0

        for i in range(1, len(segments)):
            prev_seg = segments[i - 1]
            curr_seg = segments[i]
            prev_end = prev_seg.get("end", 0)
            curr_start = curr_seg.get("start", 0)

            if curr_start < prev_end:
                overlaps += 1
                if prev_seg.get("speaker") == "Zákazník" and curr_seg.get("speaker") == "Agent":
                    interruptions += 1

        score = max(0, 10 - interruptions)

        return {
            "agent_interruptions": interruptions,
            "total_overlaps": overlaps,
            "score": score,
            "rating": self._get_rating(score),
        }

    def _calculate_overall_score(self, metrics: dict) -> dict:
        """Calculate overall quality score from all metrics."""
        scores = []

        sentiment_score = metrics.get("sentiment", {}).get("sentiment_score", 0)
        sentiment_normalized = (sentiment_score + 1) * 5
        scores.append(sentiment_normalized)

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

        interruption_score = metrics.get("interruptions", {}).get("score", 5)
        scores.append(interruption_score)

        empathy_score = metrics.get("empathy_professionalism", {}).get("empathy_score", 5)
        scores.append(empathy_score)

        prof_score = metrics.get("empathy_professionalism", {}).get("professionalism_score", 5)
        scores.append(prof_score)

        weights = [1.5, 2.0, 1.0, 1.5, 1.5]
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
        """Convert numeric score to rating label."""
        if score >= 8:
            return "Vynikajúce"
        elif score >= 5:
            return "Priemerné"
        else:
            return "Potrebuje zlepšenie"

    def _format_duration(self, seconds: float) -> str:
        """Format seconds to MM:SS format."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def _parse_json_response(self, response_text: str) -> dict:
        """Parse JSON from Gemini response with robust error handling."""
        text = response_text.strip()

        # Remove markdown code blocks
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # Find JSON boundaries
        start = text.find("{")
        end = text.rfind("}") + 1

        if start == -1 or end <= start:
            raise ValueError(f"No JSON object found in response: {response_text[:200]}")

        json_str = text[start:end]

        # Try direct parsing first
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # Clean up common JSON issues
        cleaned = json_str

        # Remove trailing commas before } or ]
        cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)

        # Remove any control characters except newlines and tabs
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', cleaned)

        # Try parsing cleaned JSON
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Last resort: try to extract individual fields manually
        try:
            result = self._extract_fields_manually(json_str)
            if result:
                return result
        except Exception:
            pass

        raise ValueError(f"Could not parse JSON from response: {response_text[:300]}")

    def _extract_fields_manually(self, json_str: str) -> dict:
        """Extract fields manually if JSON parsing fails."""
        result = {}

        # Try to extract sentiment block
        sentiment_match = re.search(
            r'"sentiment"\s*:\s*(\{[^}]+\})',
            json_str, re.DOTALL
        )
        if sentiment_match:
            try:
                # Clean and parse
                block = sentiment_match.group(1)
                block = re.sub(r',\s*\}', '}', block)
                result["sentiment"] = json.loads(block)
            except Exception:
                pass

        # Try to extract fcr block
        fcr_match = re.search(
            r'"fcr"\s*:\s*(\{[^}]+\})',
            json_str, re.DOTALL
        )
        if fcr_match:
            try:
                block = fcr_match.group(1)
                block = re.sub(r',\s*\}', '}', block)
                result["fcr"] = json.loads(block)
            except Exception:
                pass

        # Try to extract empathy_professionalism block (more complex with arrays)
        ep_match = re.search(
            r'"empathy_professionalism"\s*:\s*(\{.*?\})\s*\}',
            json_str, re.DOTALL
        )
        if ep_match:
            try:
                block = ep_match.group(1) + "}"
                block = re.sub(r',\s*\}', '}', block)
                block = re.sub(r',\s*\]', ']', block)
                result["empathy_professionalism"] = json.loads(block)
            except Exception:
                pass

        return result if result else None
