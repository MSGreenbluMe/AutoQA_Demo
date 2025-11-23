"""
Gladia.io Transcription Service

This module handles audio transcription using Gladia.io API
with speaker diarization and timestamps.
"""

import time
import requests
from typing import Optional

from utils.config import get_api_key


class GladiaTranscriber:
    """Handle audio transcription via Gladia.io API."""

    BASE_URL = "https://api.gladia.io/v2"

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Gladia transcriber.

        Args:
            api_key: Gladia API key. If not provided, reads from config.
        """
        self.api_key = api_key or get_api_key("GLADIA_API_KEY")
        if not self.api_key:
            raise ValueError("GLADIA_API_KEY not found. Set it in .env or Streamlit secrets.")

        self.headers = {
            "x-gladia-key": self.api_key,
        }

    def upload_audio(self, audio_file) -> str:
        """Upload an audio file to Gladia.

        Args:
            audio_file: File object or path to audio file.

        Returns:
            URL of the uploaded audio file.
        """
        upload_url = f"{self.BASE_URL}/upload"

        if isinstance(audio_file, str):
            with open(audio_file, "rb") as f:
                files = {"audio": f}
                response = requests.post(
                    upload_url,
                    headers=self.headers,
                    files=files
                )
        else:
            files = {"audio": (audio_file.name, audio_file, audio_file.type)}
            response = requests.post(
                upload_url,
                headers=self.headers,
                files=files
            )

        response.raise_for_status()
        return response.json()["audio_url"]

    def transcribe(
        self,
        audio_url: str,
        language: str = "sk",
        enable_diarization: bool = True,
        progress_callback=None
    ) -> dict:
        """Start transcription and wait for results.

        Args:
            audio_url: URL of the audio file to transcribe.
            language: Language code (sk, cs, en).
            enable_diarization: Enable speaker diarization.
            progress_callback: Optional callback for progress updates.

        Returns:
            Transcription result dictionary.
        """
        # Start transcription
        transcription_url = f"{self.BASE_URL}/transcription"

        payload = {
            "audio_url": audio_url,
            "diarization": enable_diarization,
            "diarization_config": {
                "number_of_speakers": 2,
                "min_speakers": 2,
                "max_speakers": 2,
            }
        }

        # Add language hint if not auto-detect
        if language and language != "auto":
            payload["language"] = language

        response = requests.post(
            transcription_url,
            headers={**self.headers, "Content-Type": "application/json"},
            json=payload
        )
        response.raise_for_status()

        result = response.json()
        result_url = result.get("result_url")

        if not result_url:
            raise ValueError("No result_url returned from Gladia")

        # Poll for results
        return self._poll_for_results(result_url, progress_callback)

    def _poll_for_results(self, result_url: str, progress_callback=None) -> dict:
        """Poll Gladia API until transcription is complete.

        Args:
            result_url: URL to poll for results.
            progress_callback: Optional callback for progress updates.

        Returns:
            Transcription result dictionary.
        """
        max_attempts = 120  # 10 minutes max
        attempt = 0

        while attempt < max_attempts:
            response = requests.get(result_url, headers=self.headers)
            response.raise_for_status()

            result = response.json()
            status = result.get("status", "unknown")

            if progress_callback:
                progress = result.get("progress", 0)
                progress_callback(status, progress)

            if status == "done":
                return result
            elif status == "error":
                error_msg = result.get("error_message", "Unknown error")
                raise RuntimeError(f"Transcription failed: {error_msg}")

            # Wait before next poll
            time.sleep(5)
            attempt += 1

        raise TimeoutError("Transcription timed out after 10 minutes")

    def process_audio(
        self,
        audio_file,
        language: str = "sk",
        progress_callback=None
    ) -> dict:
        """Full pipeline: upload audio and get transcription.

        Args:
            audio_file: File object or path to audio file.
            language: Language code.
            progress_callback: Optional callback for progress updates.

        Returns:
            Processed transcription with speaker segments.
        """
        # Upload
        if progress_callback:
            progress_callback("uploading", 0)

        audio_url = self.upload_audio(audio_file)

        if progress_callback:
            progress_callback("uploaded", 10)

        # Transcribe
        result = self.transcribe(
            audio_url,
            language=language,
            enable_diarization=True,
            progress_callback=progress_callback
        )

        # Process result
        return self._process_transcript(result)

    def _process_transcript(self, raw_result: dict) -> dict:
        """Process raw Gladia result into structured format.

        Args:
            raw_result: Raw Gladia API response.

        Returns:
            Structured transcript with segments.
        """
        transcript_result = raw_result.get("result", {})
        transcription = transcript_result.get("transcription", {})

        # Get full text
        full_text = transcription.get("full_transcript", "")

        # Get utterances with speaker info
        utterances = transcription.get("utterances", [])

        segments = []
        for utterance in utterances:
            speaker_id = utterance.get("speaker", 0)
            # In typical calls, speaker 0 is often the customer (first to speak)
            # and speaker 1 is the agent - swap the convention
            speaker_role = "Zákazník" if speaker_id == 0 else "Agent"

            segment = {
                "speaker": speaker_role,
                "speaker_id": speaker_id,
                "text": utterance.get("text", ""),
                "start": utterance.get("start", 0),
                "end": utterance.get("end", 0),
                "confidence": utterance.get("confidence", 0),
            }
            segments.append(segment)

        # Calculate duration
        duration = 0
        if segments:
            duration = max(seg["end"] for seg in segments)

        return {
            "full_text": full_text,
            "segments": segments,
            "duration": duration,
            "language": transcription.get("language", "unknown"),
            "speaker_count": len(set(s["speaker_id"] for s in segments)),
        }


def format_timestamp(seconds: float) -> str:
    """Format seconds into MM:SS format.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted time string.
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"
