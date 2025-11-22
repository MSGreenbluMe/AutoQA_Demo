"""
AutoQA - AI Quality Assurance for Call Centers

Main Streamlit application for automated call quality analysis.
Developed by Coworkers.ai
"""

import os
import streamlit as st
from dotenv import load_dotenv

from services.gladia import GladiaTranscriber, format_timestamp
from services.gemini import GeminiAnalyzer
from utils.audio_utils import (
    validate_audio_file,
    get_file_info,
    format_file_size,
    format_duration,
    get_language_name,
    SUPPORTED_FORMATS,
    MAX_FILE_SIZE_MB,
)
from components.dashboard import (
    render_metric_card,
    render_gauge_chart,
    render_sentiment_chart,
    render_word_count_chart,
    render_talk_time_pie,
    render_overall_score_card,
    render_fcr_card,
    render_component_scores,
    render_phrases_list,
    render_transcript_message,
    COLORS,
)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AutoQA - AI Quality Assurance",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #212121;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "transcript" not in st.session_state:
        st.session_state.transcript = None
    if "metrics" not in st.session_state:
        st.session_state.metrics = None
    if "processing_step" not in st.session_state:
        st.session_state.processing_step = None
    if "error" not in st.session_state:
        st.session_state.error = None


def render_header():
    """Render application header."""
    col1, col2 = st.columns([4, 1])

    with col1:
        st.markdown("""
        # 🎯 AutoQA - AI Quality Assurance
        **Automatizované hodnotenie kvality hovorov** | Powered by Coworkers.ai
        """)

    with col2:
        st.markdown("""
        <div style="text-align: right; padding-top: 1rem;">
            <img src="https://coworkers.ai/wp-content/uploads/2023/07/coworkers-ai-logo.svg"
                 alt="Coworkers.ai"
                 style="max-height: 40px; opacity: 0.8;">
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")


def render_upload_section():
    """Render file upload section."""
    st.markdown("### 📁 Nahrať audio súbor")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Vyberte audio súbor (WAV, MP3, M4A, FLAC)",
            type=["wav", "mp3", "m4a", "flac"],
            help=f"Maximálna veľkosť súboru: {MAX_FILE_SIZE_MB} MB",
        )

    with col2:
        language = st.selectbox(
            "Jazyk hovoru",
            options=["sk", "cs", "en", "auto"],
            format_func=get_language_name,
            index=0,
        )

    if uploaded_file:
        # Validate file
        is_valid, error_msg = validate_audio_file(uploaded_file)

        if not is_valid:
            st.error(f"❌ {error_msg}")
            return None, None

        # Show file info
        file_info = get_file_info(uploaded_file)
        st.success(
            f"✅ Súbor nahraný: **{file_info['name']}** "
            f"({format_file_size(file_info['size_bytes'])})"
        )

        return uploaded_file, language

    return None, None


def render_processing_status(step: str, progress: float = 0):
    """Render processing status indicator.

    Args:
        step: Current processing step.
        progress: Progress percentage (0-100).
    """
    steps = {
        "uploading": ("📤 Nahrávanie", 10),
        "uploaded": ("✅ Nahraté", 20),
        "transcribing": ("🎙️ Prepis", 50),
        "queued": ("⏳ V rade", 30),
        "processing": ("⚙️ Spracovanie", 60),
        "done": ("✅ Prepis hotový", 80),
        "analyzing": ("🤖 AI Analýza", 90),
        "complete": ("✅ Dokončené", 100),
    }

    display_name, default_progress = steps.get(step, (step, progress))
    actual_progress = progress if progress > 0 else default_progress

    st.markdown(f"**Stav:** {display_name}")
    st.progress(actual_progress / 100)


def process_audio(uploaded_file, language: str):
    """Process uploaded audio file.

    Args:
        uploaded_file: Streamlit UploadedFile object.
        language: Language code.
    """
    status_container = st.empty()
    progress_container = st.empty()

    try:
        # Initialize services
        with status_container.container():
            st.info("🔄 Inicializácia služieb...")

        gladia = GladiaTranscriber()
        gemini = GeminiAnalyzer()

        # Progress callback
        def update_progress(status, progress):
            with progress_container.container():
                render_processing_status(status, progress)

        # Transcription
        with status_container.container():
            st.info("🎙️ Prebieha prepis hovoru...")

        update_progress("transcribing", 30)

        transcript = gladia.process_audio(
            uploaded_file,
            language=language,
            progress_callback=update_progress
        )

        st.session_state.transcript = transcript

        # Analysis
        with status_container.container():
            st.info("🤖 Prebieha AI analýza...")

        update_progress("analyzing", 85)

        metrics = gemini.analyze_call(transcript)
        st.session_state.metrics = metrics

        update_progress("complete", 100)

        with status_container.container():
            st.success("✅ Analýza dokončená!")

    except ValueError as e:
        st.session_state.error = str(e)
        with status_container.container():
            st.error(f"❌ Chyba konfigurácie: {e}")
    except RuntimeError as e:
        st.session_state.error = str(e)
        with status_container.container():
            st.error(f"❌ Chyba spracovania: {e}")
    except Exception as e:
        st.session_state.error = str(e)
        with status_container.container():
            st.error(f"❌ Neočakávaná chyba: {e}")


def render_transcript_section(transcript: dict):
    """Render transcript visualization.

    Args:
        transcript: Processed transcript dictionary.
    """
    st.markdown("### 📝 Prepis hovoru")

    # Transcript info
    col1, col2, col3 = st.columns(3)

    with col1:
        duration = transcript.get("duration", 0)
        st.metric("Trvanie", format_duration(duration))

    with col2:
        lang = transcript.get("language", "unknown")
        st.metric("Jazyk", get_language_name(lang))

    with col3:
        speakers = transcript.get("speaker_count", 2)
        st.metric("Účastníci", speakers)

    st.markdown("---")

    # Chat-like transcript view
    segments = transcript.get("segments", [])

    if not segments:
        st.warning("Prepis neobsahuje žiadne segmenty.")
        return

    # Scrollable container
    with st.container():
        for segment in segments:
            render_transcript_message(
                speaker=segment.get("speaker", "Unknown"),
                text=segment.get("text", ""),
                timestamp=format_timestamp(segment.get("start", 0)),
                speaker_id=segment.get("speaker_id", 0),
            )

    # Export options
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        # Export as text
        full_text = transcript.get("full_text", "")
        st.download_button(
            label="📄 Stiahnuť prepis (TXT)",
            data=full_text,
            file_name="transcript.txt",
            mime="text/plain",
        )

    with col2:
        # Export as JSON
        import json
        json_data = json.dumps(transcript, ensure_ascii=False, indent=2)
        st.download_button(
            label="📋 Stiahnuť prepis (JSON)",
            data=json_data,
            file_name="transcript.json",
            mime="application/json",
        )


def render_metrics_dashboard(metrics: dict):
    """Render metrics dashboard.

    Args:
        metrics: Calculated metrics dictionary.
    """
    st.markdown("### 📊 Dashboard kvality hovoru")

    # Overall score at the top
    render_overall_score_card(metrics.get("overall_score", {}))

    st.markdown("---")

    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        fcr = metrics.get("fcr", {})
        render_fcr_card(fcr)

    with col2:
        sentiment = metrics.get("sentiment", {})
        st.plotly_chart(
            render_sentiment_chart(sentiment),
            use_container_width=True,
            key="sentiment_chart"
        )

    with col3:
        empathy_prof = metrics.get("empathy_professionalism", {})
        empathy_score = empathy_prof.get("empathy_score", 5)
        st.plotly_chart(
            render_gauge_chart(empathy_score, "Empatia"),
            use_container_width=True,
            key="empathy_gauge"
        )

    with col4:
        prof_score = empathy_prof.get("professionalism_score", 5)
        st.plotly_chart(
            render_gauge_chart(prof_score, "Profesionalita"),
            use_container_width=True,
            key="prof_gauge"
        )

    st.markdown("---")

    # Speech metrics row
    col1, col2 = st.columns(2)

    with col1:
        speech = metrics.get("speech", {})
        st.plotly_chart(
            render_word_count_chart(speech),
            use_container_width=True,
            key="word_chart"
        )

    with col2:
        st.plotly_chart(
            render_talk_time_pie(speech),
            use_container_width=True,
            key="talk_time_chart"
        )

    st.markdown("---")

    # Detailed metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🎯 Skóre komponentov")
        components = metrics.get("overall_score", {}).get("components", {})
        render_component_scores(components)

    with col2:
        st.markdown("#### 📊 Štatistiky reči")
        speech = metrics.get("speech", {})

        render_metric_card(
            "Celkový počet slov",
            str(speech.get("total_words", 0)),
            f"Agent: {speech.get('agent_words', 0)} | Zákazník: {speech.get('customer_words', 0)}",
            icon="📝"
        )

        render_metric_card(
            "Rýchlosť reči (slov/min)",
            f"Agent: {speech.get('agent_wpm', 0)}",
            f"Zákazník: {speech.get('customer_wpm', 0)} slov/min",
            icon="⏱️"
        )

        interruptions = metrics.get("interruptions", {})
        render_metric_card(
            "Prerušenia",
            str(interruptions.get("agent_interruptions", 0)),
            f"Skóre: {interruptions.get('score', 10)}/10",
            score=interruptions.get("score", 10),
            icon="🔇"
        )

    with col3:
        st.markdown("#### 💬 Detekované frázy")

        empathy_prof = metrics.get("empathy_professionalism", {})

        render_phrases_list(
            empathy_prof.get("empathy_phrases", []),
            "Empatické frázy",
            COLORS["success"]
        )

        st.markdown("")

        render_phrases_list(
            empathy_prof.get("professionalism_phrases", []),
            "Profesionálne frázy",
            COLORS["primary"]
        )

    st.markdown("---")

    # Improvement suggestions
    suggestions = empathy_prof.get("improvement_suggestions", [])
    if suggestions:
        st.markdown("#### 💡 Návrhy na zlepšenie")
        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")

    # Duration info
    duration_info = metrics.get("duration", {})
    st.markdown("---")
    st.markdown(
        f"**Celková dĺžka hovoru:** {duration_info.get('formatted', 'N/A')}"
    )


def render_demo_mode():
    """Render demo mode with sample data."""
    st.info(
        "💡 **Demo mód:** Pre plnú funkcionalitu nastavte API kľúče "
        "v súbore `.env` (GLADIA_API_KEY a GEMINI_API_KEY)"
    )

    # Sample transcript
    sample_transcript = {
        "full_text": "Agent: Dobrý deň, ste spojený so zákazníckou podporou. Zákazník: Dobrý deň, mám problém s faktúrou...",
        "segments": [
            {
                "speaker": "Agent",
                "speaker_id": 0,
                "text": "Dobrý deň, ste spojený so zákazníckou podporou. Ako vám môžem pomôcť?",
                "start": 0.0,
                "end": 4.5,
            },
            {
                "speaker": "Zákazník",
                "speaker_id": 1,
                "text": "Dobrý deň, mám problém s faktúrou. Prišla mi vyššia suma ako obvykle.",
                "start": 5.0,
                "end": 10.2,
            },
            {
                "speaker": "Agent",
                "speaker_id": 0,
                "text": "Rozumiem, to musí byť frustrujúce. Dovoľte mi skontrolovať váš účet.",
                "start": 10.5,
                "end": 15.0,
            },
            {
                "speaker": "Zákazník",
                "speaker_id": 1,
                "text": "Ďakujem, veľmi si to vážim.",
                "start": 15.5,
                "end": 18.0,
            },
            {
                "speaker": "Agent",
                "speaker_id": 0,
                "text": "Vidím, že máte aktivovanú novú službu. To spôsobilo navýšenie. Môžem vám ju zrušiť?",
                "start": 18.5,
                "end": 25.0,
            },
            {
                "speaker": "Zákazník",
                "speaker_id": 1,
                "text": "Áno, prosím, zrušte ju. Nevedel som, že je aktívna.",
                "start": 25.5,
                "end": 30.0,
            },
            {
                "speaker": "Agent",
                "speaker_id": 0,
                "text": "Hotovo, služba je zrušená a v nasledujúcej faktúre bude suma opravená. Je ešte niečo, s čím vám môžem pomôcť?",
                "start": 30.5,
                "end": 38.0,
            },
            {
                "speaker": "Zákazník",
                "speaker_id": 1,
                "text": "Nie, to je všetko. Ďakujem veľmi pekne za rýchle vyriešenie!",
                "start": 38.5,
                "end": 43.0,
            },
            {
                "speaker": "Agent",
                "speaker_id": 0,
                "text": "Nemáte za čo. Prajem pekný deň!",
                "start": 43.5,
                "end": 46.0,
            },
        ],
        "duration": 46.0,
        "language": "sk",
        "speaker_count": 2,
    }

    # Sample metrics
    sample_metrics = {
        "sentiment": {
            "overall_sentiment": "Positive",
            "sentiment_score": 0.72,
            "sentiment_start": "Neutral",
            "sentiment_end": "Positive",
            "explanation": "Hovor začal neutrálne, ale zákazník bol na konci veľmi spokojný s rýchlym vyriešením problému.",
        },
        "fcr": {
            "fcr_status": "Yes",
            "confidence": 95,
            "reasoning": "Problém s faktúrou bol vyriešený počas hovoru - služba bola zrušená a zákazník bol informovaný o oprave.",
        },
        "speech": {
            "total_words": 98,
            "agent_words": 62,
            "customer_words": 36,
            "word_ratio": 1.72,
            "agent_wpm": 145,
            "customer_wpm": 120,
            "agent_talk_time_percent": 58,
            "customer_talk_time_percent": 42,
            "agent_talk_time_seconds": 26.5,
            "customer_talk_time_seconds": 19.5,
        },
        "interruptions": {
            "agent_interruptions": 0,
            "total_overlaps": 0,
            "score": 10,
            "rating": "Vynikajúce",
        },
        "empathy_professionalism": {
            "empathy_score": 9,
            "professionalism_score": 9,
            "empathy_phrases": [
                "Rozumiem, to musí byť frustrujúce",
                "Dovoľte mi skontrolovať",
            ],
            "professionalism_phrases": [
                "Dobrý deň",
                "Ako vám môžem pomôcť?",
                "Prajem pekný deň",
            ],
            "improvement_suggestions": [
                "Agent zvládol hovor výborne, pokračovať v rovnakom štýle",
            ],
            "empathy_rating": "Vynikajúce",
            "professionalism_rating": "Vynikajúce",
        },
        "duration": {
            "total_seconds": 46.0,
            "formatted": "00:46",
        },
        "overall_score": {
            "score": 9.2,
            "rating": "Vynikajúce",
            "max_score": 10,
            "components": {
                "sentiment": 8.6,
                "fcr": 9.5,
                "interruptions": 10,
                "empathy": 9,
                "professionalism": 9,
            },
        },
    }

    if st.button("🎬 Zobraziť demo analýzu", type="primary"):
        st.session_state.transcript = sample_transcript
        st.session_state.metrics = sample_metrics


def main():
    """Main application entry point."""
    init_session_state()
    render_header()

    # Check for API keys
    gladia_key = os.getenv("GLADIA_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if not gladia_key or not gemini_key:
        render_demo_mode()
    else:
        # File upload
        uploaded_file, language = render_upload_section()

        if uploaded_file:
            if st.button("🚀 Spustiť analýzu", type="primary"):
                st.session_state.transcript = None
                st.session_state.metrics = None
                process_audio(uploaded_file, language)

    st.markdown("---")

    # Display results if available
    if st.session_state.transcript:
        tab1, tab2 = st.tabs(["📝 Prepis", "📊 Dashboard"])

        with tab1:
            render_transcript_section(st.session_state.transcript)

        with tab2:
            if st.session_state.metrics:
                render_metrics_dashboard(st.session_state.metrics)
            else:
                st.info("Čakám na dokončenie analýzy...")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #888; font-size: 0.8rem;">
            AutoQA POC v1.0 | Developed by Coworkers.ai | 2025
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
