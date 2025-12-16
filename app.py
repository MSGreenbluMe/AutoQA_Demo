"""
AutoQA - AI Quality Assurance for Call Centers

Main Streamlit application for automated call quality analysis.
Developed by Coworkers.ai
"""

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
from utils.config import has_api_keys
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
    render_call_timeline,
    COLORS,
)

# Advanced Analytics imports
from services.erfc import ERFCAnalyzer
from services.ssr import SSRAnalyzer
from services.ccm import CCMAnalyzer, analyze_ssr_with_ccm
from components.advanced_viz import (
    create_emotion_timeline,
    create_trajectory_gauge,
    create_distribution_bars,
    create_confidence_table,
    create_overall_confidence_indicator,
)

# Load environment variables (for local development)
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AutoQA - AI Quality Assurance",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for dark professional theme
st.markdown("""
<style>
    /* Dark theme base */
    .stApp {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
    }
    .main {
        padding-top: 1rem;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #0EA5E9 0%, #6366F1 100%);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    h1, h2, h3 {
        color: #F8FAFC !important;
    }
    h1 {
        background: linear-gradient(90deg, #0EA5E9 0%, #6366F1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    p, span, label {
        color: #94A3B8;
    }
    .stAlert {
        border-radius: 12px;
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(148, 163, 184, 0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 12px;
        color: #94A3B8;
        border: 1px solid rgba(148, 163, 184, 0.1);
        padding: 0.75rem 1.5rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0EA5E9 0%, #6366F1 100%);
        color: white !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #0EA5E9 0%, #6366F1 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(14, 165, 233, 0.4);
    }
    .stFileUploader {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 16px;
        border: 2px dashed rgba(148, 163, 184, 0.2);
        padding: 1rem;
    }
    .stFileUploader:hover {
        border-color: #0EA5E9;
    }
    .stSelectbox > div > div {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.1);
    }
    .stMetric {
        background: rgba(30, 41, 59, 0.6);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.1);
    }
    .stMetric label {
        color: #94A3B8 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #F8FAFC !important;
    }
    hr {
        border-color: rgba(148, 163, 184, 0.1);
    }
    .stDownloadButton > button {
        background: rgba(30, 41, 59, 0.8);
        color: #F8FAFC;
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
    }
    .stDownloadButton > button:hover {
        background: rgba(14, 165, 233, 0.2);
        border-color: #0EA5E9;
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
    if "file_info" not in st.session_state:
        st.session_state.file_info = None
    if "audio_data" not in st.session_state:
        st.session_state.audio_data = None
    if "audio_format" not in st.session_state:
        st.session_state.audio_format = None
    if "swap_speakers" not in st.session_state:
        st.session_state.swap_speakers = False


def render_header():
    """Render application header."""
    col1, col2 = st.columns([4, 1])

    with col1:
        st.markdown("""
        # 🐙 AutoQA - AI Quality Assurance
        **Automatizované hodnotenie kvality hovorov** | Powered by Greenblu.me
        """)

    with col2:
        st.markdown("""
        <div style="text-align: right; padding-top: 0.5rem;">
            <span style="font-size: 2.5rem;">🐙</span>
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

        # Store and show file info
        file_info = get_file_info(uploaded_file)
        st.session_state.file_info = file_info

        # Store audio bytes immediately for playback
        # This ensures we capture the data before any rerun
        uploaded_file.seek(0)
        audio_bytes = uploaded_file.read()
        uploaded_file.seek(0)  # Reset for later use

        if audio_bytes and len(audio_bytes) > 0:
            st.session_state.audio_data = audio_bytes
            st.session_state.audio_format = uploaded_file.type or "audio/wav"

        st.success(
            f"✅ Súbor: **{file_info['name']}** ({format_file_size(file_info['size_bytes'])})"
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
    # Compact info row
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

    with col1:
        duration = transcript.get("duration", 0)
        st.metric("Trvanie", format_duration(duration))

    with col2:
        lang = transcript.get("language", "unknown")
        st.metric("Jazyk", get_language_name(lang))

    with col3:
        speakers = transcript.get("speaker_count", 2)
        st.metric("Účastníci", speakers)

    with col4:
        # Export as text
        full_text = transcript.get("full_text", "")
        st.download_button(
            label="📄 TXT",
            data=full_text,
            file_name="transcript.txt",
            mime="text/plain",
        )

    with col5:
        # Export as JSON
        import json
        json_data = json.dumps(transcript, ensure_ascii=False, indent=2)
        st.download_button(
            label="📋 JSON",
            data=json_data,
            file_name="transcript.json",
            mime="application/json",
        )

    # Audio player and speaker swap option
    col_audio, col_swap = st.columns([3, 1])

    with col_audio:
        if st.session_state.audio_data:
            st.markdown("#### 🎧 Prehrať nahrávku")
            # Use HTML5 audio with base64 for better compatibility
            import base64
            audio_b64 = base64.b64encode(st.session_state.audio_data).decode()
            # Detect format from file extension stored in session
            audio_mime = st.session_state.get("audio_format", "audio/mpeg")
            audio_html = f'''
            <audio controls style="width: 100%; border-radius: 8px;">
                <source src="data:{audio_mime};base64,{audio_b64}" type="{audio_mime}">
                Váš prehliadač nepodporuje audio prehrávač.
            </audio>
            '''
            st.markdown(audio_html, unsafe_allow_html=True)

    with col_swap:
        st.markdown("#### 🔄 Role")
        swap_label = "Agent ↔ Zákazník" if not st.session_state.swap_speakers else "Zákazník ↔ Agent"
        if st.button(f"🔄 Prehodiť", help="Prehodiť role Agenta a Zákazníka"):
            st.session_state.swap_speakers = not st.session_state.swap_speakers
            st.rerun()
        if st.session_state.swap_speakers:
            st.caption("⚠️ Role prehodené")

    st.markdown("---")

    # Chat-like transcript view in scrollable container
    segments = transcript.get("segments", [])

    if not segments:
        st.warning("Prepis neobsahuje žiadne segmenty.")
        return

    # Scrollable container with fixed height
    st.markdown("""
    <style>
    .transcript-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        background: rgba(15, 23, 42, 0.5);
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

    import html
    import re

    def sanitize_text(text: str) -> str:
        """Remove control characters and escape HTML."""
        # Remove control characters (except newlines and tabs)
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', str(text))
        # Escape HTML special characters
        return html.escape(cleaned)

    # Build transcript HTML
    swap = st.session_state.swap_speakers
    transcript_html = '<div class="transcript-container">'
    for segment in segments:
        speaker_raw = segment.get("speaker", "Unknown")
        text = sanitize_text(segment.get("text", ""))
        timestamp = format_timestamp(segment.get("start", 0))

        # Determine role with swap support
        is_agent_original = speaker_raw == "Agent"
        is_agent = not is_agent_original if swap else is_agent_original

        # Display name based on actual role (after swap)
        display_speaker = "Agent" if is_agent else "Zákazník"
        speaker = sanitize_text(display_speaker)

        color = "#0EA5E9" if is_agent else "#10B981"
        icon = "🎧" if is_agent else "👤"
        align = "flex-start" if is_agent else "flex-end"

        transcript_html += f'''<div style="display:flex;justify-content:{align};margin-bottom:0.75rem;"><div style="background:linear-gradient(135deg,rgba(30,41,59,0.8),rgba(15,23,42,0.9));border:1px solid {color}30;border-left:4px solid {color};padding:0.75rem 1rem;border-radius:12px;max-width:85%;"><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.25rem;"><span style="font-weight:600;color:{color};font-size:0.85rem;">{icon} {speaker}</span><span style="color:#94A3B8;font-size:0.7rem;margin-left:1rem;background:rgba(148,163,184,0.1);padding:0.15rem 0.5rem;border-radius:10px;">{timestamp}</span></div><div style="color:#F8FAFC;font-size:0.9rem;line-height:1.5;">{text}</div></div></div>'''
    transcript_html += '</div>'

    st.markdown(transcript_html, unsafe_allow_html=True)


def render_metrics_dashboard(metrics: dict, transcript: dict = None):
    """Render metrics dashboard.

    Args:
        metrics: Calculated metrics dictionary.
        transcript: Optional transcript dictionary for timeline.
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

    # Call Timeline - interactive visualization (below component scores)
    if transcript:
        st.markdown("---")
        segments = transcript.get("segments", [])
        duration = transcript.get("duration", 0)
        st.plotly_chart(
            render_call_timeline(segments, duration, st.session_state.swap_speakers),
            use_container_width=True,
            key="call_timeline"
        )

    st.markdown("---")

    # Improvement suggestions
    suggestions = empathy_prof.get("improvement_suggestions", [])
    if suggestions:
        st.markdown("#### 💡 Návrhy na zlepšenie")
        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")

    # Advanced Analytics expander
    with st.expander("🚀 Spustiť pokročilú analýzu (ERFC, SSR, CCM)", expanded=False):
        st.markdown("""
        Pokročilé analytické metódy založené na vedeckých publikáciách:
        - **ERFC**: Emócie a predikcia trajektórie
        - **SSR**: Distribúcie pravdepodobnosti (nie pevné skóre)
        - **CCM**: Istota a potreba ľudskej kontroly
        """)

        if st.button("▶️ Analyzovať", key="advanced_analytics_btn"):
            render_advanced_analytics(transcript, metrics)

    # Duration info
    duration_info = metrics.get("duration", {})
    st.markdown("---")
    st.markdown(
        f"**Celková dĺžka hovoru:** {duration_info.get('formatted', 'N/A')}"
    )


def render_advanced_analytics(transcript: dict, metrics: dict):
    """Render advanced analytics (ERFC, SSR, CCM).

    Args:
        transcript: Processed transcript dictionary.
        metrics: Basic metrics dictionary.
    """
    st.markdown("### 🚀 Pokročilá analýza (ERFC, SSR, CCM)")

    # Initialize analyzers
    from services.gemini import init_gemini

    try:
        gemini_model = init_gemini()
    except Exception as e:
        st.error(f"❌ Chyba inicializácie Gemini: {e}")
        return

    segments = transcript.get("segments", [])
    if not segments:
        st.warning("Nedostatok dát pre pokročilú analýzu.")
        return

    # Progress indicator
    progress_bar = st.progress(0)
    status = st.empty()

    # 1. ERFC Analysis
    status.markdown("🎭 Analyzujem emócie a trajektóriu...")
    progress_bar.progress(20)

    erfc = ERFCAnalyzer(gemini_model)
    erfc_result = erfc.analyze(segments, swap_speakers=st.session_state.swap_speakers)

    progress_bar.progress(50)

    # 2. SSR Analysis
    status.markdown("📊 Vykonávam SSR analýzu (distribúcie)...")

    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    except ImportError:
        st.info("💡 Pre lepšie výsledky nainštalujte: pip install sentence-transformers")
        embedder = None

    ssr = SSRAnalyzer(gemini_model, embedding_model=embedder)
    full_text = transcript.get("full_text", "")
    ssr_results = ssr.analyze_all(full_text)

    progress_bar.progress(75)

    # 3. CCM Analysis
    status.markdown("🎯 Analyzujem istotu pomocou CCM...")

    ccm = CCMAnalyzer(coverage_level=0.90)
    ccm_results = analyze_ssr_with_ccm(ssr_results, ccm)

    progress_bar.progress(100)
    status.markdown("✅ Pokročilá analýza dokončená!")

    st.markdown("---")

    # Visualizations
    col1, col2 = st.columns([2, 1])

    with col1:
        # ERFC: Emotion timeline
        fig_timeline = create_emotion_timeline(erfc_result)
        st.plotly_chart(fig_timeline, use_container_width=True, key="erfc_timeline")

    with col2:
        # ERFC: Agent consistency gauge
        fig_gauge = create_trajectory_gauge(
            erfc_result.agent_consistency,
            erfc_result.customer_trajectory
        )
        st.plotly_chart(fig_gauge, use_container_width=True, key="erfc_gauge")

        # Trajectory info
        traj_emoji = {
            "improving": "📈",
            "declining": "📉",
            "stable": "➡️",
            "volatile": "📊"
        }
        traj_label = {
            "improving": "Zlepšuje sa",
            "declining": "Zhoršuje sa",
            "stable": "Stabilná",
            "volatile": "Volatilná"
        }
        st.markdown(f"""
        **Trajektória zákazníka:**
        {traj_emoji.get(erfc_result.customer_trajectory, '➡️')} {traj_label.get(erfc_result.customer_trajectory, erfc_result.customer_trajectory)}
        """)

    st.markdown("---")

    # SSR: Distributions
    col1, col2 = st.columns([3, 2])

    with col1:
        fig_dist = create_distribution_bars(ssr_results)
        st.plotly_chart(fig_dist, use_container_width=True, key="ssr_distributions")

    with col2:
        st.markdown("#### 📝 SSR Vysvetlenia")
        for dimension, result in ssr_results.items():
            dim_labels = {
                "empathy": "Empatia",
                "professionalism": "Profesionalita",
                "fcr_likelihood": "FCR"
            }
            st.markdown(f"**{dim_labels.get(dimension, dimension)}:**")
            st.caption(result.text_response[:150] + "...")
            st.markdown("")

    st.markdown("---")

    # CCM: Confidence table
    fig_ccm = create_confidence_table(ccm_results)
    st.plotly_chart(fig_ccm, use_container_width=True, key="ccm_table")

    # Overall confidence
    overall_conf_pct, needs_review = create_overall_confidence_indicator(ccm_results)

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Celková istota analýzy",
            f"{overall_conf_pct:.0f}%",
            help="Percento dimenzií s vysokou istotou"
        )

    with col2:
        review_status = "⚠️ Áno" if needs_review else "✅ Nie"
        st.metric(
            "Vyžaduje ľudskú kontrolu?",
            review_status
        )

    # Intervention points
    if erfc_result.intervention_points:
        st.markdown("---")
        st.markdown("#### 💡 Intervenčné body")

        for ip in erfc_result.intervention_points:
            type_emoji = {
                "positive_impact": "✅",
                "negative_impact": "❌",
                "missed_opportunity": "⚠️"
            }
            type_color = {
                "positive_impact": COLORS["success"],
                "negative_impact": COLORS["error"],
                "missed_opportunity": COLORS["warning"]
            }

            emoji = type_emoji.get(ip.intervention_type, "•")
            color = type_color.get(ip.intervention_type, COLORS["text_muted"])

            st.markdown(
                f"<div style='padding:0.5rem;border-left:3px solid {color};background:rgba(30,41,59,0.5);margin-bottom:0.5rem;border-radius:8px;'>"
                f"<strong>{emoji} Turn {ip.turn_index}</strong>: {ip.description} "
                f"<span style='color:{COLORS['text_muted']};font-size:0.85rem;'>(zmena valence: {ip.customer_change:+.2f})</span>"
                f"</div>",
                unsafe_allow_html=True
            )


def render_demo_mode():
    """Render demo mode with sample data."""
    st.info(
        "💡 **Demo mód:** Pre plnú funkcionalitu nastavte API kľúče "
        "v `.env` súbore alebo Streamlit Secrets (GLADIA_API_KEY a GEMINI_API_KEY)"
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

    # Check for API keys (supports both .env and Streamlit secrets)
    if not has_api_keys():
        render_demo_mode()
    else:
        # If analysis complete, show results first
        if st.session_state.transcript and st.session_state.metrics:
            # Dashboard is PRIMARY - shown first
            tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📝 Prepis", "⚙️ Nová analýza"])

            with tab1:
                render_metrics_dashboard(st.session_state.metrics, st.session_state.transcript)

            with tab2:
                render_transcript_section(st.session_state.transcript)

            with tab3:
                # File info from previous analysis (collapsible)
                if st.session_state.file_info:
                    with st.expander("ℹ️ Posledný analyzovaný súbor", expanded=False):
                        fi = st.session_state.file_info
                        st.markdown(f"**Názov:** {fi.get('name', 'N/A')}")
                        st.markdown(f"**Veľkosť:** {format_file_size(fi.get('size_bytes', 0))}")
                        st.markdown(f"**Formát:** {fi.get('extension', 'N/A')}")

                st.markdown("### 📁 Nahrať nový súbor")
                uploaded_file, language = render_upload_section()

                if uploaded_file:
                    if st.button("🚀 Spustiť novú analýzu", type="primary"):
                        st.session_state.transcript = None
                        st.session_state.metrics = None
                        st.session_state.swap_speakers = False  # Reset swap
                        # Audio is already stored in render_upload_section
                        process_audio(uploaded_file, language)
                        st.rerun()
        else:
            # No analysis yet - show upload section
            uploaded_file, language = render_upload_section()

            if uploaded_file:
                if st.button("🚀 Spustiť analýzu", type="primary"):
                    st.session_state.transcript = None
                    st.session_state.metrics = None
                    st.session_state.swap_speakers = False  # Reset swap
                    # Audio is already stored in render_upload_section
                    process_audio(uploaded_file, language)
                    st.rerun()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #888; font-size: 0.8rem;">
            🐙 AutoQA v1.0 | Powered by Greenblu.me | 2025
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
