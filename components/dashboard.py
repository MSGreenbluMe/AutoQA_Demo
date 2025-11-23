"""
Dashboard Visualization Components

Professional dashboard visualizations for AutoQA.
Modern glassmorphism design with gradient accents.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, Any, List


# Professional color palette - Greenblu.me theme
COLORS = {
    "primary": "#0EA5E9",      # Sky blue
    "secondary": "#6366F1",    # Indigo
    "success": "#10B981",      # Emerald
    "warning": "#F59E0B",      # Amber
    "error": "#EF4444",        # Red
    "background": "#0F172A",   # Slate 900
    "surface": "#1E293B",      # Slate 800
    "text": "#F8FAFC",         # Slate 50
    "text_muted": "#94A3B8",   # Slate 400
    "agent": "#0EA5E9",        # Sky blue
    "customer": "#10B981",     # Emerald
    "accent": "#8B5CF6",       # Violet
    "gradient_start": "#0EA5E9",
    "gradient_end": "#6366F1",
}


def get_score_color(score: float) -> str:
    """Get color based on score value."""
    if score >= 8:
        return COLORS["success"]
    elif score >= 5:
        return COLORS["warning"]
    else:
        return COLORS["error"]


def get_score_gradient(score: float) -> str:
    """Get gradient based on score value."""
    if score >= 8:
        return "linear-gradient(135deg, #10B981 0%, #059669 100%)"
    elif score >= 5:
        return "linear-gradient(135deg, #F59E0B 0%, #D97706 100%)"
    else:
        return "linear-gradient(135deg, #EF4444 0%, #DC2626 100%)"


def render_metric_card(
    title: str,
    value: str,
    subtitle: str = "",
    score: float = None,
    icon: str = ""
):
    """Render a glassmorphism metric card."""
    color = get_score_color(score) if score is not None else COLORS["primary"]

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
            backdrop-filter: blur(10px);
            padding: 1.25rem;
            border-radius: 16px;
            border: 1px solid rgba(148, 163, 184, 0.1);
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
            margin-bottom: 0.75rem;
            position: relative;
            overflow: hidden;
        ">
            <div style="
                position: absolute;
                top: 0;
                left: 0;
                width: 4px;
                height: 100%;
                background: {color};
                border-radius: 16px 0 0 16px;
            "></div>
            <div style="color: {COLORS['text_muted']}; font-size: 0.8rem; margin-bottom: 0.5rem; font-weight: 500;">
                {icon} {title}
            </div>
            <div style="font-size: 1.75rem; font-weight: 700; color: {color}; letter-spacing: -0.5px;">
                {value}
            </div>
            <div style="color: {COLORS['text_muted']}; font-size: 0.75rem; margin-top: 0.25rem;">
                {subtitle}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_gauge_chart(
    value: float,
    title: str,
    max_value: float = 10,
    height: int = 220
) -> go.Figure:
    """Create a modern gauge chart."""
    color = get_score_color(value)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 14, "color": COLORS["text_muted"]}},
        number={"font": {"size": 32, "color": COLORS["text"]}, "suffix": ""},
        gauge={
            "axis": {
                "range": [0, max_value],
                "tickwidth": 1,
                "tickcolor": COLORS["text_muted"],
                "tickfont": {"color": COLORS["text_muted"]},
            },
            "bar": {"color": color, "thickness": 0.8},
            "bgcolor": COLORS["surface"],
            "borderwidth": 0,
            "steps": [
                {"range": [0, 4], "color": "rgba(239, 68, 68, 0.15)"},
                {"range": [4, 7], "color": "rgba(245, 158, 11, 0.15)"},
                {"range": [7, 10], "color": "rgba(16, 185, 129, 0.15)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": value
            }
        }
    ))

    fig.update_layout(
        height=height,
        margin=dict(l=30, r=30, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLORS["text"]},
    )

    return fig


def render_sentiment_chart(sentiment_data: Dict[str, Any]) -> go.Figure:
    """Create modern sentiment visualization."""
    score = sentiment_data.get("sentiment_score", 0)

    if score > 0.3:
        emoji = "😊"
        color = COLORS["success"]
    elif score < -0.3:
        emoji = "😔"
        color = COLORS["error"]
    else:
        emoji = "😐"
        color = COLORS["warning"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": f"Sentiment {emoji}", "font": {"size": 14, "color": COLORS["text_muted"]}},
        number={"font": {"size": 28, "color": COLORS["text"]}, "valueformat": ".2f"},
        gauge={
            "axis": {
                "range": [-1, 1],
                "tickwidth": 1,
                "tickcolor": COLORS["text_muted"],
                "tickfont": {"color": COLORS["text_muted"]},
            },
            "bar": {"color": color, "thickness": 0.8},
            "bgcolor": COLORS["surface"],
            "borderwidth": 0,
            "steps": [
                {"range": [-1, -0.3], "color": "rgba(239, 68, 68, 0.15)"},
                {"range": [-0.3, 0.3], "color": "rgba(245, 158, 11, 0.15)"},
                {"range": [0.3, 1], "color": "rgba(16, 185, 129, 0.15)"},
            ],
        }
    ))

    fig.update_layout(
        height=220,
        margin=dict(l=30, r=30, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def render_word_count_chart(speech_metrics: Dict[str, Any]) -> go.Figure:
    """Create modern word count comparison chart."""
    agent_words = speech_metrics.get("agent_words", 0)
    customer_words = speech_metrics.get("customer_words", 0)

    fig = go.Figure(data=[
        go.Bar(
            name="Agent",
            x=["Agent", "Zákazník"],
            y=[agent_words, 0],
            marker=dict(
                color=COLORS["agent"],
                line=dict(width=0),
                cornerradius=8,
            ),
            text=[agent_words, ""],
            textposition="auto",
            textfont=dict(color=COLORS["text"], size=14, family="Arial Black"),
        ),
        go.Bar(
            name="Zákazník",
            x=["Agent", "Zákazník"],
            y=[0, customer_words],
            marker=dict(
                color=COLORS["customer"],
                line=dict(width=0),
                cornerradius=8,
            ),
            text=["", customer_words],
            textposition="auto",
            textfont=dict(color=COLORS["text"], size=14, family="Arial Black"),
        )
    ])

    fig.update_layout(
        title=dict(
            text="Počet slov",
            font=dict(size=14, color=COLORS["text_muted"]),
            x=0.5,
        ),
        barmode="stack",
        height=280,
        margin=dict(l=40, r=20, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(
            tickfont=dict(color=COLORS["text_muted"]),
            gridcolor="rgba(148, 163, 184, 0.1)",
        ),
        yaxis=dict(
            tickfont=dict(color=COLORS["text_muted"]),
            gridcolor="rgba(148, 163, 184, 0.1)",
        ),
    )

    return fig


def render_talk_time_pie(speech_metrics: Dict[str, Any]) -> go.Figure:
    """Create modern talk time donut chart."""
    agent_percent = speech_metrics.get("agent_talk_time_percent", 50)
    customer_percent = speech_metrics.get("customer_talk_time_percent", 50)

    fig = go.Figure(data=[go.Pie(
        labels=["Agent", "Zákazník"],
        values=[agent_percent, customer_percent],
        hole=0.6,
        marker=dict(
            colors=[COLORS["agent"], COLORS["customer"]],
            line=dict(color=COLORS["background"], width=3),
        ),
        textinfo="percent",
        textfont=dict(size=14, color=COLORS["text"]),
        hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
    )])

    fig.update_layout(
        title=dict(
            text="Čas hovoru",
            font=dict(size=14, color=COLORS["text_muted"]),
            x=0.5,
        ),
        height=280,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(color=COLORS["text_muted"]),
        ),
        annotations=[dict(
            text=f"{agent_percent:.0f}%<br><span style='font-size:10px'>Agent</span>",
            x=0.5, y=0.5,
            font=dict(size=16, color=COLORS["text"]),
            showarrow=False,
        )],
    )

    return fig


def render_overall_score_card(overall_data: Dict[str, Any]):
    """Render the hero overall quality score card."""
    score = overall_data.get("score", 0)
    rating = overall_data.get("rating", "N/A")
    gradient = get_score_gradient(score)
    color = get_score_color(score)

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
            backdrop-filter: blur(20px);
            padding: 2rem;
            border-radius: 24px;
            border: 1px solid rgba(148, 163, 184, 0.1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            text-align: center;
            margin-bottom: 1.5rem;
            position: relative;
            overflow: hidden;
        ">
            <div style="
                position: absolute;
                top: -50%;
                right: -20%;
                width: 300px;
                height: 300px;
                background: radial-gradient(circle, {color}20 0%, transparent 70%);
                pointer-events: none;
            "></div>
            <div style="color: {COLORS['text_muted']}; font-size: 0.9rem; margin-bottom: 0.75rem; font-weight: 500; letter-spacing: 1px; text-transform: uppercase;">
                🐙 Celkové hodnotenie
            </div>
            <div style="
                font-size: 4.5rem;
                font-weight: 800;
                background: {gradient};
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                line-height: 1;
                margin-bottom: 0.5rem;
            ">
                {score}
            </div>
            <div style="color: {COLORS['text_muted']}; font-size: 1rem; margin-bottom: 1rem;">
                z 10 bodov
            </div>
            <div style="
                background: {gradient};
                color: white;
                padding: 0.5rem 1.5rem;
                border-radius: 30px;
                display: inline-block;
                font-weight: 600;
                font-size: 0.9rem;
                letter-spacing: 0.5px;
                box-shadow: 0 4px 15px {color}40;
            ">
                {rating}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_fcr_card(fcr_data: Dict[str, Any]):
    """Render modern FCR prediction card."""
    status = fcr_data.get("fcr_status", "Uncertain")
    confidence = fcr_data.get("confidence", 0)
    reasoning = fcr_data.get("reasoning", "")

    status_display = {
        "Yes": ("✅ Vyriešené", COLORS["success"], "linear-gradient(135deg, #10B981 0%, #059669 100%)"),
        "No": ("❌ Nevyriešené", COLORS["error"], "linear-gradient(135deg, #EF4444 0%, #DC2626 100%)"),
        "Uncertain": ("❓ Nejasné", COLORS["warning"], "linear-gradient(135deg, #F59E0B 0%, #D97706 100%)"),
    }

    display_text, color, gradient = status_display.get(status, ("❓ Nejasné", COLORS["warning"], "linear-gradient(135deg, #F59E0B 0%, #D97706 100%)"))

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
            backdrop-filter: blur(10px);
            padding: 1.25rem;
            border-radius: 16px;
            border: 1px solid rgba(148, 163, 184, 0.1);
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
            position: relative;
            overflow: hidden;
        ">
            <div style="
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 3px;
                background: {gradient};
            "></div>
            <div style="color: {COLORS['text_muted']}; font-size: 0.8rem; margin-bottom: 0.75rem; font-weight: 500;">
                📞 First Call Resolution
            </div>
            <div style="font-size: 1.4rem; font-weight: 700; color: {color}; margin-bottom: 0.5rem;">
                {display_text}
            </div>
            <div style="
                display: inline-block;
                background: rgba(148, 163, 184, 0.1);
                padding: 0.25rem 0.75rem;
                border-radius: 20px;
                font-size: 0.75rem;
                color: {COLORS['text_muted']};
                margin-bottom: 0.75rem;
            ">
                Istota: {confidence}%
            </div>
            <div style="color: {COLORS['text_muted']}; font-size: 0.8rem; font-style: italic; line-height: 1.4;">
                {reasoning}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_component_scores(components: Dict[str, float]):
    """Render component scores as modern progress bars."""
    labels = {
        "sentiment": ("💭", "Sentiment"),
        "fcr": ("📞", "FCR"),
        "interruptions": ("🔇", "Prerušenia"),
        "empathy": ("💚", "Empatia"),
        "professionalism": ("⭐", "Profesionalita"),
    }

    for key, value in components.items():
        icon, label = labels.get(key, ("•", key))
        color = get_score_color(value)
        percentage = (value / 10) * 100

        st.markdown(
            f"""
            <div style="margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                    <span style="color: {COLORS['text']}; font-size: 0.85rem; font-weight: 500;">
                        {icon} {label}
                    </span>
                    <span style="
                        color: {color};
                        font-weight: 700;
                        font-size: 0.9rem;
                    ">{value}/10</span>
                </div>
                <div style="
                    background: rgba(148, 163, 184, 0.1);
                    border-radius: 10px;
                    height: 10px;
                    overflow: hidden;
                    box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
                ">
                    <div style="
                        background: linear-gradient(90deg, {color} 0%, {color}CC 100%);
                        width: {percentage}%;
                        height: 100%;
                        border-radius: 10px;
                        box-shadow: 0 0 10px {color}40;
                        transition: width 0.5s ease;
                    "></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_phrases_list(phrases: list, title: str, color: str):
    """Render a list of detected phrases as chips."""
    if not phrases:
        return

    st.markdown(f"**{title}:**")

    # Use simple markdown for each phrase to avoid HTML escaping issues
    for phrase in phrases[:5]:
        # Escape any HTML in the phrase
        safe_phrase = str(phrase).replace("<", "&lt;").replace(">", "&gt;")
        st.markdown(
            f'<span style="background:{color}20;color:{color};padding:0.3rem 0.7rem;'
            f'border-radius:16px;font-size:0.8rem;display:inline-block;margin:0.15rem;'
            f'border:1px solid {color}40;">"{safe_phrase}"</span>',
            unsafe_allow_html=True
        )


def render_transcript_message(
    speaker: str,
    text: str,
    timestamp: str,
    speaker_id: int = 0
):
    """Render a single transcript message in modern chat style."""
    is_agent = speaker_id == 0 or speaker == "Agent"
    color = COLORS["agent"] if is_agent else COLORS["customer"]
    align = "flex-start" if is_agent else "flex-end"
    icon = "🎧" if is_agent else "👤"

    st.markdown(
        f"""
        <div style="
            display: flex;
            justify-content: {align};
            margin-bottom: 0.75rem;
        ">
            <div style="
                background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
                backdrop-filter: blur(10px);
                border: 1px solid {color}30;
                border-left: 4px solid {color};
                padding: 1rem 1.25rem;
                border-radius: 12px;
                max-width: 85%;
                box-shadow: 0 2px 12px rgba(0, 0, 0, 0.15);
            ">
                <div style="
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 0.5rem;
                ">
                    <span style="
                        font-weight: 600;
                        color: {color};
                        font-size: 0.85rem;
                    ">
                        {icon} {speaker}
                    </span>
                    <span style="
                        color: {COLORS['text_muted']};
                        font-size: 0.7rem;
                        margin-left: 1rem;
                        background: rgba(148, 163, 184, 0.1);
                        padding: 0.15rem 0.5rem;
                        border-radius: 10px;
                    ">
                        {timestamp}
                    </span>
                </div>
                <div style="color: {COLORS['text']}; font-size: 0.9rem; line-height: 1.5;">
                    {text}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_stats_row(stats: list):
    """Render a row of quick stats."""
    cols = st.columns(len(stats))
    for col, (icon, label, value, color) in zip(cols, stats):
        with col:
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
                    backdrop-filter: blur(10px);
                    padding: 1rem;
                    border-radius: 12px;
                    border: 1px solid rgba(148, 163, 184, 0.1);
                    text-align: center;
                ">
                    <div style="font-size: 1.5rem; margin-bottom: 0.25rem;">{icon}</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: {color};">{value}</div>
                    <div style="font-size: 0.75rem; color: {COLORS['text_muted']};">{label}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


def render_call_timeline(segments: List[Dict[str, Any]], duration: float, swap_speakers: bool = False) -> go.Figure:
    """
    Create an interactive call timeline visualization.

    Shows:
    - Speaker segments as colored bars (agent/customer)
    - Speech rate (WPM) curves for both speakers
    - Hover tooltips with segment text

    Args:
        segments: List of transcript segments
        duration: Total call duration in seconds
        swap_speakers: If True, swap agent/customer roles
    """
    if not segments or duration <= 0:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="Call Timeline",
            annotations=[dict(
                text="Žiadne dáta pre zobrazenie",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color=COLORS["text_muted"])
            )],
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.12,
        subplot_titles=("", "Rýchlosť reči (slov/min)"),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )

    # Prepare data for timeline bars
    agent_segments = []
    customer_segments = []

    # Prepare data for WPM curves
    agent_wpm_times = []
    agent_wpm_values = []
    customer_wpm_times = []
    customer_wpm_values = []

    for seg in segments:
        start = seg.get("start", 0)
        end = seg.get("end", start + 1)
        text = seg.get("text", "")
        speaker = seg.get("speaker", "Unknown")
        speaker_id = seg.get("speaker_id", 0)

        # Calculate WPM for this segment
        seg_duration = end - start
        word_count = len(text.split())
        wpm = (word_count / (seg_duration / 60)) if seg_duration > 0 else 0

        # Truncate text for hover (max 100 chars)
        hover_text = text[:100] + "..." if len(text) > 100 else text

        # Determine if agent or customer (use speaker string, not ID)
        is_agent_original = speaker == "Agent"
        is_agent = not is_agent_original if swap_speakers else is_agent_original

        # Display name after potential swap
        display_speaker = "Agent" if is_agent else "Zákazník"

        segment_data = {
            "start": start,
            "end": end,
            "text": hover_text,
            "wpm": round(wpm, 1),
            "speaker": display_speaker,
            "word_count": word_count,
        }

        if is_agent:
            agent_segments.append(segment_data)
            # Add WPM data point at segment midpoint
            mid_time = (start + end) / 2
            agent_wpm_times.append(mid_time)
            agent_wpm_values.append(wpm)
        else:
            customer_segments.append(segment_data)
            mid_time = (start + end) / 2
            customer_wpm_times.append(mid_time)
            customer_wpm_values.append(wpm)

    # Add agent segments as bars (row 1)
    for seg in agent_segments:
        fig.add_trace(
            go.Bar(
                x=[seg["end"] - seg["start"]],
                y=["Agent"],
                base=[seg["start"]],
                orientation="h",
                marker=dict(
                    color=COLORS["agent"],
                    line=dict(width=1, color=COLORS["agent"]),
                    opacity=0.85,
                ),
                hovertemplate=(
                    f"<b>🎧 {seg['speaker']}</b><br>"
                    f"<b>Čas:</b> {_format_time(seg['start'])} - {_format_time(seg['end'])}<br>"
                    f"<b>Slov:</b> {seg['word_count']} ({seg['wpm']} slov/min)<br>"
                    f"<b>Text:</b> {seg['text']}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1, col=1
        )

    # Add customer segments as bars (row 1)
    for seg in customer_segments:
        fig.add_trace(
            go.Bar(
                x=[seg["end"] - seg["start"]],
                y=["Zákazník"],
                base=[seg["start"]],
                orientation="h",
                marker=dict(
                    color=COLORS["customer"],
                    line=dict(width=1, color=COLORS["customer"]),
                    opacity=0.85,
                ),
                hovertemplate=(
                    f"<b>👤 {seg['speaker']}</b><br>"
                    f"<b>Čas:</b> {_format_time(seg['start'])} - {_format_time(seg['end'])}<br>"
                    f"<b>Slov:</b> {seg['word_count']} ({seg['wpm']} slov/min)<br>"
                    f"<b>Text:</b> {seg['text']}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1, col=1
        )

    # Add WPM line for agent (row 2)
    if agent_wpm_times:
        # Sort by time
        agent_data = sorted(zip(agent_wpm_times, agent_wpm_values))
        times, values = zip(*agent_data) if agent_data else ([], [])

        fig.add_trace(
            go.Scatter(
                x=times,
                y=values,
                mode="lines+markers",
                name="Agent",
                line=dict(color=COLORS["agent"], width=3, shape="spline"),
                marker=dict(size=8, color=COLORS["agent"], symbol="circle"),
                hovertemplate="<b>Agent</b><br>Čas: %{x:.1f}s<br>WPM: %{y:.0f}<extra></extra>",
                fill="tozeroy",
                fillcolor=f"rgba(14, 165, 233, 0.15)",
            ),
            row=2, col=1
        )

    # Add WPM line for customer (row 2)
    if customer_wpm_times:
        # Sort by time
        customer_data = sorted(zip(customer_wpm_times, customer_wpm_values))
        times, values = zip(*customer_data) if customer_data else ([], [])

        fig.add_trace(
            go.Scatter(
                x=times,
                y=values,
                mode="lines+markers",
                name="Zákazník",
                line=dict(color=COLORS["customer"], width=3, shape="spline"),
                marker=dict(size=8, color=COLORS["customer"], symbol="diamond"),
                hovertemplate="<b>Zákazník</b><br>Čas: %{x:.1f}s<br>WPM: %{y:.0f}<extra></extra>",
                fill="tozeroy",
                fillcolor=f"rgba(16, 185, 129, 0.15)",
            ),
            row=2, col=1
        )

    # Update layout
    fig.update_layout(
        title=dict(
            text="📊 Call Timeline",
            font=dict(size=18, color=COLORS["text"]),
            x=0.5,
            xanchor="center",
        ),
        height=400,
        margin=dict(l=80, r=40, t=60, b=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        barmode="overlay",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color=COLORS["text_muted"], size=12),
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="closest",
    )

    # Update axes for timeline (row 1)
    fig.update_xaxes(
        title_text="",
        range=[0, duration],
        tickfont=dict(color=COLORS["text_muted"]),
        gridcolor="rgba(148, 163, 184, 0.1)",
        zeroline=False,
        tickformat=".0f",
        ticksuffix="s",
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="",
        tickfont=dict(color=COLORS["text_muted"], size=12),
        gridcolor="rgba(148, 163, 184, 0.1)",
        zeroline=False,
        row=1, col=1
    )

    # Update axes for WPM chart (row 2)
    fig.update_xaxes(
        title_text="Čas (sekundy)",
        title_font=dict(color=COLORS["text_muted"], size=11),
        range=[0, duration],
        tickfont=dict(color=COLORS["text_muted"]),
        gridcolor="rgba(148, 163, 184, 0.1)",
        zeroline=False,
        tickformat=".0f",
        ticksuffix="s",
        row=2, col=1
    )
    fig.update_yaxes(
        title_text="WPM",
        title_font=dict(color=COLORS["text_muted"], size=11),
        tickfont=dict(color=COLORS["text_muted"]),
        gridcolor="rgba(148, 163, 184, 0.1)",
        zeroline=False,
        row=2, col=1
    )

    # Style subtitle
    fig.update_annotations(font=dict(color=COLORS["text_muted"], size=12))

    return fig


def _format_time(seconds: float) -> str:
    """Format seconds to MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"
