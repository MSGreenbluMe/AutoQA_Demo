"""
Dashboard Visualization Components

This module provides Plotly-based visualizations for the AutoQA dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from typing import Dict, Any


# Color palette
COLORS = {
    "primary": "#1E88E5",
    "success": "#43A047",
    "warning": "#FFA726",
    "error": "#E53935",
    "background": "#F5F5F5",
    "text": "#212121",
    "agent": "#1E88E5",
    "customer": "#43A047",
}


def get_score_color(score: float) -> str:
    """Get color based on score value.

    Args:
        score: Score value (0-10).

    Returns:
        Color hex code.
    """
    if score >= 8:
        return COLORS["success"]
    elif score >= 5:
        return COLORS["warning"]
    else:
        return COLORS["error"]


def render_metric_card(
    title: str,
    value: str,
    subtitle: str = "",
    score: float = None,
    icon: str = ""
):
    """Render a metric card with optional color coding.

    Args:
        title: Card title.
        value: Main value to display.
        subtitle: Optional subtitle.
        score: Optional score for color coding.
        icon: Optional emoji icon.
    """
    color = get_score_color(score) if score is not None else COLORS["primary"]

    st.markdown(
        f"""
        <div style="
            background: white;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid {color};
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 0.5rem;
        ">
            <div style="color: #666; font-size: 0.85rem; margin-bottom: 0.25rem;">
                {icon} {title}
            </div>
            <div style="font-size: 1.5rem; font-weight: bold; color: {color};">
                {value}
            </div>
            <div style="color: #888; font-size: 0.75rem;">
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
    height: int = 200
) -> go.Figure:
    """Create a gauge chart for score visualization.

    Args:
        value: Current value.
        title: Chart title.
        max_value: Maximum value.
        height: Chart height.

    Returns:
        Plotly Figure object.
    """
    color = get_score_color(value)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 14}},
        number={"font": {"size": 24}},
        gauge={
            "axis": {"range": [0, max_value], "tickwidth": 1},
            "bar": {"color": color},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#ddd",
            "steps": [
                {"range": [0, 4], "color": "#ffebee"},
                {"range": [4, 7], "color": "#fff3e0"},
                {"range": [7, 10], "color": "#e8f5e9"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": value
            }
        }
    ))

    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": COLORS["text"]},
    )

    return fig


def render_sentiment_chart(sentiment_data: Dict[str, Any]) -> go.Figure:
    """Create sentiment visualization.

    Args:
        sentiment_data: Sentiment analysis results.

    Returns:
        Plotly Figure object.
    """
    score = sentiment_data.get("sentiment_score", 0)

    # Map sentiment to emoji
    if score > 0.3:
        emoji = "😊"
    elif score < -0.3:
        emoji = "😞"
    else:
        emoji = "😐"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": f"Sentiment {emoji}", "font": {"size": 14}},
        number={"font": {"size": 24}, "suffix": ""},
        gauge={
            "axis": {"range": [-1, 1], "tickwidth": 1},
            "bar": {"color": get_score_color((score + 1) * 5)},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#ddd",
            "steps": [
                {"range": [-1, -0.3], "color": "#ffebee"},
                {"range": [-0.3, 0.3], "color": "#fff3e0"},
                {"range": [0.3, 1], "color": "#e8f5e9"},
            ],
        }
    ))

    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def render_word_count_chart(speech_metrics: Dict[str, Any]) -> go.Figure:
    """Create word count comparison bar chart.

    Args:
        speech_metrics: Speech metrics data.

    Returns:
        Plotly Figure object.
    """
    agent_words = speech_metrics.get("agent_words", 0)
    customer_words = speech_metrics.get("customer_words", 0)

    fig = go.Figure(data=[
        go.Bar(
            name="Agent",
            x=["Počet slov"],
            y=[agent_words],
            marker_color=COLORS["agent"],
            text=[agent_words],
            textposition="auto",
        ),
        go.Bar(
            name="Zákazník",
            x=["Počet slov"],
            y=[customer_words],
            marker_color=COLORS["customer"],
            text=[customer_words],
            textposition="auto",
        )
    ])

    fig.update_layout(
        title="Porovnanie počtu slov",
        barmode="group",
        height=250,
        margin=dict(l=40, r=20, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )

    return fig


def render_talk_time_pie(speech_metrics: Dict[str, Any]) -> go.Figure:
    """Create talk time distribution pie chart.

    Args:
        speech_metrics: Speech metrics data.

    Returns:
        Plotly Figure object.
    """
    agent_percent = speech_metrics.get("agent_talk_time_percent", 50)
    customer_percent = speech_metrics.get("customer_talk_time_percent", 50)

    fig = go.Figure(data=[go.Pie(
        labels=["Agent", "Zákazník"],
        values=[agent_percent, customer_percent],
        hole=0.4,
        marker_colors=[COLORS["agent"], COLORS["customer"]],
        textinfo="percent",
        textfont_size=14,
    )])

    fig.update_layout(
        title="Rozdelenie času hovoru",
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5
        ),
    )

    return fig


def render_overall_score_card(overall_data: Dict[str, Any]):
    """Render the overall quality score card.

    Args:
        overall_data: Overall score data.
    """
    score = overall_data.get("score", 0)
    rating = overall_data.get("rating", "N/A")
    color = get_score_color(score)

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {color}22, {color}11);
            padding: 1.5rem;
            border-radius: 12px;
            border: 2px solid {color};
            text-align: center;
            margin-bottom: 1rem;
        ">
            <div style="color: #666; font-size: 1rem; margin-bottom: 0.5rem;">
                🎯 Celkové hodnotenie kvality
            </div>
            <div style="font-size: 3rem; font-weight: bold; color: {color};">
                {score}/10
            </div>
            <div style="
                background: {color};
                color: white;
                padding: 0.25rem 1rem;
                border-radius: 20px;
                display: inline-block;
                margin-top: 0.5rem;
                font-weight: bold;
            ">
                {rating}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_fcr_card(fcr_data: Dict[str, Any]):
    """Render FCR prediction card.

    Args:
        fcr_data: FCR prediction data.
    """
    status = fcr_data.get("fcr_status", "Uncertain")
    confidence = fcr_data.get("confidence", 0)
    reasoning = fcr_data.get("reasoning", "")

    # Status to display
    status_display = {
        "Yes": ("✅ Vyriešené", COLORS["success"]),
        "No": ("❌ Nevyriešené", COLORS["error"]),
        "Uncertain": ("❓ Nejasné", COLORS["warning"]),
    }

    display_text, color = status_display.get(status, ("❓ Nejasné", COLORS["warning"]))

    st.markdown(
        f"""
        <div style="
            background: white;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid {color};
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <div style="color: #666; font-size: 0.85rem; margin-bottom: 0.25rem;">
                📞 First Call Resolution (FCR)
            </div>
            <div style="font-size: 1.25rem; font-weight: bold; color: {color};">
                {display_text}
            </div>
            <div style="color: #888; font-size: 0.75rem; margin-top: 0.25rem;">
                Istota: {confidence}%
            </div>
            <div style="color: #666; font-size: 0.8rem; margin-top: 0.5rem; font-style: italic;">
                {reasoning}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_component_scores(components: Dict[str, float]):
    """Render component scores as horizontal bars.

    Args:
        components: Dictionary of component scores.
    """
    labels = {
        "sentiment": "Sentiment",
        "fcr": "FCR",
        "interruptions": "Prerušenia",
        "empathy": "Empatia",
        "professionalism": "Profesionalita",
    }

    for key, value in components.items():
        label = labels.get(key, key)
        color = get_score_color(value)
        percentage = (value / 10) * 100

        st.markdown(
            f"""
            <div style="margin-bottom: 0.5rem;">
                <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
                    <span>{label}</span>
                    <span style="color: {color}; font-weight: bold;">{value}/10</span>
                </div>
                <div style="
                    background: #eee;
                    border-radius: 4px;
                    height: 8px;
                    overflow: hidden;
                ">
                    <div style="
                        background: {color};
                        width: {percentage}%;
                        height: 100%;
                        border-radius: 4px;
                    "></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_phrases_list(phrases: list, title: str, color: str):
    """Render a list of detected phrases.

    Args:
        phrases: List of phrases.
        title: Section title.
        color: Highlight color.
    """
    if not phrases:
        return

    st.markdown(f"**{title}:**")
    for phrase in phrases[:5]:
        st.markdown(
            f"""
            <span style="
                background: {color}22;
                color: {color};
                padding: 0.25rem 0.5rem;
                border-radius: 4px;
                font-size: 0.85rem;
                display: inline-block;
                margin: 0.125rem;
            ">
                "{phrase}"
            </span>
            """,
            unsafe_allow_html=True
        )


def render_transcript_message(
    speaker: str,
    text: str,
    timestamp: str,
    speaker_id: int = 0
):
    """Render a single transcript message in chat-like style.

    Args:
        speaker: Speaker name.
        text: Message text.
        timestamp: Message timestamp.
        speaker_id: Speaker ID for styling.
    """
    is_agent = speaker_id == 0 or speaker == "Agent"
    color = COLORS["agent"] if is_agent else COLORS["customer"]
    align = "flex-start" if is_agent else "flex-end"
    bg_color = f"{color}15"
    border_color = color

    st.markdown(
        f"""
        <div style="
            display: flex;
            justify-content: {align};
            margin-bottom: 0.5rem;
        ">
            <div style="
                background: {bg_color};
                border-left: 3px solid {border_color};
                padding: 0.75rem 1rem;
                border-radius: 8px;
                max-width: 80%;
            ">
                <div style="
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 0.25rem;
                ">
                    <span style="
                        font-weight: bold;
                        color: {color};
                        font-size: 0.85rem;
                    ">
                        {"🟦" if is_agent else "🟩"} {speaker}
                    </span>
                    <span style="
                        color: #888;
                        font-size: 0.7rem;
                        margin-left: 1rem;
                    ">
                        {timestamp}
                    </span>
                </div>
                <div style="color: #333; font-size: 0.9rem;">
                    {text}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
