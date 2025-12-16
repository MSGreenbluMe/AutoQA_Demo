"""
Advanced Visualizations for ERFC, SSR, CCM

Uses existing COLORS from dashboard.py
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any

# Import colors from existing dashboard
from components.dashboard import COLORS


def create_emotion_timeline(erfc_result) -> go.Figure:
    """
    ERFC: Interactive emotion timeline.

    - Two lines: agent and customer valence
    - Color-coded by emotion category
    - Forecast zone shaded
    - Intervention points annotated
    """
    turns = erfc_result.turns
    forecasted = erfc_result.forecasted_turns

    if not turns:
        fig = go.Figure()
        fig.update_layout(
            title="Emotion Timeline",
            annotations=[dict(
                text="Nedostatok dát pre analýzu",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color=COLORS["text_muted"])
            )],
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    # Customer data
    cust_x = [t.turn_index for t in turns if t.customer_emotion]
    cust_y = [t.customer_emotion.avd.valence for t in turns if t.customer_emotion]
    cust_hover = [
        f"Turn {t.turn_index}<br>"
        f"Emócia: {t.customer_emotion.category}<br>"
        f"Valence: {t.customer_emotion.avd.valence:.2f}<br>"
        f"Text: {t.customer_text[:60]}..."
        for t in turns if t.customer_emotion
    ]

    # Agent data
    agent_x = [t.turn_index for t in turns if t.agent_emotion]
    agent_y = [t.agent_emotion.avd.valence for t in turns if t.agent_emotion]
    agent_hover = [
        f"Turn {t.turn_index}<br>"
        f"Emócia: {t.agent_emotion.category}<br>"
        f"Valence: {t.agent_emotion.avd.valence:.2f}"
        for t in turns if t.agent_emotion
    ]

    fig = go.Figure()

    # Customer line
    fig.add_trace(go.Scatter(
        x=cust_x, y=cust_y,
        mode='lines+markers',
        name='Zákazník',
        line=dict(color=COLORS["customer"], width=3),
        marker=dict(size=10),
        hovertext=cust_hover,
        hoverinfo='text'
    ))

    # Agent line
    fig.add_trace(go.Scatter(
        x=agent_x, y=agent_y,
        mode='lines+markers',
        name='Agent',
        line=dict(color=COLORS["agent"], width=3),
        marker=dict(size=8),
        hovertext=agent_hover,
        hoverinfo='text'
    ))

    # Forecast zone
    if forecasted and cust_x:
        forecast_x = [cust_x[-1]] + [f.turn_index for f in forecasted if f.customer_emotion]
        forecast_y = [cust_y[-1]] + [f.customer_emotion.avd.valence for f in forecasted if f.customer_emotion]

        if len(forecast_x) > 1:
            fig.add_trace(go.Scatter(
                x=forecast_x, y=forecast_y,
                mode='lines',
                name='Predikcia',
                line=dict(color=COLORS["customer"], width=2, dash='dash'),
                fill='tozeroy',
                fillcolor='rgba(16, 185, 129, 0.1)'
            ))

    # Intervention point annotations
    for ip in erfc_result.intervention_points:
        color = COLORS["success"] if ip.intervention_type == "positive_impact" else COLORS["error"]
        symbol = "⬆" if ip.intervention_type == "positive_impact" else "⬇"

        fig.add_annotation(
            x=ip.turn_index,
            y=1.15,
            text=symbol,
            showarrow=False,
            font=dict(size=16, color=color)
        )

    fig.update_layout(
        title=dict(
            text="🎭 Emočná časová os (ERFC)",
            font=dict(size=16, color=COLORS["text"])
        ),
        xaxis_title="Turn",
        xaxis=dict(
            tickfont=dict(color=COLORS["text_muted"]),
            gridcolor="rgba(148,163,184,0.1)"
        ),
        yaxis_title="Valence",
        yaxis=dict(
            range=[-1.3, 1.3],
            tickfont=dict(color=COLORS["text_muted"]),
            gridcolor="rgba(148,163,184,0.1)"
        ),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            font=dict(color=COLORS["text_muted"])
        ),
        height=350,
        margin=dict(l=50, r=30, t=80, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def create_trajectory_gauge(consistency: float, trajectory: str) -> go.Figure:
    """ERFC: Agent consistency gauge."""
    color = COLORS["success"] if consistency > 0.7 else COLORS["warning"] if consistency > 0.4 else COLORS["error"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=consistency * 100,
        title={"text": "Konzistencia agenta", "font": {"size": 12, "color": COLORS["text_muted"]}},
        number={"font": {"size": 24, "color": COLORS["text"]}, "suffix": "%"},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"color": COLORS["text_muted"]}},
            "bar": {"color": color, "thickness": 0.8},
            "bgcolor": COLORS["surface"],
            "steps": [
                {"range": [0, 40], "color": "rgba(239, 68, 68, 0.15)"},
                {"range": [40, 70], "color": "rgba(245, 158, 11, 0.15)"},
                {"range": [70, 100], "color": "rgba(16, 185, 129, 0.15)"},
            ],
        }
    ))

    fig.update_layout(
        height=180,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def create_distribution_bars(ssr_results: Dict[str, Any]) -> go.Figure:
    """
    SSR: Distribution bars for all dimensions.

    Shows DISTRIBUTIONS, not single numbers!
    """
    dimensions = list(ssr_results.keys())

    if not dimensions:
        fig = go.Figure()
        fig.update_layout(title="Žiadne SSR dáta")
        return fig

    # Translate dimension names
    dim_labels = {
        "empathy": "Empatia",
        "professionalism": "Profesionalita",
        "fcr_likelihood": "Pravdepodobnosť FCR"
    }

    fig = make_subplots(
        rows=len(dimensions), cols=1,
        subplot_titles=[dim_labels.get(d, d) for d in dimensions],
        vertical_spacing=0.15
    )

    color_map = {
        "low": COLORS["error"],
        "unlikely": COLORS["error"],
        "medium": COLORS["warning"],
        "possible": COLORS["warning"],
        "high": COLORS["success"],
        "likely": COLORS["success"],
    }

    # Translate category names
    cat_labels = {
        "low": "Nízka",
        "medium": "Stredná",
        "high": "Vysoká",
        "unlikely": "Nepravdepodobné",
        "possible": "Možné",
        "likely": "Pravdepodobné"
    }

    for i, (dimension, result) in enumerate(ssr_results.items(), 1):
        categories = list(result.distribution.keys())
        values = list(result.distribution.values())
        colors = [color_map.get(c, COLORS["primary"]) for c in categories]
        labels = [cat_labels.get(c, c) for c in categories]

        fig.add_trace(
            go.Bar(
                x=values,
                y=labels,
                orientation='h',
                marker_color=colors,
                text=[f'{v:.0%}' for v in values],
                textposition='inside',
                textfont=dict(color="white", size=11),
                showlegend=False,
            ),
            row=i, col=1
        )

    fig.update_layout(
        title=dict(
            text="📊 Distribúcie hodnotení (SSR)",
            font=dict(size=16, color=COLORS["text"])
        ),
        height=130 * len(dimensions) + 60,
        margin=dict(l=100, r=30, t=60, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    fig.update_xaxes(range=[0, 1], tickformat='.0%', gridcolor="rgba(148,163,184,0.1)")
    fig.update_yaxes(tickfont=dict(color=COLORS["text_muted"]))
    fig.update_annotations(font=dict(color=COLORS["text_muted"], size=12))

    return fig


def create_confidence_table(ccm_results: Dict[str, Any]) -> go.Figure:
    """CCM: Confidence indicators table."""
    if not ccm_results:
        fig = go.Figure()
        fig.update_layout(title="Žiadne CCM dáta")
        return fig

    # Translate
    dim_labels = {
        "empathy": "Empatia",
        "professionalism": "Profesionalita",
        "fcr_likelihood": "FCR"
    }

    cat_labels = {
        "low": "Nízka", "medium": "Stredná", "high": "Vysoká",
        "unlikely": "Nepravdepodobné", "possible": "Možné", "likely": "Pravdepodobné"
    }

    dimensions = []
    prediction_sets = []
    confidences = []
    reviews = []

    emoji_map = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}
    conf_labels = {"HIGH": "Vysoká", "MEDIUM": "Stredná", "LOW": "Nízka"}

    for dimension, result in ccm_results.items():
        dimensions.append(dim_labels.get(dimension, dimension))
        translated_set = [cat_labels.get(c, c) for c in result.prediction_set]
        prediction_sets.append(', '.join(translated_set))
        confidences.append(f"{emoji_map.get(result.confidence_category, '⚪')} {conf_labels.get(result.confidence_category, result.confidence_category)}")
        reviews.append("⚠️ Áno" if result.needs_review else "✓ Nie")

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Dimenzia', 'Možné hodnoty', 'Istota', 'Kontrola'],
            fill_color=COLORS["surface"],
            font=dict(color=COLORS["text"], size=12),
            align='left',
            height=32
        ),
        cells=dict(
            values=[dimensions, prediction_sets, confidences, reviews],
            fill_color=COLORS["background"],
            font=dict(color=COLORS["text_muted"], size=11),
            align='left',
            height=28
        )
    )])

    fig.update_layout(
        title=dict(
            text="🎯 Istota klasifikácie (CCM)",
            font=dict(size=16, color=COLORS["text"])
        ),
        height=50 + 32 * (len(ccm_results) + 1),
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def create_overall_confidence_indicator(ccm_results: Dict[str, Any]) -> tuple:
    """Overall analysis confidence indicator."""
    high_count = sum(1 for r in ccm_results.values() if r.confidence_category == 'HIGH')
    total = len(ccm_results)

    overall_pct = (high_count / total * 100) if total > 0 else 0
    needs_review = any(r.needs_review for r in ccm_results.values())

    return overall_pct, needs_review
