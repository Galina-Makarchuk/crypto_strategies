"""Plotly visualization for backtest results.

Key improvements over original:
- Signals batched into 4 traces (long_entry, short_entry, long_exit, short_exit)
  instead of one trace per signal → fast rendering even with 300+ trades.
- Optional SuperTrend / EMA overlays.
- Always writes to file (no fig.show() surprise browser tabs in live mode).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .models import Signal, SignalAction, Direction

logger = logging.getLogger(__name__)

# Marker config by signal type
_MARKER_STYLES = {
    (SignalAction.ENTRY, Direction.LONG): {
        "color": "#22c55e",
        "symbol": "triangle-up",
        "textpos": "top center",
    },
    (SignalAction.ENTRY, Direction.SHORT): {
        "color": "#ef4444",
        "symbol": "triangle-down",
        "textpos": "bottom center",
    },
    (SignalAction.EXIT, Direction.LONG): {
        "color": "#eab308",
        "symbol": "triangle-down",
        "textpos": "bottom center",
    },
    (SignalAction.EXIT, Direction.SHORT): {
        "color": "#eab308",
        "symbol": "triangle-up",
        "textpos": "top center",
    },
}


def build_chart(
    df: pd.DataFrame,
    signals: list[Signal],
    title: str = "BTCUSDT Strategy Signals",
    save_path: str = "chart.html",
    show_volume: bool = True,
    auto_refresh: int = 0,
) -> str:
    """Build a candlestick chart with signal overlays and save to HTML.

    Args:
        auto_refresh: If > 0, injects a meta-refresh tag so the browser
                      reloads the page every N seconds. Set to your poll
                      interval (e.g. 30) for live mode.

    Returns the path of the saved file.
    """
    rows = 2 if show_volume else 1
    row_heights = [0.8, 0.2] if show_volume else [1.0]
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    # ── Candlestick ────────────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
        ),
        row=1,
        col=1,
    )

    # ── Optional indicator overlays ────────────────────────────────────────
    if "supertrend" in df.columns:
        colors = df["trend_dir"].map({1: "#22c55e", -1: "#ef4444"}).fillna("#888")
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["supertrend"],
                mode="lines",
                name="SuperTrend",
                line=dict(width=1.5, color="#888"),
            ),
            row=1,
            col=1,
        )

    for col_name, color, label in [
        ("ema_fast", "#3b82f6", "EMA Fast"),
        ("ema_slow", "#f97316", "EMA Slow"),
    ]:
        if col_name in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col_name],
                    mode="lines",
                    name=label,
                    line=dict(width=1, color=color),
                ),
                row=1,
                col=1,
            )

    # ── Batched signal markers ─────────────────────────────────────────────
    grouped: dict[tuple, list[Signal]] = defaultdict(list)
    for sig in signals:
        key = (sig.action, sig.direction)
        grouped[key].append(sig)

    for key, sigs in grouped.items():
        style = _MARKER_STYLES.get(key, _MARKER_STYLES[(SignalAction.EXIT, Direction.LONG)])
        name = f"{key[1].value.capitalize()} {key[0].value.capitalize()}"
        fig.add_trace(
            go.Scatter(
                x=[s.timestamp for s in sigs],
                y=[s.price for s in sigs],
                mode="markers+text",
                marker=dict(
                    symbol=style["symbol"],
                    size=12,
                    color=style["color"],
                    line=dict(width=1.5, color="black"),
                ),
                text=[s.label for s in sigs],
                textposition=style["textpos"],
                textfont=dict(size=8),
                name=name,
            ),
            row=1,
            col=1,
        )

    # ── Volume ─────────────────────────────────────────────────────────────
    if show_volume and "volume" in df.columns:
        colors = [
            "rgba(34,197,94,0.5)" if c >= o else "rgba(239,68,68,0.5)"
            for c, o in zip(df["close"], df["open"])
        ]
        fig.add_trace(
            go.Bar(x=df.index, y=df["volume"], name="Volume", marker_color=colors),
            row=2,
            col=1,
        )

    # ── Layout ─────────────────────────────────────────────────────────────
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=900,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
    if show_volume:
        fig.update_yaxes(title_text="Volume", row=2, col=1)

    fig.write_html(save_path)

    if auto_refresh > 0:
        with open(save_path, "r") as f:
            html = f.read()
        meta_tag = f'<meta http-equiv="refresh" content="{auto_refresh}">'
        html = html.replace("<head>", f"<head>{meta_tag}", 1)
        with open(save_path, "w") as f:
            f.write(html)

    logger.info("Chart saved → %s", save_path)
    return save_path
