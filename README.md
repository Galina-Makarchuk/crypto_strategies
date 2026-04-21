# Entry/exit approaches in crypto trading strategies.

A lightweight research repo for comparing crypto trading strategies with a focus on how they define entry and exit points.

## Overview

Most trading performance doesn't come from the idea itself, but from how entries and exits are defined and executed. This repo explores different approaches to structuring those decisions in a systematic and testable way.

Focus on:
- Translating market behavior into rule-based signals
- Comparing entry/exit logic across strategy types
- Evaluating robustness under different market regimes

## Core Idea

Instead of treating strategies as monolithic systems, I broke them into:
- **Signal generation** — Why enter?
- **Execution logic** — When exactly enter?
- **Exit logic** — When and why exit?

This allows consistent comparison across fundamentally different strategies.

## Entry Approaches

**Trend-following**
- EMA crossovers (fast/slow)
- Supertrend (ATR-based flip)
- Donchian channel breakouts

**Mean-reversion**
- RSI oversold/overbought
- Bollinger Band fades
- VWAP reversion

**Momentum**
- MACD crosses
- Breakout on volume expansion
- Higher-high/higher-low swing structure

## Exit Approaches

- ATR trailing stop (peak-tracked)
- Fixed R:R targets
- Opposite-signal flip
- Time-based stop

## Evaluation Metrics

- Win rate
- P&L
- Max drawdown

## Goal

Build a modular framework where:
- Entry and exit logic can be swapped independently
- Strategies are comparable on equal footing
- Insights emerge from structure, not indicators

## Future Work

- Adaptive parameter selection
