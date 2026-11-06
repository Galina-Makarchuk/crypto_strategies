# Entry/exit approaches in crypto trading strategies.

A pluggable-strategy model to test a portfolio of strategies.

A research repo for comparing crypto trading strategies with a focus on how they define entry and exit points.

backtester.py supports pluggable strategies: 
- small core
- strategy classes plug in
- uniform bps-denominated comparison

## Overview

Most trading performance doesn't come from the idea itself, but from how entries and exits are defined and executed. \
This repo explores different approaches to structuring those decisions in a systematic and testable way.

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

- Adaptive parameter selection -> done

## Disclaimer

This project is provided for **research and educational purposes only**. It is not investment advice, financial advice, trading advice, or a recommendation to buy, sell, or hold any asset. Nothing in this repository — including the code, notebooks, ratios, rankings, backtest results, or any output produced by running it — should be construed as a solicitation or recommendation to engage in any trading or investment strategy.

Cryptocurrency and derivatives trading carry a high level of risk and may not be suitable for all investors. You can lose some or all of your capital. Past performance, including any backtested or simulated results produced by this software, is not indicative of future results. Backtests are simplified models and do not account for all real-world conditions (liquidity, slippage variability, exchange outages, funding costs, taxes, etc.).

The software is provided "as is" without warranty of any kind, express or implied, including but not limited to warranties of accuracy, merchantability, fitness for a particular purpose, or non-infringement. The author makes no representation that the analysis, data, or results are correct, complete, or current. Market data is fetched from third-party APIs and may contain errors or omissions.

By using this software you agree that you alone are responsible for any decisions you make and any losses you incur, and that the author and any contributors shall not be liable for any direct, indirect, incidental, consequential, or other damages arising from the use of, or inability to use, this software. You should consult a licensed financial advisor before making any investment decision.

## License

MIT
