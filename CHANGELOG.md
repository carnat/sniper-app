# Changelog

All notable changes to this project are documented in this file.

The format follows Keep a Changelog, and this project uses Semantic Versioning.

## [Unreleased]

### Fixed
- Price pipeline: per-ticker freshness check. A ticker whose last close lags the feed's newest session is flagged `stale` with `sessions_behind`, `market_date` and `data_quality` codes; the feed adds `reference_market_date` and `stale_tickers` (QRVO stuck on 2026-10-05 inside the 2026-10-22 refresh).
- Price pipeline: last-session volume is read from the close's own session; padded NaN/0 volume is reported as unavailable (`null`), not zero. ADV averages only real sessions. A stale session never sets `gate_met`.
- Command Center: Watchtower stale-quote banner and badges, dimmed stale prices with dated change, `VOL n/a` for missing volume; Arsenal stale badge; Signal Convergence STALE QUOTE warning; market-close label uses the feed's newest session.

## [1.3.0] - 2026-02-19

### Added
- Transaction correction flow with append-only reversal records.
- Yahoo-style CSV import improvements, rebuild controls, and duplicate handling toggle.
- Analytics expansion: attribution, risk panel v1, what-if rebalancing, signal scoring.
- Portfolio calendar tab.
- Attribution snapshot history and scenario backtesting.
- Scenario library: save/replay/compare and daily auto snapshots.
- Lot method policies (FIFO/LIFO/AVERAGE) per asset class.

### Changed
- Portfolio hydration now rebuilds holdings from persisted transaction history when secrets holdings are empty.
- App version now reads from a dedicated `VERSION` file.

### Fixed
- Resolved blank portfolio/news state after session reset.
- Improved import workflows where transaction history exists but visible holdings are empty.
