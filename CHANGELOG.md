# Changelog

All notable changes to this project are documented in this file.

The format follows Keep a Changelog, and this project uses Semantic Versioning.

## [Unreleased]

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
