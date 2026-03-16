# /sniper:thesis — Ticker Thesis Check

Run thesis evaluation for a ticker: fetch stock info, check Alpha Filters, evaluate thesis tripwires, output verdict.

Usage: /sniper:thesis [TICKER]

## Protocol
1. Fetch `get_stock_info` via Yahoo Finance MCP
2. Check Alpha Filters F1–F4 (from doctrine-core skill)
3. Evaluate thesis tripwires for the specific ticker
4. Output: HOLD / REVIEW / TRIM / EXIT verdict with rationale

**STATUS: PLACEHOLDER — populate in Session 2**
