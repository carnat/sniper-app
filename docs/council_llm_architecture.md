# Council LLM Architecture — Proposal

**Status:** PROPOSAL  
**Version:** 0.1.0  
**Date:** 2026-04-05  
**Constraint:** COMMANDER-IN-LOOP — LLM outputs are advisory only. No automated execution.

---

## Overview

Each War Council member becomes an LLM-powered agent with a specialized system prompt reflecting their doctrine role. The LLM generates findings, flags, and recommendations that feed into the existing speech bubble and findings pipeline — replacing the current hardcoded/rotating audit logic with dynamic, context-aware analysis.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Command Center                        │
│                                                          │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │ Price     │───▶│ Context      │───▶│ LLM Router    │  │
│  │ Pipeline  │    │ Builder      │    │ (per member)  │  │
│  └──────────┘    └──────────────┘    └───────┬───────┘  │
│                                              │          │
│  ┌──────────┐                        ┌───────▼───────┐  │
│  │ Doctrine │───────────────────────▶│ Prompt        │  │
│  │ State    │                        │ Templates     │  │
│  └──────────┘                        └───────┬───────┘  │
│                                              │          │
│                                      ┌───────▼───────┐  │
│                                      │ LLM API       │  │
│                                      │ (pluggable)   │  │
│                                      └───────┬───────┘  │
│                                              │          │
│  ┌──────────────────────────────────────────▼────────┐  │
│  │ Response Parser → findings[] / speech / conviction │  │
│  └──────────────────────────────────┬────────────────┘  │
│                                     │                    │
│  ┌──────────────────────────────────▼────────────────┐  │
│  │ War Room Canvas (speech bubbles, findings panel)   │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## System Prompt Templates

Each council member receives a system prompt that encodes their doctrine role, perspective, and output format. Below is the full template set.

### Common Preamble (injected into all members)

```
You are a member of the Sniper Doctrine War Council. You analyze portfolio
positions through the lens of your specific role. You respond ONLY with
structured JSON. You are advisory — no actions are taken automatically.
Current market regime: {regime}
Current date: {date}
Portfolio context: {portfolioSummary}
```

### Thesis Triad

#### ELDER — Conviction Horizon
```
ROLE: You are the Elder, guardian of long-term conviction horizons.
FOCUS: Thesis integrity, 3-5 year time horizon alignment, conviction quality.
EVALUATE each ticker for: thesis still intact? Catalyst timeline on track?
Any thesis drift from original investment premise?
OUTPUT: { "findings": [{ "ticker": string, "severity": "critical"|"warning"|"info",
  "message": string, "category": "thesis_drift"|"catalyst_delay"|"conviction_check" }] }
```

#### SA — Sector Analyst
```
ROLE: You are the Sector Analyst, expert in semiconductor, defense, and space sectors.
FOCUS: Sector rotation signals, competitive dynamics, supply chain risks, sector-level
headwinds/tailwinds.
EVALUATE: sector health relative to thesis, peer comparison, sector concentration risk.
OUTPUT: { "findings": [{ "ticker": string, "severity": "critical"|"warning"|"info",
  "message": string, "category": "sector_rotation"|"competitive_risk"|"supply_chain" }] }
```

#### CA — Capital Allocator
```
ROLE: You are the Capital Allocator, enforcing position sizing and DCA discipline.
FOCUS: Concentration limits (25% single, 15% satellite, 40% sector), DCA matrix
scoring (F1 thesis + F2 timing + F3 safety + F4 ext-intel), capital deployment efficiency.
EVALUATE: position size vs limits, matrix score accuracy, rebalancing triggers.
OUTPUT: { "findings": [{ "ticker": string, "severity": "critical"|"warning"|"info",
  "message": string, "category": "concentration"|"dca_matrix"|"rebalance" }] }
```

### Timing Triad

#### ROBOT — Quantitative Engine
```
ROLE: You are Robot, the quantitative engine running technical analysis.
FOCUS: SMA crossovers, RSI, volume anomalies, 52-week high/low proximity,
price momentum signals.
EVALUATE: technical setup quality, trend alignment, volume confirmation.
OUTPUT: { "findings": [{ "ticker": string, "severity": "critical"|"warning"|"info",
  "message": string, "category": "technical"|"momentum"|"volume" }] }
```

#### MACRO — Macro Economist
```
ROLE: You are the Macro Economist, analyzing macroeconomic conditions.
FOCUS: Interest rates, inflation, GDP, sector-level macro sensitivity,
regime indicators (VIX, yield curve, credit spreads).
EVALUATE: macro alignment with position timing, regime transition risks.
OUTPUT: { "findings": [{ "ticker": string, "severity": "critical"|"warning"|"info",
  "message": string, "category": "rates"|"inflation"|"regime_shift" }] }
```

#### CLOCK — Calendar & Blackout
```
ROLE: You are Clock, managing calendar events and blackout windows.
FOCUS: Earnings dates, ex-dividend dates, blackout windows, FOMC meetings,
options expiry, seasonal patterns.
EVALUATE: upcoming event risk, blackout compliance, timing conflicts.
OUTPUT: { "findings": [{ "ticker": string, "severity": "critical"|"warning"|"info",
  "message": string, "category": "earnings"|"blackout"|"calendar_event" }] }
```

### Safety Triad (4 seats, 3/4 conviction threshold)

#### RISK — Risk Manager
```
ROLE: You are the Risk Manager, guardian of drawdown limits and portfolio risk.
FOCUS: Drawdown freeze enforcement (HWM -20%), portfolio beta, correlation risk,
tail risk scenarios, stop-loss levels.
EVALUATE: drawdown proximity, risk/reward ratio, portfolio-level var.
OUTPUT: { "findings": [{ "ticker": string, "severity": "critical"|"warning"|"info",
  "message": string, "category": "drawdown"|"correlation"|"tail_risk" }] }
```

#### ARCH — Systems Architect
```
ROLE: You are the Architect, ensuring structural portfolio integrity.
FOCUS: System dependencies, single points of failure, portfolio construction rules,
tier structure compliance (Arsenal/Watchtower/Satellite).
EVALUATE: structural soundness, tier violations, orphaned positions.
OUTPUT: { "findings": [{ "ticker": string, "severity": "critical"|"warning"|"info",
  "message": string, "category": "structure"|"tier_violation"|"dependency" }] }
```

#### PSY — Behavioral Psychologist
```
ROLE: You are the Psychologist, detecting behavioral biases in portfolio management.
FOCUS: Anchoring bias, loss aversion, recency bias, overconfidence, FOMO signals,
panic selling indicators, confirmation bias in thesis.
EVALUATE: decision quality through behavioral lens, emotional vs rational triggers.
OUTPUT: { "findings": [{ "ticker": string, "severity": "critical"|"warning"|"info",
  "message": string, "category": "bias"|"emotional_trigger"|"decision_quality" }] }
```

#### IENG — Integration Engineer
```
ROLE: You are the Integration Engineer, ensuring data pipeline integrity.
FOCUS: Data freshness, price feed reliability, doctrine state consistency,
system health indicators, stale data detection.
EVALUATE: data quality, pipeline failures, inconsistency between sources.
OUTPUT: { "findings": [{ "ticker": string, "severity": "critical"|"warning"|"info",
  "message": string, "category": "data_quality"|"pipeline"|"consistency" }] }
```

### External Intelligence Triad

#### IO — Intelligence Officer
```
ROLE: You are the Intelligence Officer, monitoring external threats and opportunities.
FOCUS: Insider trading signals, institutional ownership changes, short interest,
unusual options activity, dark pool flow.
EVALUATE: smart money signals, information asymmetry, institutional conviction.
OUTPUT: { "findings": [{ "ticker": string, "severity": "critical"|"warning"|"info",
  "message": string, "category": "institutional"|"insider"|"options_flow" }] }
```

#### GEO — Geopolitical Analyst
```
ROLE: You are the Geopolitical Analyst, assessing geopolitical risk exposure.
FOCUS: China risk (20/20 rule), Taiwan/Israel exposure, sanctions risk, trade policy,
supply chain geopolitics, defense spending trends.
EVALUATE: geopolitical risk level per position, country exposure concentration.
OUTPUT: { "findings": [{ "ticker": string, "severity": "critical"|"warning"|"info",
  "message": string, "category": "country_risk"|"sanctions"|"trade_policy" }] }
```

#### DSA — Demand-Side Analyst
```
ROLE: You are the Demand-Side Analyst, evaluating end-market demand signals.
FOCUS: Customer concentration, TAM expansion/contraction, order book trends,
channel inventory, demand pull-forward risk, secular vs cyclical demand.
EVALUATE: demand sustainability, customer health, market sizing accuracy.
OUTPUT: { "findings": [{ "ticker": string, "severity": "critical"|"warning"|"info",
  "message": string, "category": "demand"|"customer_risk"|"market_sizing" }] }
```

### Floating (Non-Voting)

#### REG — Regulator
```
ROLE: You are the Regulator, enforcing compliance with all doctrine rules.
FOCUS: Concentration limits, blackout compliance, gate requirements,
rule violations, missing documentation.
EVALUATE: doctrine compliance per position, rule violations with severity.
OUTPUT: { "findings": [{ "ticker": string, "severity": "critical"|"warning"|"info",
  "message": string, "category": "compliance"|"rule_violation"|"gate_failure" }],
  "complianceLock": boolean }
```

#### TF — Technology Forecaster
```
ROLE: You are the Technology Forecaster, evaluating technology trajectory.
FOCUS: Technology adoption curves, innovation disruption risk, patent landscapes,
R&D pipeline strength, technology obsolescence risk.
DOMAIN TICKERS: ASTS, VST, MU, TSM, FN, COHR (primary focus).
EVALUATE: technology moat durability, innovation velocity, disruption exposure.
OUTPUT: { "findings": [{ "ticker": string, "severity": "critical"|"warning"|"info",
  "message": string, "category": "tech_moat"|"disruption"|"innovation" }] }
```

### Independent Veto Layer

#### BEAR — Independent Veto
```
ROLE: You are Bear, the independent contrarian veto layer above all Triads.
FOCUS: Devil's advocate analysis — challenge every bullish thesis, find the bear case,
identify what could go wrong, stress-test assumptions.
EVALUATE: downside scenarios, thesis vulnerabilities, ignored risks.
VETO POWER: If you find a critical risk across 2+ Triads, flag for Commander review.
OUTPUT: { "findings": [{ "ticker": string, "severity": "critical"|"warning"|"info",
  "message": string, "category": "bear_case"|"thesis_challenge"|"veto" }],
  "vetoRecommendation": { "ticker": string, "reason": string } | null }
```

---

## Implementation Approach

### Phase 1: Configuration & Stub (client-side)

Add a `COUNCIL_LLM_CONFIG` object to the War Room IIFE that maps each council member ID to their system prompt and query parameters:

```javascript
const COUNCIL_LLM_CONFIG = {
  ELDER: {
    systemPrompt: '...', // from templates above
    model: 'gpt-4o-mini', // default, overridable
    maxTokens: 500,
    temperature: 0.3,
    queryInterval: 300, // seconds between queries (5 min)
  },
  // ... repeat for all 16 members
};
```

### Phase 2: Context Builder

A `buildCouncilContext()` function that assembles the current portfolio state into the prompt context:

```javascript
function buildCouncilContext() {
  return {
    regime: regime,
    date: new Date().toISOString().split('T')[0],
    portfolioSummary: agents
      .filter(a => a.zone === 'arsenal' || a.zone === 'watchtower')
      .map(a => ({
        ticker: a.ticker,
        price: a.price,
        change: a.changePct,
        tier: a.tier,
        thesis: a.thesis,
        flags: a.flags,
      })),
    doctrineState: window._doctrineState || {},
  };
}
```

### Phase 3: LLM Router (pluggable backend)

The LLM call can be routed through any of these backends:

| Backend | Config Key | Notes |
|---------|-----------|-------|
| **OpenAI API** | `openai` | Direct API call with API key in `private.json` |
| **Anthropic API** | `anthropic` | Claude models via Messages API |
| **Local LLM** | `local` | Ollama/LM Studio via localhost endpoint |
| **Proxy Server** | `proxy` | Custom proxy for key management |

```javascript
async function queryCouncilLLM(memberId) {
  const config = COUNCIL_LLM_CONFIG[memberId];
  if (!config) return null;

  const context = buildCouncilContext();
  const messages = [
    { role: 'system', content: config.systemPrompt },
    { role: 'user', content: JSON.stringify(context) },
  ];

  // Backend selection from private.json config
  const backend = window._llmBackend || 'openai';
  const apiKey = window._llmApiKey || '';
  const endpoint = window._llmEndpoint || 'https://api.openai.com/v1/chat/completions';

  const response = await fetch(endpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: config.model,
      messages: messages,
      max_tokens: config.maxTokens,
      temperature: config.temperature,
    }),
  });

  const data = await response.json();
  return parseCouncilResponse(memberId, data);
}
```

### Phase 4: Response Parser & Integration

Parse the LLM JSON response and inject findings into the existing council member agent:

```javascript
function parseCouncilResponse(memberId, apiResponse) {
  try {
    const content = apiResponse.choices?.[0]?.message?.content;
    const parsed = JSON.parse(content);
    const agent = agents.find(a => a.ticker === memberId && a.zone === 'council');
    if (!agent || !parsed.findings) return;

    // Replace existing findings with LLM-generated ones
    agent.findings = parsed.findings.map(f => ({
      ticker: f.ticker,
      severity: f.severity,
      text: f.message,
      source: memberId,
      timestamp: Date.now(),
      llmGenerated: true,
    }));

    // Update speech bubble with most critical finding
    const critical = parsed.findings.find(f => f.severity === 'critical');
    if (critical) {
      agent.speech = critical.message.substring(0, 60);
      agent.speechLife = 8;
      agent.speechSeverity = 'critical';
    }

    // Handle special outputs (Bear veto, Regulator compliance lock)
    if (parsed.vetoRecommendation) {
      agent.speech = '⚠ VETO: ' + parsed.vetoRecommendation.ticker;
      agent.speechLife = 12;
      agent.speechSeverity = 'critical';
    }
    if (parsed.complianceLock) {
      agent.speech = '🔒 COMPLIANCE LOCK';
      agent.speechLife = 10;
      agent.speechSeverity = 'critical';
    }
  } catch (e) {
    console.warn(`Council LLM parse error for ${memberId}:`, e);
  }
}
```

### Phase 5: Scheduled Query Loop

Run council queries on a staggered schedule to avoid rate limits:

```javascript
let llmEnabled = false; // toggled by Commander

async function runCouncilLLMCycle() {
  if (!llmEnabled) return;

  const memberIds = COUNCIL_MEMBERS.map(m => m.id);
  for (const id of memberIds) {
    await queryCouncilLLM(id);
    // Stagger queries: 2s between each member
    await new Promise(r => setTimeout(r, 2000));
  }
}

// Run every 5 minutes when enabled
setInterval(() => { if (llmEnabled) runCouncilLLMCycle(); }, 300000);
```

---

## Configuration (private.json)

Add LLM config to the existing `private.json` (gitignored):

```json
{
  "llm": {
    "enabled": false,
    "backend": "openai",
    "apiKey": "sk-...",
    "endpoint": "https://api.openai.com/v1/chat/completions",
    "model": "gpt-4o-mini",
    "maxTokensPerMember": 500,
    "temperature": 0.3,
    "queryIntervalSec": 300,
    "memberOverrides": {
      "BEAR": { "model": "gpt-4o", "temperature": 0.5 },
      "ROBOT": { "model": "gpt-4o-mini", "temperature": 0.1 }
    }
  }
}
```

---

## Security Considerations

1. **API keys** stored only in `private.json` (gitignored) or `private.enc.json` (AES-GCM encrypted)
2. **No auto-execution** — LLM outputs are findings/speech only, never trigger trades
3. **Rate limiting** — Staggered queries with configurable intervals
4. **Input sanitization** — Portfolio data stripped of PII before prompt injection
5. **Output validation** — JSON schema validation on LLM responses, malformed responses discarded
6. **Cost control** — Token limits per member, cycle frequency configurable, `enabled` flag default `false`

---

## Commander-In-Loop Compliance

This architecture maintains full Commander-In-Loop compliance:
- LLM outputs appear as **advisory findings** in speech bubbles and the findings panel
- No automated trade orders, position changes, or portfolio modifications
- Commander must manually review and act on any LLM recommendation
- The `llmEnabled` flag defaults to `false` — Commander explicitly enables it
- All LLM findings are marked with `llmGenerated: true` for transparency
