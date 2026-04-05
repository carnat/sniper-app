# Council LLM Architecture — Triad-Grouped Multi-Turn

**Status:** APPROVED  
**Version:** 0.2.0  
**Date:** 2026-04-05  
**Constraint:** COMMANDER-IN-LOOP — LLM outputs are advisory only. No automated execution.

---

## Overview

The War Council runs **5 LLM calls per cycle** — one per Triad plus a final Bear/floating call. Within each call, member personas respond sequentially so that later members read earlier members' outputs before responding. This enables real intra-Triad deliberation while keeping costs 70% lower than per-member injection (5 calls vs 16). LLM-generated findings feed into the existing speech bubble and findings pipeline, replacing the current hardcoded/rotating audit logic with dynamic, context-aware analysis.

---

## Decision Rationale

Three approaches were evaluated before selecting the Triad-grouped pattern:

| Dimension | A: Per-Member (16 calls) | B1: Multi-Turn by Triad (5 calls) | B2: Mega-Prompt (1 call) |
|-----------|--------------------------|-------------------------------------|--------------------------|
| API calls per cycle | 16 | **5** | 1 |
| Cost per cycle | High | **Low** | Lowest |
| Role fidelity | ✅ High | **Medium** (mitigated by prompt design) | ❌ Low |
| Intra-Triad deliberation | ❌ None | **✅ Real** (sequential within call) | ❌ None |
| Triad consensus fidelity | ❌ Simulated | **✅ Enforced** in prompt | ❌ None |
| Failure isolation | ✅ Per-member | **✅ Per-Triad** | ❌ Single point |
| Bear veto quality | ❌ Blind (no cross-Triad context) | **✅ Contextual** (reads all 4 Triad outputs) | ❌ Blind |
| War Room rendering fit | ✅ Native | **✅ Native** (parsed to per-member findings) | ✅ Native |
| Fits static HTML constraint | ✅ Yes | **✅ Yes** | ✅ Yes |

**Selected: B1 — Triad-grouped multi-turn.** Rationale:
- Faithful to doctrine's deliberation model (members within a Triad see each other)
- BEAR receives full cross-Triad context before issuing veto — matches doctrine intent
- Challenge Block mechanics are possible within a Triad turn
- Per-Triad failure isolation (one Triad failing doesn't kill all findings)
- 70% cost reduction vs per-member approach

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Command Center                            │
│                                                              │
│  ┌──────────┐    ┌──────────────┐                           │
│  │ Price     │───▶│ Context      │                           │
│  │ Pipeline  │    │ Builder      │                           │
│  └──────────┘    └──────┬───────┘                           │
│                         │                                    │
│  ┌──────────┐           │         ┌────────────────────┐    │
│  │ Doctrine │───────────┤         │ Triad Orchestrator │    │
│  │ State    │           │         │                    │    │
│  └──────────┘           ▼         │  Call 1: THESIS    │    │
│                   ┌───────────┐   │   ELDER → SA → CA  │    │
│                   │ Triad     │──▶│                    │    │
│                   │ Prompt    │   │  Call 2: TIMING    │    │
│                   │ Templates │   │   ROBOT → MACRO →  │    │
│                   └───────────┘   │   CLOCK            │    │
│                                   │                    │    │
│                                   │  Call 3: SAFETY    │    │
│                                   │   RISK → ARCH →    │    │
│                                   │   PSY → IENG       │    │
│                                   │                    │    │
│                                   │  Call 4: EXT-INTEL  │    │
│                                   │   IO → GEO → DSA   │    │
│                                   │                    │    │
│                                   │  Call 5: BEAR+FLT  │    │
│                                   │   REG + TF + BEAR  │    │
│                                   │   (reads calls 1-4)│    │
│                                   └────────┬───────────┘    │
│                                            │                 │
│                                   ┌────────▼───────────┐    │
│                                   │ LLM API (pluggable)│    │
│                                   └────────┬───────────┘    │
│                                            │                 │
│  ┌─────────────────────────────────────────▼──────────────┐ │
│  │ Response Parser → per-member findings[] / speech /     │ │
│  │ conviction / vetoRecommendation / complianceLock       │ │
│  └─────────────────────────────────┬─────────────────────┘ │
│                                    │                        │
│  ┌─────────────────────────────────▼─────────────────────┐ │
│  │ War Room Canvas (speech bubbles, findings panel)       │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## System Prompt Templates

Each LLM call covers an entire Triad. Members respond sequentially within the call — each member reads the prior members' outputs before responding. This enables real deliberation within a Triad.

### Common Preamble (injected into all Triad calls)

```
You are the Sniper Doctrine War Council — {triadName} Triad.
You will respond as each member in sequence. Each member reads the prior
members' outputs before responding. You respond ONLY with structured JSON.
You are advisory — no actions are taken automatically.

Current market regime: {regime}
Current date: {date}
Portfolio context: {portfolioSummary}

RESPOND as a JSON object with one key per member ID. Each member's value
follows the standard finding schema.
```

### Call 1 — Thesis Triad (ELDER → SA → CA)

```
TRIAD: THESIS — evaluates conviction quality and thesis integrity.
CONVICTION THRESHOLD: 2 of 3 members must flag a ticker for Triad-level finding.

Respond as each member in sequence. Each member reads prior members' output.

ELDER (first):
  ROLE: Guardian of long-term conviction horizons.
  FOCUS: Thesis integrity, 3-5 year time horizon alignment, conviction quality.
  EVALUATE each ticker: thesis still intact? Catalyst timeline on track?
  Any thesis drift from original investment premise?

SA (second — reads ELDER's output):
  ROLE: Sector expert in semiconductor, defense, and space sectors.
  FOCUS: Sector rotation signals, competitive dynamics, supply chain risks.
  EVALUATE: sector health relative to thesis, peer comparison, concentration risk.
  RESPOND TO ELDER: confirm, challenge, or add sector-specific context.

CA (third — reads ELDER + SA):
  ROLE: Capital Allocator, enforcing position sizing and DCA discipline.
  FOCUS: Concentration limits (25% single, 15% satellite, 40% sector), DCA matrix
  scoring (F1 thesis + F2 timing + F3 safety + F4 ext-intel), capital deployment.
  EVALUATE: position size vs limits, matrix score accuracy, rebalancing triggers.
  SYNTHESIZE: Given ELDER and SA inputs, flag any consensus or disagreement.

OUTPUT:
{
  "triad": "thesis",
  "ELDER": { "findings": [{ "ticker": str, "severity": "critical"|"warning"|"info",
    "message": str, "category": "thesis_drift"|"catalyst_delay"|"conviction_check" }] },
  "SA": { "findings": [{ "ticker": str, "severity": "critical"|"warning"|"info",
    "message": str, "category": "sector_rotation"|"competitive_risk"|"supply_chain" }] },
  "CA": { "findings": [{ "ticker": str, "severity": "critical"|"warning"|"info",
    "message": str, "category": "concentration"|"dca_matrix"|"rebalance" }] },
  "triadConsensus": [{ "ticker": str, "flaggedBy": ["ELDER","SA"|"CA"], "summary": str }]
}
```

### Call 2 — Timing Triad (ROBOT → MACRO → CLOCK)

```
TRIAD: TIMING — evaluates market timing, technical signals, and calendar risk.
CONVICTION THRESHOLD: 2 of 3 members must flag a ticker for Triad-level finding.

ROBOT (first):
  ROLE: Quantitative engine running technical analysis.
  FOCUS: SMA crossovers, RSI, volume anomalies, 52-week high/low proximity,
  price momentum signals.
  EVALUATE: technical setup quality, trend alignment, volume confirmation.

MACRO (second — reads ROBOT's output):
  ROLE: Macro Economist analyzing macroeconomic conditions.
  FOCUS: Interest rates, inflation, GDP, sector-level macro sensitivity,
  regime indicators (VIX, yield curve, credit spreads).
  EVALUATE: macro alignment with ROBOT's technical signals, regime transition risks.
  RESPOND TO ROBOT: does macro environment support or contradict technical setup?

CLOCK (third — reads ROBOT + MACRO):
  ROLE: Calendar & Blackout manager.
  FOCUS: Earnings dates, ex-dividend dates, blackout windows, FOMC meetings,
  options expiry, seasonal patterns.
  EVALUATE: upcoming event risk, blackout compliance, timing conflicts.
  SYNTHESIZE: Given ROBOT and MACRO inputs, flag any timing alignment or conflict.

OUTPUT:
{
  "triad": "timing",
  "ROBOT": { "findings": [...] },
  "MACRO": { "findings": [...] },
  "CLOCK": { "findings": [...] },
  "triadConsensus": [{ "ticker": str, "flaggedBy": [...], "summary": str }]
}
```

### Call 3 — Safety Triad (RISK → ARCH → PSY → IENG)

```
TRIAD: SAFETY — evaluates portfolio risk, structural integrity, and behavioral bias.
CONVICTION THRESHOLD: 3 of 4 members must flag a ticker (elevated threshold).

RISK (first):
  ROLE: Risk Manager, guardian of drawdown limits and portfolio risk.
  FOCUS: Drawdown freeze enforcement (HWM -20%), portfolio beta, correlation risk,
  tail risk scenarios, stop-loss levels.
  EVALUATE: drawdown proximity, risk/reward ratio, portfolio-level VaR.

ARCH (second — reads RISK):
  ROLE: Systems Architect, ensuring structural portfolio integrity.
  FOCUS: System dependencies, single points of failure, portfolio construction rules,
  tier structure compliance (Arsenal/Watchtower/Satellite).
  EVALUATE: structural soundness, tier violations, orphaned positions.
  RESPOND TO RISK: structural risks that compound RISK's identified drawdown risks.

PSY (third — reads RISK + ARCH):
  ROLE: Behavioral Psychologist, detecting behavioral biases.
  FOCUS: Anchoring bias, loss aversion, recency bias, overconfidence, FOMO signals,
  panic selling indicators, confirmation bias in thesis.
  EVALUATE: Are RISK/ARCH findings driven by rational analysis or emotional triggers?

IENG (fourth — reads RISK + ARCH + PSY):
  ROLE: Integration Engineer, ensuring data pipeline integrity.
  FOCUS: Data freshness, price feed reliability, doctrine state consistency,
  system health indicators, stale data detection.
  EVALUATE: data quality behind the findings above. Are inputs reliable?
  SYNTHESIZE: Given all 3 prior inputs, flag consensus (3/4 required for conviction).

OUTPUT:
{
  "triad": "safety",
  "RISK": { "findings": [...] },
  "ARCH": { "findings": [...] },
  "PSY": { "findings": [...] },
  "IENG": { "findings": [...] },
  "triadConsensus": [{ "ticker": str, "flaggedBy": [...], "summary": str }]
}
```

### Call 4 — External Intelligence Triad (IO → GEO → DSA)

```
TRIAD: EXTERNAL INTELLIGENCE — evaluates external threats and opportunities.
CONVICTION THRESHOLD: 2 of 3 members must flag a ticker for Triad-level finding.

IO (first):
  ROLE: Intelligence Officer monitoring external threats and opportunities.
  FOCUS: Insider trading signals, institutional ownership changes, short interest,
  unusual options activity, dark pool flow.
  EVALUATE: smart money signals, information asymmetry, institutional conviction.

GEO (second — reads IO):
  ROLE: Geopolitical Analyst assessing geopolitical risk exposure.
  FOCUS: China risk (20/20 rule), Taiwan/Israel exposure, sanctions risk, trade policy,
  supply chain geopolitics, defense spending trends.
  EVALUATE: geopolitical risk per position, country exposure concentration.
  RESPOND TO IO: do institutional flows align with geopolitical risk signals?

DSA (third — reads IO + GEO):
  ROLE: Demand-Side Analyst evaluating end-market demand signals.
  FOCUS: Customer concentration, TAM expansion/contraction, order book trends,
  channel inventory, demand pull-forward risk, secular vs cyclical demand.
  EVALUATE: demand sustainability, customer health, market sizing accuracy.
  SYNTHESIZE: Given IO and GEO, flag demand-side confirmation or contradiction.

OUTPUT:
{
  "triad": "extintel",
  "IO": { "findings": [...] },
  "GEO": { "findings": [...] },
  "DSA": { "findings": [...] },
  "triadConsensus": [{ "ticker": str, "flaggedBy": [...], "summary": str }]
}
```

### Call 5 — Bear + Floating (REG + TF + BEAR)

This call receives the `triadConsensus` arrays from calls 1-4 as additional context, enabling BEAR to make an informed cross-Triad veto decision.

```
CONTEXT: You receive all 4 Triad consensus outputs plus the full portfolio context.
Previous Triad findings: {triadConsensusFromCalls1to4}

REG (first):
  ROLE: Regulator enforcing compliance with all doctrine rules (non-voting).
  FOCUS: Concentration limits, blackout compliance, gate requirements,
  rule violations, missing documentation.
  EVALUATE: doctrine compliance per position, rule violations with severity.
  OUTPUT includes: "complianceLock": boolean

TF (second — reads REG):
  ROLE: Technology Forecaster evaluating technology trajectory (non-voting).
  FOCUS: Technology adoption curves, disruption risk, patent landscapes, R&D pipeline.
  DOMAIN TICKERS: ASTS, VST, MU, TSM, FN, COHR (primary focus).
  EVALUATE: technology moat durability, innovation velocity, disruption exposure.
  NOTE: Self-trigger only — findings generated only for domain tickers.

BEAR (third — reads REG + TF + ALL 4 TRIAD CONSENSUS OUTPUTS):
  ROLE: Independent contrarian veto layer above all Triads.
  FOCUS: Devil's advocate — challenge every bullish thesis, find the bear case,
  identify what could go wrong, stress-test assumptions.
  EVALUATE: downside scenarios, thesis vulnerabilities, ignored risks.
  CROSS-TRIAD ANALYSIS: Review all 4 Triad consensus outputs. If a critical risk
  spans 2+ Triads for the same ticker, issue a vetoRecommendation.
  VETO POWER: Flag for Commander review if cross-Triad critical risk detected.

OUTPUT:
{
  "triad": "bear_floating",
  "REG": { "findings": [...], "complianceLock": boolean },
  "TF": { "findings": [...] },
  "BEAR": { "findings": [...],
    "vetoRecommendation": { "ticker": str, "reason": str, "triadsAffected": [...] } | null },
  "crossTriadConsensus": [{ "ticker": str, "triads": [...], "summary": str }]
}
```

---

## Implementation Approach

### Phase 1: Configuration & Stub (client-side)

Add a `TRIAD_LLM_CONFIG` object to the War Room IIFE that maps each Triad to its members, prompt, and query parameters:

```javascript
const TRIAD_LLM_CONFIG = {
  thesis: {
    members: ['ELDER', 'SA', 'CA'],
    convictionThreshold: 2,
    model: 'gpt-4o-mini',
    maxTokens: 1200,  // ~400 per member
    temperature: 0.3,
  },
  timing: {
    members: ['ROBOT', 'MACRO', 'CLOCK'],
    convictionThreshold: 2,
    model: 'gpt-4o-mini',
    maxTokens: 1200,
    temperature: 0.3,
  },
  safety: {
    members: ['RISK', 'ARCH', 'PSY', 'IENG'],
    convictionThreshold: 3,  // 3 of 4 required
    model: 'gpt-4o-mini',
    maxTokens: 1600,  // ~400 per member
    temperature: 0.3,
  },
  extintel: {
    members: ['IO', 'GEO', 'DSA'],
    convictionThreshold: 2,
    model: 'gpt-4o-mini',
    maxTokens: 1200,
    temperature: 0.3,
  },
  bear_floating: {
    members: ['REG', 'TF', 'BEAR'],
    convictionThreshold: null,  // Bear uses cross-Triad consensus
    model: 'gpt-4o',  // stronger model for cross-Triad analysis
    maxTokens: 1500,
    temperature: 0.5,  // higher creativity for bear case
    dependsOn: ['thesis', 'timing', 'safety', 'extintel'],
  },
};
```

### Phase 2: Context Builder

A `buildCouncilContext()` function assembles the current portfolio state. For Call 5, it also includes the `triadConsensus` outputs from Calls 1-4:

```javascript
function buildCouncilContext(priorTriadResults) {
  const base = {
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

  // For Call 5 (bear_floating), inject prior Triad consensus
  if (priorTriadResults) {
    base.priorTriadConsensus = priorTriadResults.map(r => ({
      triad: r.triad,
      consensus: r.triadConsensus || [],
    }));
  }

  return base;
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
async function queryTriadLLM(triadId, priorTriadResults) {
  const config = TRIAD_LLM_CONFIG[triadId];
  if (!config) return null;

  const context = buildCouncilContext(
    config.dependsOn ? priorTriadResults : null
  );

  const systemPrompt = TRIAD_PROMPTS[triadId]; // from templates above
  const messages = [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: JSON.stringify(context) },
  ];

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
  return parseTriadResponse(triadId, data);
}
```

### Phase 4: Response Parser & Integration

Parse the Triad-level JSON response and distribute findings to individual council member agents:

```javascript
function parseTriadResponse(triadId, apiResponse) {
  try {
    const content = apiResponse.choices?.[0]?.message?.content;
    const parsed = JSON.parse(content);
    const config = TRIAD_LLM_CONFIG[triadId];

    // Distribute findings to each member agent
    for (const memberId of config.members) {
      const memberData = parsed[memberId];
      if (!memberData || !memberData.findings) continue;

      const agent = agents.find(a => a.ticker === memberId && a.zone === 'council');
      if (!agent) continue;

      agent.findings = memberData.findings.map(f => ({
        ticker: f.ticker,
        severity: f.severity,
        text: f.message,
        source: memberId,
        triad: triadId,
        timestamp: Date.now(),
        llmGenerated: true,
      }));

      // Update speech bubble with most critical finding
      const critical = memberData.findings.find(f => f.severity === 'critical');
      if (critical) {
        agent.speech = critical.message.substring(0, 60);
        agent.speechLife = 8;
        agent.speechSeverity = 'critical';
      }

      // Handle special outputs
      if (memberData.vetoRecommendation) {
        agent.speech = '⚠ VETO: ' + memberData.vetoRecommendation.ticker;
        agent.speechLife = 12;
        agent.speechSeverity = 'critical';
      }
      if (memberData.complianceLock) {
        agent.speech = '🔒 COMPLIANCE LOCK';
        agent.speechLife = 10;
        agent.speechSeverity = 'critical';
      }
    }

    // Return parsed result for downstream Triads (especially Call 5)
    return {
      triad: triadId,
      triadConsensus: parsed.triadConsensus || parsed.crossTriadConsensus || [],
      raw: parsed,
    };
  } catch (e) {
    console.warn(`Council LLM parse error for Triad ${triadId}:`, e);
    return { triad: triadId, triadConsensus: [], raw: null, error: e.message };
  }
}
```

### Phase 5: Scheduled Query Loop (5 calls per cycle)

Run Triad queries in sequence. Calls 1-4 run in parallel (independent), Call 5 waits for all 4 to finish:

```javascript
let llmEnabled = false; // toggled by Commander

async function runCouncilLLMCycle() {
  if (!llmEnabled) return;

  // Calls 1-4: Independent Triads run in parallel
  const [thesis, timing, safety, extintel] = await Promise.allSettled([
    queryTriadLLM('thesis'),
    queryTriadLLM('timing'),
    queryTriadLLM('safety'),
    queryTriadLLM('extintel'),
  ]);

  // Collect successful results for Call 5
  const priorResults = [thesis, timing, safety, extintel]
    .filter(r => r.status === 'fulfilled' && r.value)
    .map(r => r.value);

  // Call 5: Bear + Floating — receives all prior Triad consensus
  await queryTriadLLM('bear_floating', priorResults);
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
    "temperature": 0.3,
    "queryIntervalSec": 300,
    "triadOverrides": {
      "bear_floating": {
        "model": "gpt-4o",
        "temperature": 0.5,
        "maxTokens": 1500
      },
      "safety": {
        "maxTokens": 1600
      }
    }
  }
}
```

---

## Security Considerations

1. **API keys** stored only in `private.json` (gitignored) or `private.enc.json` (AES-GCM encrypted)
2. **No auto-execution** — LLM outputs are findings/speech only, never trigger trades
3. **Rate limiting** — 5 calls per cycle (vs 16), Calls 1-4 parallel, configurable interval
4. **Input sanitization** — Portfolio data stripped of PII before prompt injection
5. **Output validation** — JSON schema validation on LLM responses, malformed responses discarded
6. **Cost control** — Token limits per Triad, cycle frequency configurable, `enabled` flag default `false`
7. **Failure isolation** — Per-Triad error handling; one Triad failing doesn't block others

---

## Commander-In-Loop Compliance

This architecture maintains full Commander-In-Loop compliance:
- LLM outputs appear as **advisory findings** in speech bubbles and the findings panel
- No automated trade orders, position changes, or portfolio modifications
- Commander must manually review and act on any LLM recommendation
- The `llmEnabled` flag defaults to `false` — Commander explicitly enables it
- All LLM findings are marked with `llmGenerated: true` for transparency
- BEAR veto is advisory — Commander reviews flagged cross-Triad risks, not auto-enforced
