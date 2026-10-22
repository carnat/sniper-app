"""Per-ticker quote freshness: one stale symbol inside a freshly refreshed feed.

Regression for the 2026-10-22 06:46:33Z refresh, where QRVO still carried its
2026-10-05 close (volume 0, no SMA50/SMA200) while every other ticker had the
2026-10-21 session and the feed's last_updated said Oct 22.
"""

import contextlib
import io
import json
import shutil
import subprocess
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.fetch_prices import (  # noqa: E402
    DQ_SMA200_MISSING,
    DQ_SMA50_MISSING,
    DQ_STALE_QUOTE,
    DQ_VOLUME_MISSING,
    _positive_volume,
    annotate_freshness,
    build_adv_section,
    build_output,
    build_prices_from_history,
    business_days_between,
)

FIXTURE = ROOT / 'tests' / 'fixtures' / 'prices_2026-10-22_qrvo_stale.json'
INDEX_HTML = ROOT / 'command_center' / 'index.html'

FRESH_SESSION = '2026-10-21'
QRVO_SESSION = '2026-10-05'
QRVO_SESSIONS_BEHIND = 12   # weekdays Oct 6 .. Oct 21


def _history(fresh=('VRT', 'PLTR', 'TSEM', 'SPY'), qrvo_valid_closes=30):
    """Build a yf.download-shaped (metric, ticker) frame ending 2026-10-21.

    Fresh tickers have 250 full sessions. QRVO's last `qrvo_valid_closes`
    closes end on 2026-10-05; after that yfinance-style padding: NaN close,
    0 volume. Before its valid window QRVO is all NaN (thin history).
    """
    idx = pd.bdate_range(end=FRESH_SESSION, periods=250)
    cols, data = [], {}
    for i, sym in enumerate(fresh):
        base = 100.0 + 10 * i
        close = base + np.arange(250) * 0.1
        data[('Close', sym)] = close
        data[('High', sym)] = close + 1
        data[('Low', sym)] = close - 1
        data[('Volume', sym)] = np.full(250, 1_000_000.0 + i)
    qrvo_last = idx.get_loc(pd.Timestamp(QRVO_SESSION))
    close = np.full(250, np.nan)
    vol = np.full(250, np.nan)
    start = qrvo_last - qrvo_valid_closes + 1
    close[start:qrvo_last + 1] = np.linspace(100.0, 114.17, qrvo_valid_closes)
    close[qrvo_last - 1] = 107.98   # so the Oct 5 change is +5.73%
    vol[start:qrvo_last + 1] = 1_500_000.0
    vol[qrvo_last + 1:] = 0.0       # padded "observations" with no trade data
    data[('Close', 'QRVO')] = close
    data[('High', 'QRVO')] = close + 1
    data[('Low', 'QRVO')] = close - 1
    data[('Volume', 'QRVO')] = vol
    cols = pd.MultiIndex.from_tuples(list(data.keys()))
    return pd.DataFrame({k: v for k, v in data.items()}, index=idx)[cols]


def _build(raw, macro=None):
    """build_prices_from_history without its per-ticker WARN noise on stderr."""
    with contextlib.redirect_stderr(io.StringIO()):
        return build_prices_from_history(raw, macro if macro is not None else _macro())


def _macro():
    idx = pd.bdate_range(end='2026-10-22', periods=5)
    data = {('Close', '^VIX'): [17.0, 17.5, 17.3, 17.31, 16.68],
            ('Close', 'THB=X'): [33.5, 33.6, 33.6, 33.6, 33.66]}
    return pd.DataFrame(data, index=idx)


class TestBusinessDays(unittest.TestCase):
    def test_qrvo_gap(self):
        self.assertEqual(business_days_between(QRVO_SESSION, FRESH_SESSION), QRVO_SESSIONS_BEHIND)

    def test_same_day_and_reverse(self):
        self.assertEqual(business_days_between(FRESH_SESSION, FRESH_SESSION), 0)
        self.assertEqual(business_days_between('2026-10-22', FRESH_SESSION), 0)

    def test_weekend_is_one_session(self):
        # Friday close vs Monday close is one session, not three days
        self.assertEqual(business_days_between('2026-10-16', '2026-10-19'), 1)

    def test_missing_dates(self):
        self.assertEqual(business_days_between(None, FRESH_SESSION), 0)


class TestPositiveVolume(unittest.TestCase):
    def test_zero_is_unavailable_not_zero(self):
        self.assertIsNone(_positive_volume(0))
        self.assertIsNone(_positive_volume(0.0))

    def test_nan_and_none(self):
        self.assertIsNone(_positive_volume(float('nan')))
        self.assertIsNone(_positive_volume(None))

    def test_real_volume(self):
        self.assertEqual(_positive_volume(6386000.0), 6386000)


class TestStaleTickerInFreshFeed(unittest.TestCase):
    """End to end through build_prices_from_history with a yfinance-shaped frame."""

    @classmethod
    def setUpClass(cls):
        cls.prices = _build(_history())

    def test_qrvo_flagged_stale_with_its_own_date(self):
        q = self.prices['QRVO']
        self.assertEqual(q['market_date'], QRVO_SESSION)
        self.assertEqual(q['market_time'], '2026-10-05T00:00:00')
        self.assertTrue(q['stale'])
        self.assertEqual(q['sessions_behind'], QRVO_SESSIONS_BEHIND)
        self.assertEqual(q['data_quality'][0], DQ_STALE_QUOTE)

    def test_qrvo_price_and_change_kept_from_its_session(self):
        q = self.prices['QRVO']
        self.assertAlmostEqual(q['price'], 114.17, places=2)
        self.assertAlmostEqual(q['change'], 5.73, places=2)

    def test_qrvo_volume_comes_from_its_own_session_not_padding(self):
        # Old code read vol.iloc[-1]: the padded 0 from Oct 21. Volume must be
        # the Oct 5 session's, matching the Oct 5 price it sits next to.
        q = self.prices['QRVO']
        self.assertEqual(q['volume'], 1_500_000)
        self.assertNotIn(DQ_VOLUME_MISSING, q['data_quality'])

    def test_zero_volume_on_close_session_is_unavailable_not_zero(self):
        raw = _history()
        raw.loc[pd.Timestamp(QRVO_SESSION), ('Volume', 'QRVO')] = 0.0
        q = _build(raw)['QRVO']
        self.assertIsNone(q['volume'])
        self.assertIn(DQ_VOLUME_MISSING, q['data_quality'])
        self.assertNotIn('QRVO', build_adv_section({'QRVO': q}))

    def test_nan_volume_on_close_session_is_unavailable(self):
        raw = _history()
        raw.loc[pd.Timestamp(FRESH_SESSION), ('Volume', 'TSEM')] = np.nan
        e = _build(raw)['TSEM']
        self.assertIsNone(e['volume'])
        self.assertFalse(e['stale'])   # missing volume alone is not a stale quote
        self.assertEqual(e['data_quality'], [DQ_VOLUME_MISSING])

    def test_qrvo_adv_ignores_padded_zero_sessions(self):
        # Only real prior sessions count: 29 sessions of 1.5M before Oct 5
        self.assertEqual(self.prices['QRVO']['adv'], 1_500_000)

    def test_qrvo_missing_smas_flagged(self):
        q = self.prices['QRVO']
        self.assertIsNone(q['sma50'])
        self.assertIsNone(q['sma200'])
        self.assertIn(DQ_SMA50_MISSING, q['data_quality'])
        self.assertIn(DQ_SMA200_MISSING, q['data_quality'])

    def test_fresh_tickers_preserved(self):
        for sym in ('VRT', 'PLTR', 'TSEM', 'SPY'):
            e = self.prices[sym]
            self.assertEqual(e['market_date'], FRESH_SESSION, sym)
            self.assertFalse(e['stale'], sym)
            self.assertEqual(e['sessions_behind'], 0, sym)
            self.assertEqual(e['data_quality'], [], sym)
            self.assertIsNotNone(e['sma50'], sym)
            self.assertIsNotNone(e['sma200'], sym)

    def test_fresh_ticker_values_unchanged_by_new_logic(self):
        # VRT: close = 100 + 0.1*i, volume flat 1,000,000
        e = self.prices['VRT']
        self.assertAlmostEqual(e['price'], 100 + 0.1 * 249, places=4)
        self.assertEqual(e['volume'], 1_000_000)
        self.assertEqual(e['adv'], 1_000_000)
        self.assertAlmostEqual(e['sma50'], round(100 + 0.1 * (249 + 200) / 2, 4), places=4)

    def test_macro_ahead_of_equities_not_stale(self):
        thb = self.prices['THB=X']
        self.assertEqual(thb['market_date'], '2026-10-22')
        self.assertFalse(thb['stale'])
        self.assertFalse(self.prices['^VIX']['stale'])

    def test_full_history_stale_ticker_keeps_smas_but_is_stale(self):
        prices = _build(_history(qrvo_valid_closes=230))
        q = prices['QRVO']
        self.assertTrue(q['stale'])
        self.assertIsNotNone(q['sma50'])
        self.assertIsNotNone(q['sma200'])
        self.assertEqual(q['data_quality'], [DQ_STALE_QUOTE])

    def test_adv_section_does_not_certify_stale_session(self):
        adv = build_adv_section(self.prices)
        self.assertTrue(adv['QRVO']['stale'])
        self.assertEqual(adv['QRVO']['market_date'], QRVO_SESSION)
        self.assertFalse(adv['QRVO']['gate_met'])
        self.assertFalse(adv['TSEM']['stale'])

    def test_output_metadata(self):
        out = build_output(self.prices, build_adv_section(self.prices), {}, {})
        self.assertEqual(out['reference_market_date'], FRESH_SESSION)
        self.assertEqual(out['stale_tickers'], ['QRVO'])
        # Feed run timestamp is still "now": the two coexist, which is the bug
        updated = datetime.strptime(out['last_updated'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
        self.assertLess((datetime.now(timezone.utc) - updated).total_seconds(), 60)


class TestAdvGateStaleSession(unittest.TestCase):
    def test_stale_high_ratio_is_not_gate_met(self):
        prices = {'TSEM': {'adv': 100000, 'volume': 300000, 'stale': True, 'market_date': QRVO_SESSION}}
        adv = build_adv_section(prices)
        self.assertEqual(adv['TSEM']['adv_ratio'], 3.0)
        self.assertFalse(adv['TSEM']['gate_met'])
        self.assertTrue(adv['TSEM']['stale'])

    def test_fresh_high_ratio_still_gate_met(self):
        prices = {'TSEM': {'adv': 100000, 'volume': 300000, 'stale': False}}
        self.assertTrue(build_adv_section(prices)['TSEM']['gate_met'])


class TestObservedFeedSnapshot(unittest.TestCase):
    """The prices.json written on 2026-10-22, re-annotated by the new code."""

    def setUp(self):
        self.feed = json.loads(FIXTURE.read_text())

    def test_only_qrvo_is_stale(self):
        meta = annotate_freshness(self.feed['prices'])
        self.assertEqual(meta['reference_market_date'], FRESH_SESSION)
        self.assertEqual(meta['stale_tickers'], ['QRVO'])
        q = self.feed['prices']['QRVO']
        self.assertEqual(q['sessions_behind'], QRVO_SESSIONS_BEHIND)
        self.assertIn(DQ_STALE_QUOTE, q['data_quality'])

    def test_fresh_symbols_untouched(self):
        before = {s: dict(e) for s, e in self.feed['prices'].items()}
        annotate_freshness(self.feed['prices'])
        for sym, e in self.feed['prices'].items():
            if sym == 'QRVO':
                continue
            for k, v in before[sym].items():
                self.assertEqual(e[k], v, f'{sym}.{k} changed')
            if e.get('market_time'):
                self.assertFalse(e['stale'], sym)

    def test_old_feed_qrvo_adv_gate_now_withheld(self):
        annotate_freshness(self.feed['prices'])
        adv = build_adv_section(self.feed['prices'])
        self.assertTrue(adv['QRVO']['stale'])
        self.assertFalse(adv['QRVO']['gate_met'])


def _extract_js_freshness():
    html = INDEX_HTML.read_text()
    begin = html.index('// ─── QUOTE FRESHNESS (per-ticker) — BEGIN ───')
    end = html.index('// ─── QUOTE FRESHNESS (per-ticker) — END ───')
    return html[begin:end]


@unittest.skipUnless(shutil.which('node'), 'node not installed')
class TestCommandCenterFreshnessJS(unittest.TestCase):
    """Runs the Command Center's freshness helpers under node against the snapshot."""

    def _run(self, feed):
        script = _extract_js_freshness() + """
const d = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const out = {};
for (const [s, e] of Object.entries(d.prices)) out[s] = quoteFreshness(e, d);
out.__ref = feedReferenceDate(d);
out.__stale = staleTickers(d, Object.keys(d.prices).filter(s => !s.startsWith('^') && !s.includes('=')));
process.stdout.write(JSON.stringify(out));
"""
        res = subprocess.run(['node', '-e', script], input=json.dumps(feed),
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(res.returncode, 0, res.stderr)
        return json.loads(res.stdout)

    def test_old_feed_without_flags_falls_back_to_market_time(self):
        out = self._run(json.loads(FIXTURE.read_text()))
        self.assertEqual(out['__ref'], FRESH_SESSION)
        self.assertEqual(out['__stale'], ['QRVO'])
        q = out['QRVO']
        self.assertTrue(q['stale'])
        self.assertEqual(q['sessionsBehind'], QRVO_SESSIONS_BEHIND)
        self.assertEqual(q['marketDate'], QRVO_SESSION)
        self.assertTrue(q['volumeMissing'])       # volume 0 is not real volume
        self.assertTrue(q['sma50Missing'])
        self.assertTrue(q['sma200Missing'])
        v = out['VRT']
        self.assertFalse(v['stale'])
        self.assertFalse(v['volumeMissing'])
        self.assertFalse(v['sma200Missing'])

    def test_new_feed_flags_are_used(self):
        feed = json.loads(FIXTURE.read_text())
        meta = annotate_freshness(feed['prices'])
        feed.update(meta)
        out = self._run(feed)
        self.assertEqual(out['__stale'], ['QRVO'])
        self.assertEqual(out['QRVO']['sessionsBehind'], QRVO_SESSIONS_BEHIND)


if __name__ == '__main__':
    unittest.main()
