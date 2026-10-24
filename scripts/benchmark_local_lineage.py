#!/usr/bin/env python3
"""Measure #143833's projection on synthetic data, never the user's state.db.

Run: python scripts/benchmark_local_lineage.py --prompt-bytes 8192
The unchanged index builder is the uncached control; the cached path is the one
used by session list/count calls. Timings are component timings, not UI latency.
"""
from __future__ import annotations

import argparse
import json
import platform
import sqlite3
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hermes_state import SessionDB
from hermes_state_local import POLICY_PREFIX, _cached_local_lineage_index, local_lineage_index


def measure(fn, repeat, expected):
    samples = []
    for _ in range(repeat):
        start = time.perf_counter_ns()
        result = fn()
        samples.append((time.perf_counter_ns() - start) / 1_000_000)
        assert result == expected
    return {'median_ms': statistics.median(samples), 'min_ms': min(samples), 'max_ms': max(samples)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipts', type=int, default=2098)
    parser.add_argument('--sessions', type=int, default=4583)
    parser.add_argument('--prompt-bytes', type=int, default=8192)
    parser.add_argument('--reset-lineages', type=int, default=0)
    parser.add_argument('--repeat', type=int, default=9)
    args = parser.parse_args()
    if not (0 <= args.reset_lineages <= args.receipts and args.sessions >= args.receipts + args.reset_lineages
            and args.receipts > 0 and args.prompt_bytes >= 0 and args.repeat > 0):
        parser.error('positive counts, nonnegative payload, and enough sessions for owners plus reset children required')
    with tempfile.TemporaryDirectory(prefix='hermes-lineage-bench-') as folder:
        db = SessionDB(db_path=Path(folder) / 'state.db')
        try:
            conn = db._conn
            assert conn is not None
            conn.executemany('INSERT INTO sessions(id,source,started_at,end_reason) VALUES(?,?,0,?)', (
                (f's-{i}', 'cli', 'session_reset' if i < args.reset_lineages else None)
                for i in range(args.sessions)))
            conn.executemany('INSERT INTO state_meta(key,value) VALUES(?,?)', (
                (POLICY_PREFIX + f's-{i}', json.dumps({
                    'session_id': f's-{i}',
                    'lineage': [f's-{i}', f's-{args.receipts+i}'] if i < args.reset_lineages else [f's-{i}'],
                    'policy': {'request_json': json.dumps({'prompt': 'x' * args.prompt_bytes})},
                })) for i in range(args.receipts)))
            conn.commit()
            with db._read_ctx() as reader:
                expected = local_lineage_index(reader)
                assert len(expected[0]) == args.reset_lineages
                uncached = measure(lambda: local_lineage_index(reader), args.repeat, expected)
                cold = measure(lambda: _cached_local_lineage_index(db, reader), 1, expected)
                warm = measure(lambda: _cached_local_lineage_index(db, reader), args.repeat, expected)
                scans = []
                reader.set_trace_callback(lambda sql: scans.append(sql) if 'FROM state_meta' in sql and 'GLOB' in sql else None)
                try:
                    for _ in range(args.repeat):
                        assert _cached_local_lineage_index(db, reader) == expected
                    warm_scans = len(scans)
                    # Any committed DB write conservatively invalidates, not just receipt writes.
                    conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', ('benchmark-write', '1'))
                    conn.commit()
                    invalidated = measure(lambda: _cached_local_lineage_index(db, reader), 1, expected)
                    invalidated_scans = len(scans) - warm_scans
                finally:
                    reader.set_trace_callback(None)
                assert warm_scans == 0 and invalidated_scans == 1
            print(json.dumps({
                'workload': vars(args), 'python': platform.python_version(), 'sqlite': sqlite3.sqlite_version,
                'platform': platform.system(), 'synthetic': True, 'uncached': uncached,
                'cache_first_read': cold, 'cache_warm': warm, 'after_committed_write': invalidated,
                'warm_receipt_scans': warm_scans, 'after_write_receipt_scans': invalidated_scans,
            }, indent=2))
        finally:
            db.close()


if __name__ == '__main__':
    main()
