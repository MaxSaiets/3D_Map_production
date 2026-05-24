"""Benchmark DuckDB insertion strategies."""
import sys
import time
import duckdb

# Synthetic data
N = 100_000
rows = [(i, "primary", "no", "LINESTRING(30.5 50.4, 30.6 50.5)", 30.5, 50.4, 30.6, 50.5) for i in range(N)]
print(f"Generated {N} synthetic rows", flush=True)


def setup():
    if "test.duckdb" in __import__("os").listdir("."):
        __import__("os").remove("test.duckdb")
    c = duckdb.connect("test.duckdb")
    c.execute("CREATE TABLE roads (id BIGINT, highway VARCHAR, bridge VARCHAR, wkt VARCHAR, minlon DOUBLE, minlat DOUBLE, maxlon DOUBLE, maxlat DOUBLE)")
    return c


def bench_executemany(rows):
    c = setup()
    t = time.time()
    c.executemany("INSERT INTO roads VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    el = time.time() - t
    c.close()
    return el


def bench_appender(rows):
    c = setup()
    t = time.time()
    app = c.appender("roads")
    for row in rows:
        app.append_row(row)
    app.close()
    el = time.time() - t
    c.close()
    return el


def bench_batched(rows, batch_size=10000):
    c = setup()
    t = time.time()
    for i in range(0, len(rows), batch_size):
        c.executemany("INSERT INTO roads VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows[i:i+batch_size])
    el = time.time() - t
    c.close()
    return el


def bench_pandas(rows):
    """Через pandas DataFrame — DuckDB read_pandas."""
    import pandas as pd
    c = setup()
    t = time.time()
    df = pd.DataFrame(rows, columns=["id","highway","bridge","wkt","minlon","minlat","maxlon","maxlat"])
    c.execute("INSERT INTO roads SELECT * FROM df")
    el = time.time() - t
    c.close()
    return el


print("\nbench_executemany (single call)...", flush=True)
t = bench_executemany(rows); print(f"  {t:.2f}s = {N/t:.0f} rows/sec", flush=True)

print("\nbench_batched (10K chunks)...", flush=True)
t = bench_batched(rows); print(f"  {t:.2f}s = {N/t:.0f} rows/sec", flush=True)

print("\nbench_appender...", flush=True)
t = bench_appender(rows); print(f"  {t:.2f}s = {N/t:.0f} rows/sec", flush=True)

print("\nbench_pandas (bulk INSERT FROM df)...", flush=True)
t = bench_pandas(rows); print(f"  {t:.2f}s = {N/t:.0f} rows/sec", flush=True)

import os
os.remove("test.duckdb")
