"""Вливає колії (parquet) у таблицю `roads` бойової ukraine.duckdb.

Після цього і генерація, і превʼю беруть залізницю з бази, а Overpass більше
не потрібен — саме він був причиною «то є, то нема» (504/406/тротлінг).

Ідемпотентний: спершу видаляє наявні рядки highway='railway', потім вставляє.
Перед записом РОБИТЬ БЕКАП (duckdb тримає весь файл, і зіпсувати його —
означає покласти всі мапи).

ВАЖЛИВО: бекенд тримає read-only конект до цієї БД. Перед запуском зупини
його (`pm2 stop 3dmap-backend`), інакше duckdb не дасть ексклюзивний лок.

Usage:
    python merge_rails_into_osm_db.py <rails.parquet> <ukraine.duckdb> [--no-backup]
"""

import shutil
import sys
import time
from pathlib import Path

import duckdb


def main():
    if len(sys.argv) < 3:
        print("Usage: python merge_rails_into_osm_db.py <rails.parquet> <ukraine.duckdb> [--no-backup]")
        sys.exit(1)
    parquet = Path(sys.argv[1])
    db_path = Path(sys.argv[2])
    do_backup = "--no-backup" not in sys.argv

    if not parquet.exists():
        print(f"Parquet not found: {parquet}")
        sys.exit(1)
    if not db_path.exists():
        print(f"DB not found: {db_path}")
        sys.exit(1)

    if do_backup:
        backup = db_path.with_suffix(db_path.suffix + f".bak_{int(time.time())}")
        print(f"Backup -> {backup} ({db_path.stat().st_size/1e9:.2f} GB)...", flush=True)
        shutil.copy2(db_path, backup)
        print("Backup done.", flush=True)

    conn = duckdb.connect(str(db_path))
    conn.execute("SET memory_limit='1500MB'")

    before = conn.execute("SELECT COUNT(*) FROM roads").fetchone()[0]
    existing_rails = conn.execute(
        "SELECT COUNT(*) FROM roads WHERE highway = 'railway'"
    ).fetchone()[0]
    print(f"roads before: {before:,} (railway: {existing_rails:,})", flush=True)

    if existing_rails:
        conn.execute("DELETE FROM roads WHERE highway = 'railway'")
        print(f"deleted {existing_rails:,} old railway rows", flush=True)

    conn.execute(f"""
        INSERT INTO roads
        SELECT id, highway, bridge, wkt, minlon, minlat, maxlon, maxlat
        FROM read_parquet('{parquet.as_posix()}')
    """)

    after = conn.execute("SELECT COUNT(*) FROM roads").fetchone()[0]
    rails = conn.execute(
        "SELECT COUNT(*) FROM roads WHERE highway = 'railway'"
    ).fetchone()[0]
    print(f"roads after: {after:,} (railway: {rails:,})", flush=True)

    # Санітарна перевірка на реальній ділянці — вокзал Ужгорода
    probe = conn.execute("""
        SELECT COUNT(*) FROM roads
        WHERE highway = 'railway'
          AND minlon <= 22.30251 AND maxlon >= 22.29989
          AND minlat <= 48.61034 AND maxlat >= 48.60839
    """).fetchone()[0]
    print(f"probe (вокзал Ужгорода): {probe} колій", flush=True)
    conn.close()

    if rails == 0 or probe == 0:
        print("FAIL: колії не залились або не знаходяться по bbox")
        sys.exit(2)
    print("OK")


if __name__ == "__main__":
    main()
