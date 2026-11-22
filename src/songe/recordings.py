"""Upkeep of the recorded conversations in experience.sqlite3 (issue #73).

- dedupe: rows recorded before schema v3 keep each request's whole history.
  They are rewritten in place, a few at a time, to share repeated parts
  (parts.py). Each batch is one transaction that also moves a saved cursor,
  so a stopped run loses nothing and the next one carries on where it was.
  Every rewritten row is read back and compared before it replaces the old
  one. Rows not yet rewritten stay plain JSON, which every reader reads.
- compact: give the freed space back to the disk, in the background.
"""

import json
import logging
import shutil

from . import parts

logger = logging.getLogger(__name__)

# Rows rewritten per transaction. Old rows can hold a whole agent history
# (megabytes each); a small batch keeps the write lock short for the proxy's
# recording thread and memory low in the guarded child.
DEDUPE_BATCH = 16
# Free space below this is left for SQLite to reuse for new recordings:
# rewriting the file to give it back would cost more than it returns.
COMPACT_MIN_BYTES = 64 * 1024 * 1024
# Pages given back per transaction (4 KiB pages: 16 MiB), so a recording
# waiting for the write lock never waits long.
COMPACT_STEP_PAGES = 4096

_TABLES = (
    ("exchanges", "id", "request", parts.REQUEST_LISTS),
    ("steps", "rowid", "payload", parts.STEP_LISTS),
)


def cursor_key(table):
    return f"parts_cursor_{table}"


def dedupe(store, stop=None):
    """Rewrite older rows to share repeated parts. Returns how many rows
    were rewritten. stop(): checked between batches."""
    rewritten = 0
    for table, key, column, lists in _TABLES:
        while stop is None or not stop():
            with store.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                saved = db.execute(
                    "SELECT value FROM state WHERE key=?", (cursor_key(table),)
                ).fetchone()
                cursor = json.loads(saved[0]) if saved else 0
                rows = db.execute(
                    f"SELECT {key} AS k, {column} AS body FROM {table}"
                    f" WHERE {key}>? AND refs IS NULL ORDER BY {key} LIMIT ?",
                    (cursor, DEDUPE_BATCH),
                ).fetchall()
                for row in rows:
                    if row["body"] is None:
                        continue
                    try:
                        value = json.loads(row["body"])
                    except ValueError:
                        # Unreadable before as after: left as it is, and the
                        # cursor moves past it so the rest goes on.
                        logger.warning("dedupe_unreadable %s %s", table, row["k"])
                        continue
                    skeleton, refs = parts.pack(db, value, lists)
                    if refs is None:
                        continue  # nothing large or repeated in it
                    back = parts.unpack(db, skeleton, refs)
                    if json.dumps(back) != json.dumps(value):
                        # Never replace a recording by something that reads
                        # back differently: it stays as it is (its unused
                        # parts are collected later).
                        logger.warning("dedupe_mismatch %s %s", table, row["k"])
                        parts.collect(db, parts.ids_of(refs))
                        continue
                    db.execute(
                        f"UPDATE {table} SET {column}=?, refs=? WHERE {key}=?",
                        (skeleton, refs, row["k"]),
                    )
                    if table == "steps":
                        # The evidence column copied the payload's evidence,
                        # which the payload keeps (log.Store._put_step).
                        db.execute(
                            "UPDATE steps SET evidence='[]' WHERE rowid=?", (row["k"],)
                        )
                    rewritten += 1
                if len(rows) < DEDUPE_BATCH:
                    # Every row up to here is done; newer ones were recorded
                    # sharing their parts already.
                    end = db.execute(f"SELECT COALESCE(MAX({key}),0) FROM {table}").fetchone()[0]
                else:
                    end = rows[-1]["k"]
                db.execute(
                    "INSERT OR REPLACE INTO state VALUES (?,?)",
                    (cursor_key(table), json.dumps(max(cursor, end))),
                )
            if len(rows) < DEDUPE_BATCH:
                break
    if rewritten:
        logger.info("recordings_deduped rows=%s", rewritten)
    return rewritten


def sizes(store):
    """(bytes in use, free bytes inside the file, auto_vacuum mode)."""
    with store.connect() as db:
        page = db.execute("PRAGMA page_size").fetchone()[0]
        count = db.execute("PRAGMA page_count").fetchone()[0]
        free = db.execute("PRAGMA freelist_count").fetchone()[0]
        mode = db.execute("PRAGMA auto_vacuum").fetchone()[0]
    return (count - free) * page, free * page, mode


INCREMENTAL = 2  # PRAGMA auto_vacuum value


def tidy(store):
    """The service's upkeep in an idle gap: give freed space back. Skipped
    while the guarded experience child (reconstruction and dedupe) runs: it
    holds experience.lock, and the next idle gap tries again."""
    from .platform.locking import acquire

    with (store.root / "experience.lock").open("a+") as lease:
        try:
            acquire(lease)
        except BlockingIOError:
            return None
        return compact(store)


def compact(store, minimum=COMPACT_MIN_BYTES):
    """Give free space inside the file back to the disk. A store made
    before Songe could do this in steps is rewritten once (VACUUM, while
    nothing else runs: the caller waits for an idle moment); after that
    space goes back in short steps. Returns the bytes given back."""
    with store.connect() as db:
        # Parts nothing uses any more (normally collected where they were
        # freed); safe in one write transaction: a part and the row using it
        # are always written together.
        db.execute("BEGIN IMMEDIATE")
        parts.collect(db)
    used, free, mode = sizes(store)
    if free < minimum:
        return 0
    before = store.path.stat().st_size
    if mode != INCREMENTAL:
        # VACUUM copies the data in use to a temporary file and back
        # through the write-ahead log: room for two copies.
        if shutil.disk_usage(store.root).free < 2 * used:
            logger.warning("compact_skipped_disk used=%s", used)
            return 0
        with store.connect() as db:
            db.execute("PRAGMA auto_vacuum=INCREMENTAL")
            db.execute("VACUUM")
    else:
        while True:
            with store.connect() as db:
                db.execute(f"PRAGMA incremental_vacuum({COMPACT_STEP_PAGES})").fetchall()
                left = db.execute("PRAGMA freelist_count").fetchone()[0]
            if not left:
                break
    with store.connect() as db:
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
    after = store.path.stat().st_size
    logger.info("recordings_compacted before=%s after=%s", before, after)
    return max(0, before - after)
