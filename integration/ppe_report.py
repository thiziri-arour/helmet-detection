# helmet report for the vision dashboard, goes in backend/ppe_report.py

import sqlite3
from collections import Counter, defaultdict


def collect_tracks(results, names):
    # one result per frame -> {track id: Counter of labels}
    tracks = defaultdict(Counter)
    for result in results:
        if result.boxes.id is None:
            continue
        for cls, track_id in zip(result.boxes.cls.tolist(), result.boxes.id.tolist()):
            tracks[int(track_id)][names[int(cls)]] += 1
    return tracks


def summarize_tracks(tracks, min_frames=5):
    # same as in predict.py: most frequent label per track
    counts = Counter({"helmet": 0, "no_helmet": 0, "vest": 0, "person": 0})
    for history in tracks.values():
        if sum(history.values()) >= min_frames:
            counts[history.most_common(1)[0][0]] += 1
    return dict(counts)


def create_table(conn: sqlite3.Connection):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ppe_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            helmet INTEGER,
            no_helmet INTEGER,
            vest INTEGER,
            person INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()


def save_report(conn: sqlite3.Connection, filename: str, counts: dict):
    conn.execute(
        "INSERT INTO ppe_reports (filename, helmet, no_helmet, vest, person) VALUES (?, ?, ?, ?, ?)",
        (filename, counts["helmet"], counts["no_helmet"], counts["vest"], counts["person"]),
    )
    conn.commit()


def list_reports(conn: sqlite3.Connection, limit: int = 20):
    rows = conn.execute(
        "SELECT filename, helmet, no_helmet, vest, person, created_at "
        "FROM ppe_reports ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    keys = ["filename", "helmet", "no_helmet", "vest", "person", "created_at"]
    return [dict(zip(keys, row)) for row in rows]
