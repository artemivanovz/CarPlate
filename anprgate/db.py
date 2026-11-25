"""Хранилище SQLite: справочник транспортных средств и журнал событий проезда."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS vehicles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    plate       TEXT    NOT NULL UNIQUE,
    color       TEXT    NOT NULL,
    color_lab   TEXT,
    owner       TEXT,
    note        TEXT,
    active      INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ts               TEXT    NOT NULL,
    source           TEXT    NOT NULL,
    media_pos        REAL,
    track_id         INTEGER,
    plate_raw        TEXT,
    plate            TEXT,
    plate_conf       REAL,
    plate_valid      INTEGER NOT NULL DEFAULT 0,
    votes            INTEGER NOT NULL DEFAULT 0,
    color_detected   TEXT,
    color_lab        TEXT,
    vehicle_id       INTEGER REFERENCES vehicles(id) ON DELETE SET NULL,
    match_distance   INTEGER,
    decision         TEXT    NOT NULL CHECK (decision IN ('allow', 'deny', 'review')),
    reasons          TEXT    NOT NULL,
    snapshot_path    TEXT,
    plate_crop_path  TEXT,
    operator_action  TEXT CHECK (operator_action IN ('confirm', 'allow', 'deny')),
    operator_ts      TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_ts    ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_plate ON events(plate);
"""


@dataclass
class Vehicle:
    id: int
    plate: str
    color: str
    color_lab: Optional[Tuple[float, float, float]]
    owner: Optional[str]
    note: Optional[str]
    active: bool


@dataclass
class EventRecord:
    ts: datetime
    source: str
    track_id: Optional[int]
    plate_raw: Optional[str]
    plate: Optional[str]
    plate_conf: Optional[float]
    plate_valid: bool
    votes: int
    color_detected: Optional[str]
    color_lab: Optional[Tuple[float, float, float]]
    vehicle_id: Optional[int]
    match_distance: Optional[int]
    decision: str
    reasons: Sequence[str]
    snapshot_path: Optional[str] = None
    plate_crop_path: Optional[str] = None
    media_pos: Optional[float] = None


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _lab_to_json(lab: Optional[Sequence[float]]) -> Optional[str]:
    return None if lab is None else json.dumps([round(float(v), 2) for v in lab])


def _lab_from_json(raw: Optional[str]) -> Optional[Tuple[float, float, float]]:
    if not raw:
        return None
    values = json.loads(raw)
    return (float(values[0]), float(values[1]), float(values[2]))


class Database:
    """Потокобезопасная обёртка над одним соединением SQLite."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self.init_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(SCHEMA)
            self._conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    # --- Справочник ТС -------------------------------------------------

    @staticmethod
    def _vehicle(row: sqlite3.Row) -> Vehicle:
        return Vehicle(
            id=row["id"],
            plate=row["plate"],
            color=row["color"],
            color_lab=_lab_from_json(row["color_lab"]),
            owner=row["owner"],
            note=row["note"],
            active=bool(row["active"]),
        )

    def add_vehicle(
        self,
        plate: str,
        color: str,
        owner: Optional[str] = None,
        note: Optional[str] = None,
        color_lab: Optional[Sequence[float]] = None,
        active: bool = True,
    ) -> int:
        now = _now()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO vehicles(plate, color, color_lab, owner, note, active, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (plate, color, _lab_to_json(color_lab), owner, note, int(active), now, now),
            )
            return int(cur.lastrowid)

    def update_vehicle(self, vehicle_id: int, **changes) -> None:
        allowed = {"plate", "color", "color_lab", "owner", "note", "active"}
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"Недопустимые поля: {sorted(unknown)}")
        if not changes:
            return
        if "color_lab" in changes:
            changes["color_lab"] = _lab_to_json(changes["color_lab"])
        if "active" in changes:
            changes["active"] = int(bool(changes["active"]))
        changes["updated_at"] = _now()
        assignments = ", ".join(f"{key} = ?" for key in changes)
        with self._lock, self._conn:
            self._conn.execute(
                f"UPDATE vehicles SET {assignments} WHERE id = ?",
                (*changes.values(), vehicle_id),
            )

    def remove_vehicle(self, vehicle_id: int) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM vehicles WHERE id = ?", (vehicle_id,))
            return cur.rowcount > 0

    def get_vehicle(self, vehicle_id: int) -> Optional[Vehicle]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM vehicles WHERE id = ?", (vehicle_id,)).fetchone()
        return self._vehicle(row) if row else None

    def find_by_plate(self, plate: str) -> Optional[Vehicle]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM vehicles WHERE plate = ?", (plate,)).fetchone()
        return self._vehicle(row) if row else None

    def list_vehicles(self) -> List[Vehicle]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM vehicles ORDER BY plate").fetchall()
        return [self._vehicle(r) for r in rows]

    def body_vector(self, vehicle_id: int) -> Optional[bytes]:
        """Эталонный вектор кузова, если колонка есть и вектор записан."""
        with self._lock:
            columns = {row[1] for row in self._conn.execute("PRAGMA table_info(vehicles)")}
            if "reid_vector" not in columns:
                return None
            row = self._conn.execute(
                "SELECT reid_vector FROM vehicles WHERE id = ?", (vehicle_id,)
            ).fetchone()
        if row is None or row[0] is None:
            return None
        return bytes(row[0])

    def set_body_vector(self, vehicle_id: int, vector: bytes) -> None:
        """Записывает эталон кузова. Старый вектор другой длины больше не сравнивается."""
        with self._lock, self._conn:
            columns = {row[1] for row in self._conn.execute("PRAGMA table_info(vehicles)")}
            if "reid_vector" not in columns:
                self._conn.execute("ALTER TABLE vehicles ADD COLUMN reid_vector BLOB")
            self._conn.execute(
                "UPDATE vehicles SET reid_vector = ? WHERE id = ?",
                (vector, vehicle_id),
            )

    # --- Журнал событий ------------------------------------------------

    def log_event(self, event: EventRecord) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO events(ts, source, media_pos, track_id, plate_raw, plate, plate_conf, plate_valid, "
                "votes, color_detected, color_lab, vehicle_id, match_distance, decision, reasons, snapshot_path, "
                "plate_crop_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event.ts.isoformat(timespec="milliseconds"),
                    event.source,
                    event.media_pos,
                    event.track_id,
                    event.plate_raw,
                    event.plate,
                    event.plate_conf,
                    int(event.plate_valid),
                    event.votes,
                    event.color_detected,
                    _lab_to_json(event.color_lab),
                    event.vehicle_id,
                    event.match_distance,
                    event.decision,
                    json.dumps(list(event.reasons), ensure_ascii=False),
                    event.snapshot_path,
                    event.plate_crop_path,
                ),
            )
            return int(cur.lastrowid)

    def set_operator_action(self, event_id: int, action: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE events SET operator_action = ?, operator_ts = ? WHERE id = ?",
                (action, _now(), event_id),
            )

    def recent_events(self, limit: int = 50) -> List[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()

    def import_vehicles(self, rows: Iterable[Tuple[str, str, Optional[str], Optional[str]]]) -> int:
        count = 0
        for plate, color, owner, note in rows:
            existing = self.find_by_plate(plate)
            if existing:
                self.update_vehicle(existing.id, color=color, owner=owner, note=note)
            else:
                self.add_vehicle(plate, color, owner=owner, note=note)
            count += 1
        return count
