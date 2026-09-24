"""
Thin, dependency-free SQLite wrapper.

No ORM on purpose — the schema is small (5 tables) and the POC favors
readability over abstraction. All writes go through explicit helper
methods so the rest of the codebase never writes raw SQL.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Optional


class Database:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    # -- connection handling -------------------------------------------------
    @property
    def conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA foreign_keys = ON")
        return self._local.conn

    def _init_schema(self) -> None:
        schema_path = Path(__file__).parent / "schema.sql"
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(schema_path.read_text())

    # -- generic helpers -------------------------------------------------------
    def execute(self, sql: str, params: Any = ()) -> sqlite3.Cursor:
        # `params` may be a dict (for ":name" style queries) or a
        # sequence (for "?" style queries) — sqlite3 needs the original
        # type preserved, so we don't coerce dicts into tuples here.
        cur = self.conn.execute(sql, params)
        self.conn.commit()
        return cur

    def query(self, sql: str, params: Any = ()) -> list[dict]:
        cur = self.conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    def query_one(self, sql: str, params: Iterable[Any] = ()) -> Optional[dict]:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    # -- videos ---------------------------------------------------------------
    def insert_video(self, video: dict) -> None:
        self.execute(
            """INSERT INTO videos (video_id, filename, filepath, uploaded_at, duration_sec,
               fps, width, height, status)
               VALUES (:video_id, :filename, :filepath, :uploaded_at, :duration_sec,
               :fps, :width, :height, :status)""",
            video,
        )

    def update_video_status(self, video_id: str, status: str, error_message: str = None,
                             annotated_path: str = None) -> None:
        self.execute(
            """UPDATE videos SET status = ?, error_message = COALESCE(?, error_message),
               annotated_path = COALESCE(?, annotated_path) WHERE video_id = ?""",
            (status, error_message, annotated_path, video_id),
        )

    def get_video(self, video_id: str) -> Optional[dict]:
        return self.query_one("SELECT * FROM videos WHERE video_id = ?", (video_id,))

    def list_videos(self) -> list[dict]:
        return self.query("SELECT * FROM videos ORDER BY uploaded_at DESC")

    # -- vehicles ---------------------------------------------------------------
    def upsert_vehicle(self, vehicle: dict) -> None:
        existing = self.query_one(
            "SELECT vehicle_id FROM vehicles WHERE vehicle_id = ?", (vehicle["vehicle_id"],)
        )
        if existing:
            self.execute(
                """UPDATE vehicles SET last_seen_sec = :last_seen_sec,
                   last_seen_frame = :last_seen_frame, avg_confidence = :avg_confidence,
                   direction = :direction, vehicle_type = :vehicle_type,
                   is_active = :is_active WHERE vehicle_id = :vehicle_id""",
                vehicle,
            )
        else:
            self.execute(
                """INSERT INTO vehicles (vehicle_id, video_id, track_id, vehicle_type,
                   first_seen_sec, last_seen_sec, first_seen_frame, last_seen_frame,
                   avg_confidence, direction, is_active)
                   VALUES (:vehicle_id, :video_id, :track_id, :vehicle_type,
                   :first_seen_sec, :last_seen_sec, :first_seen_frame, :last_seen_frame,
                   :avg_confidence, :direction, :is_active)""",
                vehicle,
            )

    def list_vehicles(self, video_id: str) -> list[dict]:
        return self.query(
            "SELECT * FROM vehicles WHERE video_id = ? ORDER BY first_seen_sec", (video_id,)
        )

    # -- events ---------------------------------------------------------------
    def insert_event(self, event: dict) -> None:
        row = dict(event)
        row["metadata_json"] = json.dumps(row.pop("metadata", {}))
        self.execute(
            """INSERT INTO events (event_id, video_id, vehicle_id, event_type, timestamp_sec,
               frame_number, confidence, metadata_json)
               VALUES (:event_id, :video_id, :vehicle_id, :event_type, :timestamp_sec,
               :frame_number, :confidence, :metadata_json)""",
            row,
        )

    def list_events(self, video_id: str, event_type: str = None, min_confidence: float = None,
                     start_sec: float = None, end_sec: float = None) -> list[dict]:
        sql = "SELECT * FROM events WHERE video_id = ?"
        params: list[Any] = [video_id]
        if event_type:
            sql += " AND event_type = ?"
            params.append(event_type)
        if min_confidence is not None:
            sql += " AND (confidence IS NULL OR confidence >= ?)"
            params.append(min_confidence)
        if start_sec is not None:
            sql += " AND timestamp_sec >= ?"
            params.append(start_sec)
        if end_sec is not None:
            sql += " AND timestamp_sec <= ?"
            params.append(end_sec)
        sql += " ORDER BY timestamp_sec"
        rows = self.query(sql, params)
        for r in rows:
            r["metadata"] = json.loads(r.pop("metadata_json") or "{}")
        return rows

    # -- ocr ---------------------------------------------------------------
    def insert_ocr_result(self, result: dict) -> None:
        self.execute(
            """INSERT INTO ocr_results (ocr_id, video_id, vehicle_id, frame_number,
               plate_text, confidence, is_experimental)
               VALUES (:ocr_id, :video_id, :vehicle_id, :frame_number, :plate_text,
               :confidence, :is_experimental)""",
            result,
        )

    def list_ocr_results(self, video_id: str) -> list[dict]:
        return self.query("SELECT * FROM ocr_results WHERE video_id = ?", (video_id,))

    # -- vlm ---------------------------------------------------------------
    def insert_vlm_analysis(self, analysis: dict) -> None:
        row = dict(analysis)
        row["vehicle_types_json"] = json.dumps(row.pop("vehicle_types", []))
        row["visual_conditions_json"] = json.dumps(row.pop("visual_conditions", []))
        self.execute(
            """INSERT INTO vlm_analysis (analysis_id, video_id, frame_number, timestamp_sec,
               vehicle_count, vehicle_types_json, lane_occupied, visual_conditions_json,
               is_unclear, confidence, raw_response)
               VALUES (:analysis_id, :video_id, :frame_number, :timestamp_sec, :vehicle_count,
               :vehicle_types_json, :lane_occupied, :visual_conditions_json, :is_unclear,
               :confidence, :raw_response)""",
            row,
        )

    def list_vlm_analysis(self, video_id: str) -> list[dict]:
        rows = self.query("SELECT * FROM vlm_analysis WHERE video_id = ?", (video_id,))
        for r in rows:
            r["vehicle_types"] = json.loads(r.pop("vehicle_types_json") or "[]")
            r["visual_conditions"] = json.loads(r.pop("visual_conditions_json") or "[]")
        return rows
