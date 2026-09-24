-- AI Toll Booth Vision Analytics — SQLite schema
-- Kept intentionally simple: five tables, no ORM migrations.

CREATE TABLE IF NOT EXISTS videos (
    video_id        TEXT PRIMARY KEY,
    filename        TEXT NOT NULL,
    filepath        TEXT NOT NULL,
    uploaded_at      TEXT NOT NULL,
    duration_sec    REAL,
    fps             REAL,
    width           INTEGER,
    height          INTEGER,
    status          TEXT NOT NULL DEFAULT 'uploaded',  -- uploaded | processing | done | failed
    annotated_path  TEXT,
    error_message   TEXT
);

CREATE TABLE IF NOT EXISTS vehicles (
    vehicle_id      TEXT PRIMARY KEY,   -- "{video_id}-{track_id}"
    video_id        TEXT NOT NULL REFERENCES videos(video_id),
    track_id        INTEGER NOT NULL,
    vehicle_type    TEXT NOT NULL,      -- car | truck | bus | motorcycle | other
    first_seen_sec  REAL NOT NULL,
    last_seen_sec   REAL NOT NULL,
    first_seen_frame INTEGER NOT NULL,
    last_seen_frame  INTEGER NOT NULL,
    avg_confidence  REAL NOT NULL,
    direction       TEXT,               -- entering | exiting | unknown
    is_active       INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS events (
    event_id        TEXT PRIMARY KEY,
    video_id        TEXT NOT NULL REFERENCES videos(video_id),
    vehicle_id      TEXT REFERENCES vehicles(vehicle_id),
    event_type      TEXT NOT NULL,      -- VEHICLE_ENTERED | VEHICLE_EXITED | VEHICLE_TYPE_CHANGED
                                         -- PLATE_DETECTED | OCR_COMPLETED | LOW_CONFIDENCE |
                                         -- UNUSUAL_VISUAL_CONDITION
    timestamp_sec   REAL NOT NULL,
    frame_number    INTEGER,
    confidence      REAL,
    metadata_json   TEXT                -- free-form JSON blob
);

CREATE TABLE IF NOT EXISTS ocr_results (
    ocr_id          TEXT PRIMARY KEY,
    video_id        TEXT NOT NULL REFERENCES videos(video_id),
    vehicle_id      TEXT REFERENCES vehicles(vehicle_id),
    frame_number    INTEGER,
    plate_text      TEXT,
    confidence      REAL,
    is_experimental INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS vlm_analysis (
    analysis_id     TEXT PRIMARY KEY,
    video_id        TEXT NOT NULL REFERENCES videos(video_id),
    frame_number    INTEGER,
    timestamp_sec   REAL,
    vehicle_count   INTEGER,
    vehicle_types_json TEXT,
    lane_occupied   INTEGER,
    visual_conditions_json TEXT,
    is_unclear      INTEGER,
    confidence      REAL,
    raw_response    TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_video ON events(video_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_vehicles_video ON vehicles(video_id);
