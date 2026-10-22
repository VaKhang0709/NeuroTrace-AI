%%writefile /content/drive/MyDrive/NEUROTRACE/audit_real_datasets.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_real_datasets.py — READ-ONLY audit of the REAL NeuroTrace datasets
Kiểm toán CHỈ ĐỌC các tập dữ liệu THẬT của NeuroTrace (không huấn luyện gì cả)

===========================================================================
 WHY THIS FILE EXISTS / VÌ SAO CÓ FILE NÀY
===========================================================================
Before a single gradient is computed we must know two things with certainty:

  1. HOW MANY *UNIQUE HUMAN BEINGS* are actually in the data.
     (N drawings != N participants. If 40 JSON files are 40 drawings made by
      1 person, we have 1 subject, not 40. Training on that and reporting
      "95% accuracy" is self-deception — the model just memorised one hand.)
  2. HOW MANY recordings exist per task (Static Spiral / Dynamic Spiral /
     Circular-Stability), so we know what is even trainable.

Trước khi huấn luyện, phải chắc chắn 2 điều: (1) có bao nhiêu NGƯỜI THẬT sự
khác biệt trong dữ liệu (N bản vẽ ≠ N người tham gia); (2) có bao nhiêu bản
ghi theo từng bài test để biết dữ liệu nào dùng được.

===========================================================================
 WHAT THIS SCRIPT DOES / SCRIPT NÀY LÀM GÌ
===========================================================================
  [1] UCI ROOT   — recursive scan of every *.txt (semicolon-separated).
  [2] UCI        — unique SUBJECT count, computed 6 different ways, because
                   "1 file = 1 subject" is an ASSUMPTION that must be tested,
                   not trusted (UCI-395 stores several tests per file in the
                   `hw_dataset/` split and one test per file in `new_dataset/`).
  [3] UCI        — Test-ID census: Test ID 0 = Static Spiral Test (SST),
                   Test ID 1 = Dynamic Spiral Test (DST),
                   Test ID 2 = Stability Test on Certain Point (STCP,
                   a.k.a. the "Circular" task). Reported as FILES, TRIALS
                   (contiguous runs) and SAMPLES (rows).
  [4] JSON ROOT  — recursive scan of every *.json.
  [5] JSON       — participant-identity forensics: looks for real participant
                   ids, separates them from per-DRAWING ids (id / sessionId /
                   drawingId …), falls back to filename heuristics, and states
                   a VERDICT: distinct participants vs. one person drawing many
                   times (with the leakage warning that follows).
  [6] TOTAL      — unique participants across ALL classes, with namespaced ids
                   so that UCI subject "1" and JSON participant "1" are never
                   silently merged into the same human being.
  [7][8]         — data-quality warnings + concrete recommendations.

===========================================================================
 SAFETY CONTRACT / CAM KẾT AN TOÀN
===========================================================================
  * NO model is trained, no checkpoint touched, NO data file is written,
    renamed, moved or deleted. The only writes are optional audit reports
    into a brand-new `audit_reports/` folder (disable with --no-export).
  * Every file operation is individually wrapped: a corrupt / empty /
    permission-denied / wrongly-encoded / 2 GB file can never abort the run.
    Failures are counted and listed at the end.
  * pandas is OPTIONAL — the script falls back to the stdlib `csv` module so
    that the audit still runs on a bare Colab kernel.

===========================================================================
 COLAB USAGE / CÁCH DÙNG TRONG GOOGLE COLAB
===========================================================================
  Cell 1 (this file):      writes the script into Google Drive.
  Cell 2:                  %run /content/drive/MyDrive/NEUROTRACE/audit_real_datasets.py
                           # or:  !python3 /content/drive/MyDrive/NEUROTRACE/audit_real_datasets.py
  Override paths if needed:
                           !python3 .../audit_real_datasets.py --uci-root /path --json-root /path

  NOTE ON LINE 1 / CHÚ THÍCH DÒNG 1
  The leading `%%writefile` is a Colab/IPython magic — it is what *creates*
  the .py file on Drive. It is NOT valid Python outside a notebook; strip it
  if you want to run this file as a plain script elsewhere.
  Dòng đầu `%%writefile` là magic của Colab để TẠO file .py trên Drive, không
  phải Python thuần; xoá dòng này nếu chạy file ngoài notebook.

Author: NeuroTrace AI — data engineering / Kỹ sư dữ liệu
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import sys
import textwrap
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# pandas is a SPEED optimisation, never a requirement.
# pandas chỉ để TĂNG TỐC, không bắt buộc.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - environment dependent
    import pandas as pd

    HAS_PANDAS = True
except Exception:  # pragma: no cover
    pd = None
    HAS_PANDAS = False

IN_COLAB = ("google.colab" in sys.modules) or os.path.exists("/content")

# ===========================================================================
# Configuration / Cấu hình
# ===========================================================================
PROJECT_ROOT = Path("/content/drive/MyDrive/NEUROTRACE")
# Relative to the project root, so that `--project-root` relocates everything.
UCI_REL = Path("external_test_data") / "uci_395_raw"
JSON_REL = Path("healthy")
REPORT_REL = Path("audit_reports")
UCI_ROOT = PROJECT_ROOT / UCI_REL
JSON_ROOT = PROJECT_ROOT / JSON_REL
REPORT_DIR = PROJECT_ROOT / REPORT_REL

# UCI dataset #395 (Isenkul, Sakar & Kursun, 2013/2017):
#   Parkinson Disease Spiral Drawings Using Digitized Graphics Tablet.
#   62 PWP (people with Parkinson) + 15 healthy controls = 77 subjects,
#   3 tests each: SST / DST / STCP.
UCI_EXPECTED: Dict[str, Optional[int]] = {"total": 77, "parkinson": 62, "control": 15}

TEST_ID_LABELS: Dict[int, str] = {
    0: "Static Spiral Test (SST)",
    1: "Dynamic Spiral Test (DST)",
    2: "Stability Test on Certain Point (STCP) — 'Circular'",
}

# Textual Test-ID spellings seen in the wild / các cách ghi Test-ID dạng chữ.
TEST_ID_TEXT_HINTS: List[Tuple[str, int]] = [
    ("static", 0), ("sst", 0), ("spiral_static", 0),
    ("dynamic", 1), ("dst", 1), ("blink", 1),
    ("stability", 2), ("stcp", 2), ("circular", 2), ("circle", 2), ("point", 2),
]

# Directory / filename tokens that identify the clinical class.
# Từ khoá thư mục / tên file để xác định lớp lâm sàng.
CONTROL_TOKENS = {
    "control", "controls", "controlled", "healthy", "health", "hc", "normal",
    "normals", "ctl", "con", "saglikli", "kontrol",
}
PD_TOKENS = {
    "parkinson", "parkinsons", "pd", "pwp", "patient", "patients", "pdp",
    "disease", "hasta", "parkinsonlu",
}
# Ambiguous tokens are deliberately NOT used. / Không dùng từ khoá mơ hồ.

ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
CANDIDATE_DELIMITERS = (";", ",", "\t", "|")

# Identity key tiers for the JSON audit / Phân hạng khoá định danh trong JSON.
PARTICIPANT_STRONG = {
    "participantid", "participant", "participants", "participantcode",
    "participantno", "participantnumber", "participantname", "participantkey",
    "subjectid", "subject", "subjectcode", "subjectno", "subjectname", "subjid",
    "patientid", "patient", "patientcode", "patientno",
    "userid", "user", "username", "usermail", "uid", "uuid",
    "personid", "person", "volunteerid", "volunteer", "testerid", "tester",
    "email", "mail", "phone", "respondentid", "respondent",
}
# Proxy: usually one device == one person, but NOT guaranteed.
PARTICIPANT_PROXY = {
    "deviceid", "deviceuuid", "device", "installationid", "clientid",
    "sessionid", "session", "browserid", "anonymousid", "anonid",
}
# These identify the DRAWING, not the human — never used as participant id.
DRAWING_LEVEL_IDS = {
    "id", "drawingid", "drawing_id", "recordid", "sampleid", "captureid",
    "fileid", "filename", "trialid", "taskid", "testid", "index", "rowid",
    "guid", "uidrawingid", "measurementid", "eventid",
}
# Used only as a last resort and always flagged as UNRELIABLE.
# 'label' is deliberately excluded — it is a CLASS label, not a person.
PARTICIPANT_WEAK = {"name", "code", "tag", "owner", "author", "nickname", "initials"}

DEMOGRAPHIC_KEYS = {
    "age", "gender", "sex", "handedness", "hand", "education", "ethnicity",
    "height", "weight", "updrs", "hoehnyahr", "diagnosis", "diagnosed",
    "medication", "onset", "duration",
}
# NOTE: 'y' is deliberately NOT a label key — in stroke data it is the y axis.
# CHÚ Ý: không đưa 'y' vào khoá nhãn vì trong dữ liệu nét vẽ đó là trục y.
LABEL_KEYS = {
    "label", "class", "group", "diagnosis", "diagnosed", "condition", "status",
    "target", "ispd", "is_parkinson", "pd", "category", "healthystatus",
}
TASK_KEYS = {"task", "taskid", "testid", "tasktype", "testtype", "type", "test", "shape"}
POINT_KEYS = ("strokes", "points", "path", "samples", "trajectory", "coordinates", "data")
TIME_KEYS = ("timestamp", "time", "t", "timestamps", "times", "createdat", "datetime", "date")

# Top-level keys that mean "this dict is a container, unpack my list".
# Whitelist on purpose: unpacking ANY list would swallow `strokes` (a drawing's
# own point array) and turn the stroke list into the "records".
# Chỉ mở gói các khoá bao bọc này, nếu không sẽ nuốt mất mảng `strokes`.
WRAPPER_KEYS = (
    "data", "records", "recordings", "items", "drawings", "sessions", "results",
    "entries", "rows", "captures", "samples", "files", "list", "content",
    "payload", "export", "exports",
)

# Only these parent keys make a scalar *inside an array* worth reporting.
SCALAR_ARRAY_KEYS = (
    PARTICIPANT_STRONG | PARTICIPANT_PROXY | DRAWING_LEVEL_IDS | PARTICIPANT_WEAK
    | DEMOGRAPHIC_KEYS | LABEL_KEYS | TASK_KEYS
)

# Positional fallback for header-less files that match the documented UCI-395
# layout: X; Y; Z; Pressure; GripAngle; Timestamp; Test ID
UCI_POSITIONAL_LAYOUT = ("x", "y", "z", "pressure", "gripangle", "timestamp", "testid")

MAX_NODES_PER_RECORD = 20_000   # guard against pathological JSON / chống treo
MAX_TRAVERSAL_DEPTH = 8
PROGRESS_EVERY = 50


# ===========================================================================
# Tiny formatting helpers / Hàm định dạng nhỏ
# ===========================================================================
def fmt_num(value: Any) -> str:
    """1_234_567 -> '1,234,567' (or 'n/a')."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


def rule(char: str = "=", width: int = 88) -> str:
    return char * width


def section(title: str, width: int = 88) -> None:
    print("\n" + rule("=", width))
    print(f" {title}")
    print(rule("=", width))


def subsection(title: str, width: int = 88) -> None:
    print(f"\n--- {title} " + "-" * max(0, width - len(title) - 5))


def kv(label: str, value: Any, indent: int = 2, width: int = 34) -> None:
    print(f"{' ' * indent}{label:<{width}}: {value}")


def pct(part: int, whole: int) -> str:
    if not whole:
        return "n/a"
    return f"{100.0 * part / whole:5.1f}%"


def _wrap_text(text: str, width: int = 82, indent: int = 0) -> List[str]:
    """Soft-wrap a long message so the console report stays readable."""
    return textwrap.wrap(text, width=width, initial_indent=" " * indent,
                         subsequent_indent=" " * indent) or [" " * indent]


def mount_google_drive() -> bool:
    """Mount Drive inside Colab (no-op elsewhere). Never raises."""
    if not IN_COLAB:
        return False
    try:
        if Path("/content/drive/MyDrive").exists():
            print("[drive] Google Drive already mounted / Drive đã được gắn.")
            return True
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive", force_remount=False)
        return Path("/content/drive/MyDrive").exists()
    except Exception as exc:  # pragma: no cover
        print(f"[drive][WARN] could not mount Google Drive / không gắn được Drive: {exc}")
        return False


# ===========================================================================
# Robust text + delimited-table reading / Đọc văn bản & bảng an toàn
# ===========================================================================
def read_text_safely(path: Path, max_bytes: Optional[int] = None) -> Tuple[str, str]:
    """Read a text file trying several encodings. Returns (text, encoding)."""
    last_error: Optional[Exception] = None
    for enc in ENCODINGS:
        try:
            with open(path, "r", encoding=enc, errors="strict") as fh:
                text = fh.read(max_bytes) if max_bytes else fh.read()
            return text, enc
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except Exception as exc:  # noqa: BLE001 - permission, is-a-directory...
            last_error = exc
            break
    # Final permissive fallback / phương án cuối: bỏ qua ký tự lỗi
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return (fh.read(max_bytes) if max_bytes else fh.read()), "utf-8/replace"
    except Exception as exc:  # pragma: no cover
        raise IOError(f"unreadable file / không đọc được file: {exc}") from last_error


_NUM_RE = re.compile(r"^[+-]?(\d+([.,]\d*)?|[.,]\d+)([eE][+-]?\d+)?$")


def looks_numeric(token: str) -> bool:
    return bool(_NUM_RE.match(token.strip()))


def sniff_delimiter(sample: str, preferred: str = ";") -> str:
    """Pick the delimiter: ';' wins when present (UCI spec), else best guess."""
    counts = {d: sample.count(d) for d in CANDIDATE_DELIMITERS}
    if counts.get(preferred, 0) > 0:
        return preferred
    best = max(counts, key=lambda d: counts[d])
    return best if counts[best] > 0 else preferred


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    try:
        return float(str(value).strip().replace(",", "."))
    except Exception:  # noqa: BLE001
        return None


def read_delimited_table(
    path: Path, preferred_delim: str = ";"
) -> Tuple[List[str], Dict[str, List[str]], str, bool, str]:
    """Read a delimited text file into columns of raw strings.

    Returns ``(columns, data, delimiter, had_header, encoding)`` where ``data``
    maps column name -> list of string cells (ragged rows are tolerated).
    Handles BOM, mixed line endings, ragged rows and a missing header row.

    Trả về ``(columns, data, delimiter, had_header, encoding)``. Chịu được BOM,
    xuống dòng hỗn hợp, dòng lệch cột và file KHÔNG có dòng tiêu đề.
    """
    text, encoding = read_text_safely(path)
    if not text.strip():
        return [], {}, preferred_delim, False, encoding

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return [], {}, preferred_delim, False, encoding

    delimiter = sniff_delimiter("\n".join(lines[:5]), preferred_delim)

    if HAS_PANDAS:  # fast path / đường nhanh
        try:
            import io

            df = pd.read_csv(
                io.StringIO(text),
                sep=delimiter,
                dtype=str,
                engine="python",
                on_bad_lines="skip",
                skipinitialspace=True,
                header=0,
            )
            had_header = True
            if len(df.columns) and all(
                looks_numeric(str(c)) or str(c).strip() == "" for c in df.columns
            ):
                # First row was data, not a header -> re-read with header=None
                df = pd.read_csv(
                    io.StringIO(text),
                    sep=delimiter,
                    dtype=str,
                    engine="python",
                    on_bad_lines="skip",
                    skipinitialspace=True,
                    header=None,
                )
                had_header = False
            columns = [str(c) for c in df.columns]
            data = {c: df[c].where(df[c].notna(), "").astype(str).tolist() for c in df.columns}
            data = {str(k): v for k, v in data.items()}
            return columns, data, delimiter, had_header, encoding
        except Exception:  # noqa: BLE001 - fall through to the stdlib reader
            pass

    # ---- stdlib fallback / phương án dự phòng bằng csv chuẩn ----
    reader = csv.reader(lines, delimiter=delimiter, skipinitialspace=True)
    rows = [r for r in reader if any(cell.strip() for cell in r)]
    if not rows:
        return [], {}, delimiter, False, encoding

    first = rows[0]
    had_header = any((not looks_numeric(cell)) and cell.strip() != "" for cell in first)
    if had_header:
        columns = [c.strip() for c in first]
        body = rows[1:]
    else:
        width = max(len(r) for r in rows)
        columns = [f"col_{i}" for i in range(width)]
        body = rows

    data: Dict[str, List[str]] = {c: [] for c in columns}
    for row in body:
        for i, col in enumerate(columns):
            data[col].append(row[i].strip() if i < len(row) else "")
    return columns, data, delimiter, had_header, encoding


def normalise_key(name: str) -> str:
    """' Grip Angle (deg) ' -> 'grip_angle_deg' ; 'participantId' -> 'participantid'."""
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower())
    return cleaned.strip("_").replace("_", "")


def find_column(columns: Sequence[str], aliases: Iterable[str]) -> Optional[str]:
    """Return the first column whose normalised name matches an alias."""
    alias_set = {normalise_key(a) for a in aliases}
    for col in columns:
        if normalise_key(col) in alias_set:
            return col
    return None


# ===========================================================================
# UCI (.txt) audit / Kiểm toán UCI (.txt)
# ===========================================================================
COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "x": ("x", "x_coord", "xcoordinate", "xposition", "pos_x", "xpos"),
    "y": ("y", "y_coord", "ycoordinate", "yposition", "pos_y", "ypos"),
    "z": ("z", "z_coord", "zcoordinate"),
    "pressure": ("pressure", "press", "force", "pen_pressure"),
    "gripangle": ("gripangle", "grip_angle", "angle", "grip", "tilt"),
    "timestamp": ("timestamp", "time", "t", "timestamp_ms", "time_ms", "time_sec",
                  "systemtime", "system_time", "time_stamp", "datetime"),
    "testid": ("testid", "test_id", "test", "task", "taskid", "task_id",
               "testtype", "test_type", "testno", "test_number"),
    "subjectid": ("subjectid", "subject_id", "subject", "patientid", "patient_id",
                  "participantid", "participant_id", "userid", "user_id", "subj"),
    "label": ("diagnosis", "class", "label", "group", "target", "condition", "status"),
}

CONTROL_ALIASES = ("control", "controls", "healthy", "health", "hc", "normal", "0", "ctl")
PD_ALIASES = ("parkinson", "parkinsons", "pd", "pwp", "patient", "patients", "1", "disease")


@dataclass
class UciFileAudit:
    """Per-file audit result for one UCI .txt recording."""

    path: Path
    rel: str
    size_bytes: int = 0
    ok: bool = False
    error: Optional[str] = None
    delimiter: str = ";"
    encoding: str = ""
    had_header: bool = True
    columns: List[str] = field(default_factory=list)
    n_rows: int = 0
    test_ids: Counter = field(default_factory=Counter)      # test_id -> samples
    test_runs: Counter = field(default_factory=Counter)     # test_id -> trials
    dataset: str = ""                                       # hw_dataset / new_dataset / ...
    subject_raw: str = ""                                   # file stem, untouched
    subject_norm: str = ""                                  # stem minus trailing trial index
    group: str = "unknown"                                  # control / parkinson / unknown
    group_source: str = ""
    infile_subject: Optional[str] = None
    infile_label: Optional[str] = None
    duration_s: Optional[float] = None
    n_bad_xy: int = 0
    ts_monotonic: Optional[bool] = None


def normalise_subject_code(stem: str) -> str:
    """Strip a trailing trial/index block: 'H_P000-0025' -> 'H_P000'.

    The UCI-395 `new_dataset/` split appends a recording number to the subject
    code; leaving it on would count one person three times. Only the LAST
    numeric block is stripped, and only when >= 2 numeric blocks exist, so
    'C_0009' and 'P_27110003' stay untouched.
    """
    blocks = list(re.finditer(r"\d+", stem))
    if len(blocks) >= 2:
        last = blocks[-1]
        if last.end() == len(stem):
            head = stem[: last.start()].rstrip("-_ .")
            if head:
                return head
    return stem


def infer_group_from_path(rel_path: Path) -> Tuple[str, str]:
    """Infer clinical class from directory names, then from the filename."""
    parts = [p.lower() for p in rel_path.parts]
    # Nearest directory first / ưu tiên thư mục gần nhất
    for part in reversed(parts[:-1]):
        token = re.sub(r"[^a-z]", "", part)
        if token in PD_TOKENS:
            return "parkinson", f"dir:{part}"
        if token in CONTROL_TOKENS:
            return "control", f"dir:{part}"
    stem = rel_path.stem
    # 'C_0009' -> control ; 'P_27110003' / 'H_P000-0025' -> parkinson
    match = re.match(r"^[A-Za-z][-_ ]?([CPcp])[-_ ]?\d", stem)
    if match:
        letter = match.group(1).upper()
        if letter == "C":
            return "control", "filename-prefix:C"
        if letter == "P":
            return "parkinson", "filename-prefix:P"
    return "unknown", ""


def normalise_test_id(raw: str) -> Optional[int]:
    """Map a raw Test-ID cell onto 0/1/2 (numeric or textual)."""
    if raw is None:
        return None
    token = str(raw).strip()
    if not token:
        return None
    value = _to_float(token)
    if value is not None and float(value).is_integer():
        ivalue = int(value)
        if ivalue in TEST_ID_LABELS:
            return ivalue
    low = token.lower()
    for hint, tid in TEST_ID_TEXT_HINTS:
        if hint in low:
            return tid
    return None


def infer_duration_seconds(times: Sequence[float]) -> Tuple[Optional[float], str]:
    """Return (duration_in_seconds, unit_guess) for a timestamp column.

    The unit is decided from the MEDIAN SAMPLE INTERVAL, not from the absolute
    span: a 4-second recording sampled at 100 Hz has dt = 10 (ms) — a small
    number that a naive 'span > 10000 ? ms : s' test would read as seconds and
    report a 4,990-second drawing.

    Đơn vị được suy ra từ KHOẢNG CÁCH MẪU TRUNG VỊ (dt), không phải từ độ lớn
    tuyệt đối, để tránh đọc nhầm mili-giây thành giây.
    """
    if len(times) < 2:
        return None, "unknown"
    deltas = [b - a for a, b in zip(times, times[1:]) if b > a]
    if not deltas:
        return 0.0, "unknown"
    median_dt = statistics.median(deltas)
    span = max(times) - min(times)
    unit = "ms" if median_dt >= 1.0 else "s"
    return (span / 1000.0 if unit == "ms" else span), unit


def count_trials(test_ids: Sequence[Optional[int]]) -> Counter:
    """Count contiguous runs: a new trial starts whenever the Test-ID changes.

    UCI files store each test as one contiguous block, so a file holding
    SST+DST+STCP yields 3 trials. Đếm số "trial" (đoạn liên tiếp cùng Test-ID).
    """
    runs: Counter = Counter()
    previous: Any = object()  # sentinel: nothing seen yet
    for tid in test_ids:
        if tid is None:
            previous = tid
            continue
        if tid != previous:
            runs[tid] += 1
            previous = tid
    return runs


def audit_uci_file(path: Path, root: Path) -> UciFileAudit:
    """Audit ONE UCI .txt file. Never raises — errors land in ``.error``."""
    rel = path.relative_to(root) if str(path).startswith(str(root)) else path
    audit = UciFileAudit(path=path, rel=str(rel))
    try:
        audit.size_bytes = path.stat().st_size
        if audit.size_bytes == 0:
            audit.error = "empty file / file rỗng"
            return audit

        columns, data, delim, had_header, encoding = read_delimited_table(path)
        if not columns:
            audit.error = "no parsable rows / không có dòng nào đọc được"
            return audit

        # Header-less file that matches the documented UCI-395 column order:
        # X; Y; Z; Pressure; GripAngle; Timestamp; Test ID  -> adopt it positionally
        # so the Test-ID census still works. (File không có tiêu đề nhưng đúng
        # thứ tự cột chuẩn -> gán tên cột theo vị trí.)
        if not had_header and len(columns) == len(UCI_POSITIONAL_LAYOUT):
            renamed: Dict[str, List[str]] = {}
            for original, canonical in zip(columns, UCI_POSITIONAL_LAYOUT):
                renamed[canonical] = data.pop(original, [])
            columns = list(UCI_POSITIONAL_LAYOUT)
            data = renamed

        audit.columns = [str(c) for c in columns]
        audit.delimiter = delim
        audit.had_header = had_header
        audit.encoding = encoding
        audit.n_rows = len(data[columns[0]]) if columns else 0
        if audit.n_rows == 0:
            audit.error = "header only, no data rows / chỉ có tiêu đề"
            return audit

        # ---- Test-ID census ----
        test_col = find_column(columns, COLUMN_ALIASES["testid"])
        if test_col is not None:
            raw_test = data[test_col]
            normalised = [normalise_test_id(v) for v in raw_test]
            audit.test_ids = Counter(t for t in normalised if t is not None)
            audit.test_runs = count_trials(normalised)
            if not audit.test_ids:
                audit.error = (
                    f"Test-ID column '{test_col}' found but no value mapped to 0/1/2 "
                    f"/ có cột Test-ID nhưng không map được 0/1/2 "
                    f"(samples: {sorted(set(raw_test))[:5]})"
                )
        else:
            # No Test-ID column -> infer a single test from the file name.
            inferred = normalise_test_id(path.stem) or normalise_test_id(str(rel))
            if inferred is None:
                audit.error = (
                    "no Test-ID column and no keyword in the filename "
                    "/ không có cột Test-ID và tên file không gợi ý bài test"
                )
            else:
                audit.test_ids = Counter({inferred: audit.n_rows})
                audit.test_runs = Counter({inferred: 1})

        # ---- Identity / định danh ----
        audit.subject_raw = path.stem
        audit.subject_norm = normalise_subject_code(path.stem)
        audit.dataset = path.parent.name if path.parent != root else ""
        audit.group, audit.group_source = infer_group_from_path(Path(rel))

        subj_col = find_column(columns, COLUMN_ALIASES["subjectid"])
        if subj_col is not None:
            values = [v.strip() for v in data[subj_col] if str(v).strip()]
            if values:
                audit.infile_subject = Counter(values).most_common(1)[0][0]
                if not audit.group_source:
                    audit.group = "unknown"

        label_col = find_column(columns, COLUMN_ALIASES["label"])
        if label_col is not None:
            values = [v.strip().lower() for v in data[label_col] if str(v).strip()]
            if values:
                label = Counter(values).most_common(1)[0][0]
                audit.infile_label = label
                if label in CONTROL_ALIASES:
                    audit.group, audit.group_source = "control", f"column:{label_col}"
                elif label in PD_ALIASES:
                    audit.group, audit.group_source = "parkinson", f"column:{label_col}"

        # ---- Data quality / chất lượng dữ liệu ----
        ts_col = find_column(columns, COLUMN_ALIASES["timestamp"])
        if ts_col is not None:
            times = [t for t in (_to_float(v) for v in data[ts_col]) if t is not None]
            if len(times) >= 2:
                audit.duration_s, unit = infer_duration_seconds(times)
                del unit
                audit.ts_monotonic = all(b >= a for a, b in zip(times, times[1:]))
        x_col = find_column(columns, COLUMN_ALIASES["x"])
        y_col = find_column(columns, COLUMN_ALIASES["y"])
        if x_col and y_col:
            audit.n_bad_xy = sum(
                1 for xv, yv in zip(data[x_col], data[y_col])
                if _to_float(xv) is None or _to_float(yv) is None
            )

        audit.ok = True
        return audit
    except Exception as exc:  # noqa: BLE001 - a bad file must never kill the audit
        audit.ok = False
        audit.error = f"{type(exc).__name__}: {exc}"
        return audit


def audit_uci_root(root: Path, limit: Optional[int] = None, quiet: bool = False) -> Dict[str, Any]:
    """Recursively audit every .txt under ``root``."""
    result: Dict[str, Any] = {
        "root": root,
        "exists": False,
        "files": [],
        "n_files": 0,
        "n_ok": 0,
        "n_failed": 0,
        "errors": [],
        "n_samples": 0,
        "delimiters": Counter(),
        "encodings": Counter(),
        "columns_seen": Counter(),
        "header_variants": set(),
        "skipped": 0,
    }
    if not root.exists():
        result["errors"].append((str(root), "root does not exist / thư mục không tồn tại"))
        return result
    result["exists"] = True

    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".txt")
    if limit:
        files = files[:limit]
        result["skipped"] = max(0, len(list(root.rglob("*.txt"))) - len(files))
    result["n_files"] = len(files)

    for index, path in enumerate(files, start=1):
        if not quiet and (index % PROGRESS_EVERY == 0 or index == len(files)):
            print(f"    ... scanned {index}/{len(files)} .txt files", flush=True)
        audit = audit_uci_file(path, root)
        result["files"].append(audit)
        if audit.ok:
            result["n_ok"] += 1
            result["n_samples"] += audit.n_rows
            result["delimiters"][audit.delimiter] += 1
            result["encodings"][audit.encoding] += 1
            result["columns_seen"].update(audit.columns)
            result["header_variants"].add(tuple(audit.columns))
        else:
            result["n_failed"] += 1
            result["errors"].append((str(path), audit.error or "unknown error"))
    return result


# ===========================================================================
# JSON audit / Kiểm toán JSON
# ===========================================================================
def iter_scalar_fields(node: Any, max_nodes: int = MAX_NODES_PER_RECORD) -> Iterable[Tuple[str, str, Any]]:
    """Yield ``(json_path, normalised_key, value)`` for every scalar in ``node``.

    Depth- and node-limited so a 500 MB JSON can never hang the audit.
    Giới hạn độ sâu & số node để file JSON khổng lồ không làm treo script.
    """
    visited = 0
    stack: List[Tuple[str, Any, int]] = [("", node, 0)]
    while stack:
        path, current, depth = stack.pop()
        visited += 1
        if visited > max_nodes or depth > MAX_TRAVERSAL_DEPTH:
            return
        if isinstance(current, dict):
            for key, value in current.items():
                child = f"{path}.{key}" if path else str(key)
                if isinstance(value, (dict, list)):
                    stack.append((child, value, depth + 1))
                elif value is not None and not isinstance(value, (dict, list)):
                    yield child, normalise_key(key), value
        elif isinstance(current, list):
            for value in current[:50]:  # sample the head of huge arrays
                if isinstance(value, (dict, list)):
                    stack.append((path, value, depth + 1))
            # Scalars inside an array are only interesting when the parent key
            # is an id / label-ish key — otherwise this floods the audit with
            # x/y/t/pressure numbers.
            parent_key = normalise_key(path.split(".")[-1]) if path else ""
            if parent_key in SCALAR_ARRAY_KEYS:
                for value in current[:5]:
                    if value is not None and not isinstance(value, (dict, list)):
                        yield path, parent_key, value


def classify_identity_keys(record: Any) -> Dict[str, List[Tuple[str, Any]]]:
    """Split every id-like key into strong / proxy / drawing-level / weak.

    This is the crux of the JSON audit: ``{"id": 812}`` on every file is a
    DRAWING id, not a person. Đây là điểm mấu chốt: ``{"id": 812}`` là mã bản
    vẽ, KHÔNG PHẢI mã người.
    """
    buckets: Dict[str, List[Tuple[str, Any]]] = {
        "strong": [], "proxy": [], "drawing": [], "weak": [], "demographic": [],
        "label": [], "task": [],
    }
    for path, key, value in iter_scalar_fields(record):
        if isinstance(value, str) and not value.strip():
            continue
        if key in PARTICIPANT_STRONG:
            buckets["strong"].append((path, value))
        elif key in PARTICIPANT_PROXY:
            buckets["proxy"].append((path, value))
        elif key in DRAWING_LEVEL_IDS:
            buckets["drawing"].append((path, value))
        elif key in PARTICIPANT_WEAK:
            buckets["weak"].append((path, value))
        elif key in DEMOGRAPHIC_KEYS:
            buckets["demographic"].append((path, value))
        elif key in LABEL_KEYS:
            buckets["label"].append((path, value))
        elif key in TASK_KEYS:
            buckets["task"].append((path, value))
    return buckets


def count_points(record: Any) -> Optional[int]:
    """Best-effort number of sampled points in one drawing record."""
    if not isinstance(record, dict):
        return None
    for key in POINT_KEYS:
        if key in record:
            value = record[key]
            if isinstance(value, list):
                if all(isinstance(item, list) for item in value) and value:
                    return sum(len(item) for item in value)
                return len(value)
            if isinstance(value, dict):
                for axis in ("x", "y"):
                    if isinstance(value.get(axis), list):
                        return len(value[axis])
    for axis in ("x", "y"):
        if isinstance(record.get(axis), list):
            return len(record[axis])
    return None


def has_time_field(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    for key in record:
        if normalise_key(key) in {normalise_key(t) for t in TIME_KEYS}:
            return True
    for key in ("strokes", "points", "samples"):
        value = record.get(key)
        if isinstance(value, list) and value and isinstance(value[0], dict):
            if any(normalise_key(k) in {normalise_key(t) for t in TIME_KEYS} for k in value[0]):
                return True
    return False


def load_json_records(path: Path) -> Tuple[List[Any], Optional[str], Optional[Tuple[str, Any]]]:
    """Load a JSON file into a list of records.

    Tries (1) whole-file JSON, (2) a container key such as ``{"drawings": [...]}``,
    (3) JSON-Lines. Returns ``(records, error_message, wrapper_identity)`` where
    ``wrapper_identity`` is ``(key, value)`` if the CONTAINER itself carries a
    participant id (it then applies to every drawing inside it).

    Trả về ``(records, lỗi, định_danh_bao_bọc)``. Nếu object bao ngoài có mã
    người tham gia thì mã đó áp dụng cho mọi bản vẽ bên trong.
    """
    text, _ = read_text_safely(path)
    if not text.strip():
        return [], "empty file / file rỗng", None
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            # Only whitelisted container keys are unpacked — never `strokes`.
            for key in WRAPPER_KEYS:
                value = payload.get(key)
                if isinstance(value, list):
                    return list(value), None, _wrapper_identity(payload)
            return [payload], None, None
        if isinstance(payload, list):
            return list(payload), None, None
        return [payload], None, None
    except Exception as first_error:  # noqa: BLE001
        # JSON-Lines fallback / thử định dạng JSON Lines
        records: List[Any] = []
        ok_lines = 0
        for line in text.splitlines():
            line = line.strip().rstrip(",")
            if not line or line in ("[", "]", "{", "}"):
                continue
            try:
                records.append(json.loads(line))
                ok_lines += 1
            except Exception:  # noqa: BLE001
                continue
        if ok_lines:
            return records, (
                f"recovered {ok_lines} JSON-Lines records "
                f"(whole-file parse failed: {type(first_error).__name__})"
            ), None
        return [], f"malformed JSON / JSON hỏng: {first_error}", None


def _wrapper_identity(payload: dict) -> Optional[Tuple[str, Any]]:
    """Return (key, value) if the container dict itself names a participant."""
    for key, value in payload.items():
        norm = normalise_key(key)
        if norm in PARTICIPANT_STRONG and value is not None and str(value).strip():
            return (str(key), value)
    for key, value in payload.items():
        norm = normalise_key(key)
        if norm in PARTICIPANT_PROXY and value is not None and str(value).strip():
            return (str(key), value)
    return None


def filename_participant_hint(rel_path: Path) -> Optional[str]:
    """Last-resort identity: pull a person-looking token out of the path.

    e.g. 'participant_07/session2.json' -> 'participant_07' ;
         'p12_spiral_3.json'            -> 'p12'
    """
    patterns = [
        r"(participant[-_ ]?\w?\d+)",
        r"(subject[-_ ]?\w?\d+)",
        r"(patient[-_ ]?\w?\d+)",
        r"(user[-_ ]?\w?\d+)",
        r"(person[-_ ]?\w?\d+)",
        r"(volunteer[-_ ]?\w?\d+)",
        r"(?:^|[_\-/])([pc]\d{1,4})(?:[_./-]|$)",
    ]
    text = str(rel_path).lower()
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).lower()
    # A generic folder name ("batch2", "session1"…) is NOT a person: inventing a
    # participant id here would silently merge unrelated drawings into one
    # "subject" and understate the identity problem. Return None instead.
    # Tên thư mục chung chung KHÔNG phải con người — trả về None để báo "không rõ".
    return None


def choose_identity_key(
    key_files: Counter, key_values: Dict[str, Counter], key_tiers: Dict[str, str]
) -> Tuple[Optional[str], str, str]:
    """Decide, after the whole scan, which field may be treated as a PERSON.

    The decisive test is REPETITION: a participant id must appear on MORE THAN
    ONE drawing. A field whose value is unique for every single drawing is a
    drawing id wearing a person's name — accepting it would silently report
    "N drawings = N participants" and hide the real problem.

      * strong-named key + 1 value per drawing -> kept, but FLAGGED (it may
        legitimately be one drawing per person, or a mislabelled drawing id).
      * proxy/weak key  + 1 value per drawing -> DEMOTED to "not a person".

    Trả về ``(key, tier, note)``. Tiêu chí chính là SỰ LẶP LẠI: mã người phải
    xuất hiện trên nhiều hơn một bản vẽ.
    """
    chosen: Optional[str] = None
    for tier in ("strong", "proxy", "weak", "filename"):
        keys = [k for k in key_files if key_tiers.get(k) == tier]
        if keys:
            chosen = max(keys, key=lambda k: sum(key_values[k].values()))
            break
    if chosen is None:
        return None, "none", ""

    tier = key_tiers.get(chosen, "weak")
    distinct = len(key_values[chosen])
    occurrences = sum(key_values[chosen].values())
    note = ""
    if occurrences > 1 and distinct == occurrences:
        if tier in ("proxy", "weak"):
            note = (
                f"DEMOTED: '{chosen}' has a unique value on every drawing "
                f"({distinct} values / {occurrences} drawings) — that is a DRAWING "
                f"id, not a person. Those drawings are reported as UNIDENTIFIED."
            )
            return None, "none", note
        note = (
            f"SUSPECT: '{chosen}' has a unique value on every drawing "
            f"({distinct} values / {occurrences} drawings). Either every person "
            f"drew exactly once, or this field is a drawing id mislabelled as a "
            f"participant id — verify it by hand."
        )
    return chosen, tier, note


@dataclass
class JsonFileAudit:
    """Per-file audit result for one .json drawing capture."""

    path: Path
    rel: str
    size_bytes: int = 0
    ok: bool = False
    error: Optional[str] = None
    warning: Optional[str] = None
    n_records: int = 0
    participants: List[str] = field(default_factory=list)
    identity_tier: str = "none"        # strong / proxy / weak / filename / none
    identity_key: str = ""
    # (tier, json_path, value) for every participant-shaped field found.
    identity_candidates: List[Tuple[str, str, str]] = field(default_factory=list)
    drawing_ids: List[str] = field(default_factory=list)
    tasks: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    points: List[int] = field(default_factory=list)
    has_time: bool = False
    content_hash: str = ""
    top_keys: List[str] = field(default_factory=list)


def audit_json_file(path: Path, root: Path) -> JsonFileAudit:
    """Audit ONE .json file. Never raises — errors land in ``.error``."""
    rel = path.relative_to(root) if str(path).startswith(str(root)) else path
    audit = JsonFileAudit(path=path, rel=str(rel))
    try:
        audit.size_bytes = path.stat().st_size
        if audit.size_bytes == 0:
            audit.error = "empty file / file rỗng"
            return audit

        records, error, wrapper_identity = load_json_records(path)
        if not records:
            audit.error = error or "no records / không có bản ghi"
            return audit
        audit.warning = error  # set only for the JSON-Lines recovery case
        audit.n_records = len(records)

        raw_text, _ = read_text_safely(path)
        audit.content_hash = hashlib.sha256(
            re.sub(r"\s+", "", raw_text).encode("utf-8", errors="replace")
        ).hexdigest()

        dict_records = [r for r in records if isinstance(r, dict)]
        for record in dict_records:
            audit.top_keys.extend(str(k) for k in record.keys())
            buckets = classify_identity_keys(record)

            # Collect every participant-shaped field; WHICH one is trustworthy
            # is decided globally after the whole scan (see choose_identity_key).
            # Ghi lại mọi trường "giống mã người"; quyết định tin trường nào được
            # đưa ra SAU khi quét xong (xem choose_identity_key).
            for tier in ("strong", "proxy", "weak"):
                for key_path, value in buckets[tier]:
                    text = str(value).strip()
                    if text:
                        audit.identity_candidates.append((tier, key_path, text))

            for _, value in buckets["drawing"]:
                audit.drawing_ids.append(str(value))
            for _, value in buckets["task"]:
                audit.tasks.append(str(value))
            for _, value in buckets["label"]:
                audit.labels.append(str(value))

            points = count_points(record)
            if points is not None:
                audit.points.append(points)
            audit.has_time = audit.has_time or has_time_field(record)

        # The container itself may name the participant
        # (e.g. {"participantId": "P01", "drawings": [...]}) — it then applies
        # to every drawing inside. / Object bao ngoài có thể chứa mã người.
        if not audit.identity_candidates and wrapper_identity:
            key, value = wrapper_identity
            audit.identity_candidates.append(("strong", f"container:{key}", str(value).strip()))

        # One entry per (field, value) per FILE — a file is normally one
        # participant, so repeated records must not inflate the field counts.
        # Mỗi (trường, giá trị) chỉ tính MỘT lần cho mỗi file.
        seen: set = set()
        deduped: List[Tuple[str, str, str]] = []
        for candidate in audit.identity_candidates:
            marker = (candidate[1], candidate[2])
            if marker not in seen:
                seen.add(marker)
                deduped.append(candidate)
        audit.identity_candidates = deduped

        # Last resort: guess from the path (and say so).
        if not audit.identity_candidates:
            hint = filename_participant_hint(Path(rel))
            if hint:
                audit.identity_candidates.append(("filename", "filename-heuristic", hint))

        audit.ok = True
        return audit
    except Exception as exc:  # noqa: BLE001
        audit.ok = False
        audit.error = f"{type(exc).__name__}: {exc}"
        return audit


def audit_json_root(root: Path, limit: Optional[int] = None, quiet: bool = False) -> Dict[str, Any]:
    """Recursively audit every .json under ``root``."""
    result: Dict[str, Any] = {
        "root": root,
        "exists": False,
        "files": [],
        "n_files": 0,
        "n_ok": 0,
        "n_failed": 0,
        "errors": [],
        "n_records": 0,
        "participants": Counter(),          # pid (casefolded) -> drawings
        "participant_display": defaultdict(Counter),  # pid -> original spellings
        "n_unidentified_records": 0,        # drawings with NO identity at all
        "n_recovered": 0,                   # files salvaged as JSON-Lines
        "participant_tiers": Counter(),     # tier -> files
        "identity_keys": Counter(),         # json path -> files using it
        "drawing_id_uniqueness": Counter(), # drawing id -> occurrences
        "tasks": Counter(),
        "labels": Counter(),
        "points": [],
        "n_with_time": 0,
        "n_empty_drawings": 0,
        "hashes": Counter(),
        "key_freq": Counter(),
        "skipped": 0,
        "identity_key": None,
        "identity_tier": "none",
        "identity_note": "",
        "demoted_keys": [],
        "candidate_keys": Counter(),   # json path -> files carrying it
        "candidate_tiers": {},         # json path -> tier
        "candidate_values": {},        # json path -> distinct values
    }
    if not root.exists():
        result["errors"].append((str(root), "root does not exist / thư mục không tồn tại"))
        return result
    result["exists"] = True

    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".json")
    if limit:
        result["skipped"] = max(0, len(files) - limit)
        files = files[:limit]
    result["n_files"] = len(files)

    # Candidate participant fields discovered during pass 1.
    key_files: Counter = Counter()                    # json path -> files
    key_values: Dict[str, Counter] = defaultdict(Counter)   # json path -> {value: n}
    key_tiers: Dict[str, str] = {}                    # json path -> tier

    for index, path in enumerate(files, start=1):
        if not quiet and (index % PROGRESS_EVERY == 0 or index == len(files)):
            print(f"    ... scanned {index}/{len(files)} .json files", flush=True)
        audit = audit_json_file(path, root)
        result["files"].append(audit)
        if not audit.ok:
            result["n_failed"] += 1
            result["errors"].append((str(path), audit.error or "unknown error"))
            continue

        result["n_ok"] += 1
        result["n_records"] += audit.n_records
        if audit.warning:
            result["n_recovered"] += 1
        for key in audit.top_keys:
            result["key_freq"][key] += 1
        for did in audit.drawing_ids:
            result["drawing_id_uniqueness"][did] += 1
        for task in audit.tasks:
            result["tasks"][task] += 1
        for label in audit.labels:
            result["labels"][str(label).lower()] += 1
        result["points"].extend(audit.points)
        if audit.has_time:
            result["n_with_time"] += 1
        if audit.points and sum(audit.points) == 0:
            result["n_empty_drawings"] += 1
        result["hashes"][audit.content_hash] += 1

        # Candidate participant fields, indexed by json path.
        for tier, key_path, value in audit.identity_candidates:
            key_tiers[key_path] = tier
            key_files[key_path] += 1
            key_values[key_path][value.strip().casefold()] += 1

    # ---- PASS 2: decide which field may count as a person, then aggregate ----
    result["identity_key"], result["identity_tier"], result["identity_note"] = \
        choose_identity_key(key_files, key_values, key_tiers)
    result["candidate_keys"] = key_files
    result["candidate_tiers"] = key_tiers
    result["candidate_values"] = {k: len(v) for k, v in key_values.items()}
    result["candidate_occurrences"] = {k: sum(v.values()) for k, v in key_values.items()}
    if result["identity_note"].startswith("DEMOTED"):
        result["demoted_keys"] = [k for k in key_files if key_tiers.get(k) in ("proxy", "weak")]

    for audit in result["files"]:
        if not audit.ok:
            continue
        pid = None
        for tier, key_path, value in audit.identity_candidates:
            if key_path == result["identity_key"]:
                pid = value
                break
        if pid is None:
            audit.identity_tier = "none"
            audit.identity_key = ""
            result["n_unidentified_records"] += max(1, audit.n_records)
        else:
            audit.identity_tier = result["identity_tier"]
            audit.identity_key = result["identity_key"]
            audit.participants = [pid]
            key = pid.strip().casefold()     # 'P01' and 'p01' are one person
            # Count DRAWINGS (records), not files: one JSON-Lines file may hold
            # several drawings. / Đếm bản vẽ (record), không đếm file.
            result["participants"][key] += max(1, audit.n_records)
            result["participant_display"][key][pid] += 1
            result["identity_keys"][result["identity_key"]] += 1
            result["n_identified_records"] = result.get("n_identified_records", 0) + max(
                1, audit.n_records
            )
        result["participant_tiers"][audit.identity_tier] += 1
    return result


# ===========================================================================
# Reporting / Báo cáo
# ===========================================================================
def summarise_uci_subjects(uci: Dict[str, Any]) -> Dict[str, Any]:
    """Compute unique-subject counts under 6 competing identity strategies."""
    ok_files: List[UciFileAudit] = [f for f in uci["files"] if f.ok]
    n_files = len(ok_files)

    s1 = n_files                                                    # 1 file = 1 subject
    s2 = len({(f.dataset, f.subject_raw) for f in ok_files})         # dataset + stem
    s3 = len({f.subject_raw for f in ok_files})                      # stem only
    s4 = len({(f.dataset, f.subject_norm) for f in ok_files})        # dataset + normalised
    s5 = len({f.subject_norm for f in ok_files})                     # normalised only
    infile = {f.infile_subject for f in ok_files if f.infile_subject}
    s6 = len(infile) if infile else None

    strategies = [
        ("S1  one file = one subject (the stated assumption)", s1,
         "what you assumed / giả định ban đầu"),
        ("S2  dataset-folder + file stem", s2,
         "splits hw_dataset vs new_dataset"),
        ("S3  file stem only", s3,
         "'H_P000-0025' and 'H_P000-0026' count as 2 people"),
        ("S4  dataset-folder + normalised code", s4,
         "trailing recording index stripped, per split"),
        ("S5  normalised subject code (RECOMMENDED)", s5,
         "'H_P000-0025' -> 'H_P000' = 1 person"),
        ("S6  subject-id column inside the files", s6,
         "only if the .txt files carry a Subject-ID column"),
    ]

    # Recommended = S5 (cross-split de-duplication), unless the files carry an
    # explicit subject column, which is always the strongest evidence.
    if s6:
        recommended_key: Any = "S6"
        key_fn = lambda f: f.infile_subject  # noqa: E731
        recommended = s6
    else:
        recommended_key = "S5"
        key_fn = lambda f: f.subject_norm    # noqa: E731
        recommended = s5

    # Class split under the recommended identity.
    groups: Dict[str, set] = defaultdict(set)
    ambiguous: set = set()
    for f in ok_files:
        key = key_fn(f)
        if not key:
            continue
        groups[f.group].add(key)
    if "control" in groups and "parkinson" in groups:
        ambiguous = groups["control"] & groups["parkinson"]

    return {
        "n_files_ok": n_files,
        "strategies": strategies,
        "recommended_key": recommended_key,
        "recommended": recommended,
        "groups": {g: len(v) for g, v in groups.items()},
        "ambiguous": ambiguous,
        "multi_test_files": sum(1 for f in ok_files if len(f.test_ids) > 1),
        "multi_dataset_codes": [
            code for code in {f.subject_norm for f in ok_files}
            if len({f.dataset for f in ok_files if f.subject_norm == code}) > 1
        ],
        "infile_subject_present": bool(infile),
    }


def print_uci_report(uci: Dict[str, Any], subjects: Dict[str, Any], show_errors: int) -> None:
    section("[1] UCI ROOT — .txt SCAN / QUÉT .txt")

    kv("UCI root", str(uci["root"]))
    kv("root exists", "YES" if uci["exists"] else "NO  <-- CHECK YOUR PATH")
    if not uci["exists"]:
        for path, err in uci["errors"][:show_errors]:
            print(f"      ! {path}: {err}")
        print("\n  >> Nothing to audit under the UCI root. / Không có gì để kiểm toán.")
        return

    kv("files discovered", fmt_num(uci["n_files"]))
    kv("files parsed OK", f"{fmt_num(uci['n_ok'])}  ({pct(uci['n_ok'], uci['n_files'])})")
    kv("files FAILED", f"{fmt_num(uci['n_failed'])}  ({pct(uci['n_failed'], uci['n_files'])})")
    kv("total samples (rows)", fmt_num(uci["n_samples"]))
    if uci["skipped"]:
        kv("files skipped by --limit", fmt_num(uci["skipped"]))
    if uci["delimiters"]:
        kv("delimiters detected",
           ", ".join(f"{d!r} x{n}" for d, n in uci["delimiters"].most_common()))
    if uci["encodings"]:
        kv("encodings detected",
           ", ".join(f"{e} x{n}" for e, n in uci["encodings"].most_common()))

    subsection("Detected column layout / Cấu trúc cột phát hiện được")
    variants = sorted(uci["header_variants"], key=lambda v: "|".join(v))  # deterministic
    if not variants:
        print("      (none)")
    else:
        print(f"      {len(variants)} distinct header layout(s) / "
              f"{len(variants)} kiểu tiêu đề khác nhau:")
        for variant in variants[:5]:
            print(f"        - {' | '.join(variant)}")
        if len(variants) > 5:
            print(f"        - ... and {len(variants) - 5} more")

    ok_files = [f for f in uci["files"] if f.ok]
    if ok_files:
        durations = [f.duration_s for f in ok_files if f.duration_s is not None]
        if durations:
            subsection("Recording duration / Thời lượng bản ghi")
            kv("files with timestamps", fmt_num(len(durations)))
            kv("median duration (s)", f"{statistics.median(durations):,.1f}")
            kv("min / max (s)", f"{min(durations):,.1f} / {max(durations):,.1f}")
        non_mono = [f for f in ok_files if f.ts_monotonic is False]
        if non_mono:
            kv("non-monotonic timestamps",
               f"{fmt_num(len(non_mono))} file(s)  <-- trials concatenated in one file")
        bad_xy = sum(f.n_bad_xy for f in ok_files)
        if bad_xy:
            kv("non-numeric x/y cells", fmt_num(bad_xy))

    # ------------------------------------------------------------------
    section("[2] UCI — HOW MANY UNIQUE SUBJECTS? / BAO NHIÊU NGƯỜI DUY NHẤT?")
    print("  'Assuming each file = 1 subject' is an ASSUMPTION. Here it is tested")
    print("  against 5 alternative readings of the same files:")
    print("  (Giả định '1 file = 1 người' cần được KIỂM CHỨNG, không mặc định tin.)")
    print()
    print(f"      {'strategy':<52}{'subjects':>12}")
    print(f"      {'-' * 52}{'-' * 12}")
    for label, value, note in subjects["strategies"]:
        shown = fmt_num(value) if value is not None else "n/a"
        print(f"      {label:<52}{shown:>12}")
    print()
    for label, value, note in subjects["strategies"]:
        print(f"        * {label.split()[0]}: {note}")

    print()
    print(f"  >>> UNIQUE SUBJECTS (UCI, recommended = {subjects['recommended_key']}): "
          f"{fmt_num(subjects['recommended'])}")
    for group, count in sorted(subjects["groups"].items()):
        expected = UCI_EXPECTED.get(group)
        flag = ""
        if expected is not None and count != expected:
            flag = f"   <-- literature says {expected}"
        print(f"        {group:<12}: {fmt_num(count)}{flag}")

    if subjects["multi_test_files"]:
        n_multi = fmt_num(subjects["multi_test_files"])
        print(f"\n  NOTE / LƯU Ý: {n_multi} file(s) contain MORE THAN ONE Test-ID")
        print("       -> a single file is a SESSION (several tests), so '1 file = 1")
        print("          subject' is plausible but '1 file = 1 trial' is NOT.")
    if subjects["multi_dataset_codes"]:
        print(f"\n  NOTE: {len(subjects['multi_dataset_codes'])} subject code(s) appear in BOTH "
              f"hw_dataset and new_dataset")
        print("        (counted once, per strategy S5): "
              f"{', '.join(sorted(subjects['multi_dataset_codes'])[:8])}")
    if subjects["ambiguous"]:
        print(f"\n  [WARN] {len(subjects['ambiguous'])} subject code(s) appear under BOTH the "
              f"control and the parkinson")
        print("         folders — class label is ambiguous for these: "
              f"{', '.join(sorted(subjects['ambiguous'])[:8])}")

    expected_total = UCI_EXPECTED["total"]
    if expected_total and subjects["recommended"] != expected_total:
        print(f"\n  [WARN] UCI-395 is documented as 77 subjects (62 PD + 15 controls).")
        print(f"         We counted {subjects['recommended']}. Likely causes:")
        print("           (a) the Drive copy is partial (only hw_dataset or new_dataset);")
        print("           (b) filename suffixes differ from what this script strips;")
        print("           (c) files were renamed/flattened when copied to Drive.")
        print("         Inspect the per-file table in the exported CSV to confirm.")

    # ------------------------------------------------------------------
    section("[3] UCI — TEST-ID CENSUS / THỐNG KÊ TEST-ID")
    files_per_test: Counter = Counter()
    trials_per_test: Counter = Counter()
    samples_per_test: Counter = Counter()
    subjects_per_test: Dict[int, set] = defaultdict(set)
    durations_per_test: Dict[int, List[float]] = defaultdict(list)
    unknown = 0

    for f in ok_files:
        if not f.test_ids:
            unknown += 1
            continue
        for tid in f.test_ids:
            files_per_test[tid] += 1
            subjects_per_test[tid].add(f.subject_norm)
        for tid, count in f.test_runs.items():
            trials_per_test[tid] += count
        for tid, count in f.test_ids.items():
            samples_per_test[tid] += count
            if f.duration_s is not None:
                durations_per_test[tid].append(f.duration_s)

    header = f"  {'ID':<4}{'Test':<52}{'files':>8}{'trials':>9}{'samples':>12}{'subj':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for tid in sorted(TEST_ID_LABELS):
        print(f"  {tid:<4}{TEST_ID_LABELS[tid]:<52}"
              f"{fmt_num(files_per_test.get(tid, 0)):>8}"
              f"{fmt_num(trials_per_test.get(tid, 0)):>9}"
              f"{fmt_num(samples_per_test.get(tid, 0)):>12}"
              f"{fmt_num(len(subjects_per_test.get(tid, set()))):>7}")
    other = sorted(t for t in files_per_test if t not in TEST_ID_LABELS)
    for tid in other:
        print(f"  {tid:<4}{'(unexpected Test-ID)':<52}"
              f"{fmt_num(files_per_test[tid]):>8}{fmt_num(trials_per_test.get(tid, 0)):>9}"
              f"{fmt_num(samples_per_test.get(tid, 0)):>12}"
              f"{fmt_num(len(subjects_per_test.get(tid, set()))):>7}")
    if unknown:
        print(f"\n  [WARN] {fmt_num(unknown)} file(s) with NO usable Test-ID "
              f"(no column, no filename hint).")

    print("\n  Legend / Giải thích:")
    print("    files   = how many .txt files contain that Test-ID")
    print("    trials  = contiguous Test-ID blocks = number of RECORDINGS")
    print("    samples = raw pen samples (rows) attributed to that Test-ID")
    print("    subj    = unique subjects who performed that test")

    if durations_per_test:
        subsection("Median recording duration per test / Thời lượng trung vị mỗi bài test")
        for tid in sorted(durations_per_test):
            values = durations_per_test[tid]
            print(f"      Test ID {tid} ({TEST_ID_LABELS.get(tid, '?')}): "
                  f"median {statistics.median(values):,.1f} s  "
                  f"(n={fmt_num(len(values))})")

    if uci["errors"]:
        subsection(f"UCI file errors / Lỗi file UCI (showing {min(show_errors, len(uci['errors']))})")
        for path, err in uci["errors"][:show_errors]:
            print(f"      ! {path}\n          {err}")


def print_json_report(js: Dict[str, Any], show_errors: int) -> Dict[str, Any]:
    section("[4] JSON ROOT — .json SCAN / QUÉT .json")

    kv("JSON root", str(js["root"]))
    kv("root exists", "YES" if js["exists"] else "NO  <-- CHECK YOUR PATH")
    if not js["exists"]:
        for path, err in js["errors"][:show_errors]:
            print(f"      ! {path}: {err}")
        print("\n  >> Nothing to audit under the JSON root.")
        return {"n_participants": 0, "verdict": "no data", "tier": "none",
                "n_drawings": 0, "participants": Counter()}

    kv("files discovered", fmt_num(js["n_files"]))
    kv("files parsed OK", f"{fmt_num(js['n_ok'])}  ({pct(js['n_ok'], js['n_files'])})")
    kv("files FAILED", f"{fmt_num(js['n_failed'])}  ({pct(js['n_failed'], js['n_files'])})")
    kv("drawing records", fmt_num(js["n_records"]))
    if js["skipped"]:
        kv("files skipped by --limit", fmt_num(js["skipped"]))

    n_unique_hashes = len(js["hashes"])
    n_dupe_files = js["n_ok"] - n_unique_hashes
    kv("unique file contents", fmt_num(n_unique_hashes))
    if n_dupe_files > 0:
        kv("EXACT duplicate files",
           f"{fmt_num(n_dupe_files)}  <-- same bytes, inflates the participant count")
    if js["points"]:
        kv("points per drawing (median)",
           f"{statistics.median(js['points']):,.0f}  "
           f"(min {min(js['points']):,}, max {max(js['points']):,})")
    kv("records with timestamps",
       f"{fmt_num(js['n_with_time'])} / {fmt_num(js['n_ok'])}")
    if js.get("n_recovered"):
        kv("files recovered (JSON-Lines)",
           f"{fmt_num(js['n_recovered'])}  "
           f"<-- whole-file parse failed, lines salvaged / cứu được từng dòng")
    if js["n_empty_drawings"]:
        kv("EMPTY drawings (0 points)", f"{fmt_num(js['n_empty_drawings'])}  <-- unusable")

    if js["key_freq"]:
        subsection("Most common top-level keys / Khoá cấp 1 thường gặp nhất")
        for key, count in js["key_freq"].most_common(12):
            print(f"      {key:<32} {fmt_num(count):>8} file(s)")

    if js["tasks"]:
        subsection("Task / test types found inside the JSONs / Loại bài test trong JSON")
        for task, count in js["tasks"].most_common(10):
            print(f"      {str(task):<32} {fmt_num(count):>8}")
    if js["labels"]:
        subsection("Class labels found inside the JSONs / Nhãn lớp trong JSON")
        for label, count in js["labels"].most_common(10):
            print(f"      {str(label):<32} {fmt_num(count):>8}")

    # ------------------------------------------------------------------
    section("[5] JSON — DISTINCT PARTICIPANTS OR ONE PERSON? / NHIỀU NGƯỜI HAY MỘT NGƯỜI?")
    participants: Counter = js["participants"]

    subsection("Identity fields found in the JSONs / Trường định danh tìm thấy")
    if js.get("candidate_keys"):
        print(f"      {'field':<34}{'tier':<10}{'files':>8}{'distinct values':>18}")
        print(f"      {'-' * 34}{'-' * 10}{'-' * 8}{'-' * 18}")
        for key, count in js["candidate_keys"].most_common(10):
            print(f"      {str(key)[:33]:<34}{js['candidate_tiers'].get(key, '?'):<10}"
                  f"{fmt_num(count):>8}"
                  f"{fmt_num(js['candidate_values'].get(key, 0)):>18}")
        print("\n      (a field with ONE distinct value per drawing is a drawing id,")
        print("       not a person — see the note below / xem chú thích bên dưới)")
        for key, files_n in js["candidate_keys"].most_common(10):
            if key == js.get("identity_key"):
                continue
            distinct = js.get("candidate_values", {}).get(key, 0)
            occurrences = js.get("candidate_occurrences", {}).get(key, 0)
            if occurrences > 1 and distinct == occurrences:
                print(f"\n      REJECTED '{key}': {distinct} distinct values on "
                      f"{fmt_num(occurrences)} drawings — one per drawing, so it")
                print("      identifies the DRAWING, not the person. / BỊ TỪ CHỐI vì")
                print("      mỗi bản vẽ một giá trị: đó là mã bản vẽ, không phải người.")
    else:
        print("      (no identity-shaped field found anywhere / "
              "không tìm thấy trường định danh nào)")

    if js.get("identity_note"):
        print()
        for line in _wrap_text(js["identity_note"], indent=6):
            print(line)

    subsection("How identity was established / Cách xác định danh tính")
    for tier, count in js["participant_tiers"].most_common():
        meaning = {
            "strong": "real participant/subject/user id INSIDE the JSON  (trustworthy)",
            "proxy": "device/session id used as a stand-in for the person (approximate)",
            "weak": "'name'/'code'-style field — could be a person OR a drawing",
            "filename": "no id inside the file -> guessed from the path  (UNRELIABLE)",
            "none": "no identity signal at all  (cannot de-duplicate)",
        }.get(tier, "")
        print(f"      {tier:<10} {fmt_num(count):>8} file(s)   {meaning}")
    if js["identity_keys"]:
        print("\n      Identity fields used / Trường định danh được dùng:")
        for key, count in js["identity_keys"].most_common(8):
            print(f"        {key:<44} {fmt_num(count):>6} file(s)")

    # Did every file carry a *unique* drawing-level id? That is the classic
    # trap: 40 files each with a unique "id" look like 40 people.
    drawing_ids = js["drawing_id_uniqueness"]
    if drawing_ids:
        unique_drawing_ids = len(drawing_ids)
        print(f"\n      Drawing-level ids (id / drawingId / sessionId / recordId...): "
              f"{fmt_num(unique_drawing_ids)} unique")
        print("      ^^ These identify the DRAWING, not the person. They are NOT")
        print("         counted as participants. / Đây là mã bản vẽ, KHÔNG tính là người.")

    n_participants = len(participants)
    n_drawings = js["n_records"] or js["n_ok"]
    n_unidentified = js.get("n_unidentified_records", 0)
    display = js.get("participant_display", {})

    def show(pid: str) -> str:
        """Prefer the spelling actually used in the files / ưu tiên cách viết gốc."""
        if pid in display and display[pid]:
            return display[pid].most_common(1)[0][0]
        return pid

    subsection("RESULT / KẾT QUẢ")
    kv("drawings (records)", fmt_num(n_drawings))
    kv("UNIQUE participants", fmt_num(n_participants))
    if n_unidentified:
        kv("drawings with NO identity",
           f"{fmt_num(n_unidentified)}  "
           f"<-- excluded from the participant count / không tính là người")

    identified = sum(participants.values())
    if n_participants:
        counts = sorted(participants.values(), reverse=True)
        kv("drawings per participant", f"min {min(counts)} · median "
                                       f"{statistics.median(counts):,.1f} · max {max(counts)}")
        kv("mean drawings / participant", f"{identified / n_participants:,.1f}")
        subsection("Top participants by drawing count / Người có nhiều bản vẽ nhất")
        for pid, count in participants.most_common(10):
            print(f"      {show(pid)[:44]:<46} {fmt_num(count):>6} drawing(s)")
        if n_participants > 10:
            print(f"      ... and {fmt_num(n_participants - 10)} more participant(s)")

    # ---- VERDICT ----
    tier = max(js["participant_tiers"], key=js["participant_tiers"].get) if js["participant_tiers"] else "none"
    if n_participants == 0:
        verdict = ("UNKNOWN — no participant identifier exists anywhere in the JSONs. "
                   "You CANNOT claim N participants; you have N drawings of an unknown "
                   "number of people. Add a participantId to the capture payload.")
    elif n_participants == 1:
        verdict = ("SINGLE PARTICIPANT — every identified drawing comes from ONE person. "
                   "This is repeated-measures data from 1 subject, NOT a cohort.")
    else:
        verdict = (f"DISTINCT PARTICIPANTS — {n_participants} people produced "
                   f"{identified} identified drawings "
                   f"({identified / max(1, n_participants):.1f} each)"
                   + (f", plus {n_unidentified} unidentified drawing(s)."
                      if n_unidentified else "."))

    print()
    print(rule("-"))
    verdict_lines = _wrap_text(f"VERDICT: {verdict}", width=84, indent=2)
    for line in verdict_lines:
        print(line)
    print(rule("-"))
    if n_unidentified:
        print(f"  [CAUTION] {fmt_num(n_unidentified)} drawing(s) carry no identity at all: "
              f"the real participant")
        print("            count is somewhere between "
              f"{fmt_num(max(1, n_participants))} and "
              f"{fmt_num(n_participants + n_unidentified)}.")
    if tier in ("filename", "weak"):
        print("  [CAUTION] Identity came from filenames / weak fields. Treat the")
        print("            participant count as an estimate, not a measurement.")
    if n_participants == 1:
        print("  [CAUTION] With 1 participant there is ZERO inter-subject variance:")
        print("            any train/test split leaks the same hand on both sides, and")
        print("            the resulting accuracy is meaningless. Collect more people")
        print("            before training. / Cần thu thêm người trước khi huấn luyện.")
    elif n_participants > 1:
        print("  [CAUTION] Split by PARTICIPANT (GroupKFold / GroupShuffleSplit), never")
        print("            by drawing — otherwise the same person appears in train and")
        print("            test and the score is inflated.")

    if js["errors"]:
        subsection(f"JSON file errors / Lỗi file JSON (showing {min(show_errors, len(js['errors']))})")
        for path, err in js["errors"][:show_errors]:
            print(f"      ! {path}\n          {err}")

    return {
        "n_participants": n_participants,
        "n_drawings": n_drawings,
        "n_unidentified": n_unidentified,
        "verdict": verdict,
        "tier": tier,
        "participants": participants,
    }


def print_total_report(uci: Dict[str, Any], uci_subjects: Dict[str, Any],
                       json_info: Dict[str, Any]) -> None:
    section("[6] TOTAL UNIQUE PARTICIPANTS ACROSS ALL CLASSES / TỔNG SỐ NGƯỜI DUY NHẤT")

    uci_groups = uci_subjects.get("groups", {})
    n_uci = uci_subjects.get("recommended", 0)
    n_uci_pd = uci_groups.get("parkinson", 0)
    n_uci_hc = uci_groups.get("control", 0)
    n_uci_unknown = uci_groups.get("unknown", 0)
    n_json = json_info.get("n_participants", 0)

    # Namespaced ids: UCI subject "1" and JSON participant "1" are NOT the same
    # human being. / Gắn namespace để không trộn nhầm 2 người khác nhau.
    uci_ids = {f"uci::{code}" for code in uci_subjects.get("_codes", set())}
    json_ids = {f"json::{pid}" for pid in json_info.get("participants", {})}
    total = len(uci_ids | json_ids)
    naive_total = n_uci + n_json
    collisions = naive_total - total

    print(f"  {'source':<44}{'class':<12}{'people':>10}{'drawings':>12}")
    print("  " + "-" * 78)
    uci_label = ("UCI-395 (external_test_data/uci_395_raw)"
                 if uci.get("exists") else "UCI-395  [ROOT NOT FOUND]")
    if not uci.get("exists"):
        print(f"  {uci_label:<44}{'--':<12}{'0':>10}{'--':>12}   "
              f"<-- not scanned, excluded / chưa quét, không tính")
    else:
        print(f"  {uci_label:<44}{'PARKINSON':<12}"
              f"{fmt_num(n_uci_pd):>10}{'n/a*':>12}")
        print(f"  {uci_label:<44}{'HEALTHY':<12}"
              f"{fmt_num(n_uci_hc):>10}{'n/a*':>12}")
        if n_uci_unknown:
            print(f"  {uci_label:<44}{'UNKNOWN':<12}"
                  f"{fmt_num(n_uci_unknown):>10}{'n/a*':>12}")
    print(f"  {'NeuroTrace app captures (healthy/)':<44}{'HEALTHY':<12}"
          f"{fmt_num(n_json):>10}{fmt_num(json_info.get('n_drawings', 0)):>12}")
    print("  " + "-" * 78)
    print(f"  {'TOTAL UNIQUE PARTICIPANTS':<44}{'':<12}{fmt_num(total):>10}")
    print()
    print("  * per-class drawing counts are the Test-ID census in section [3].")

    plural = "PARTICIPANT" if total == 1 else "PARTICIPANTS"
    print()
    print(rule("="))
    print(f"  >>> WE HAVE {fmt_num(total)} UNIQUE {plural} IN TOTAL "
          f"ACROSS ALL CLASSES.")
    print(f"      Parkinson's : {fmt_num(n_uci_pd)}")
    print(f"      Healthy     : {fmt_num(n_uci_hc + n_json)}"
          f"   (UCI controls {fmt_num(n_uci_hc)} + app captures {fmt_num(n_json)})")
    if n_uci_unknown:
        print(f"      Unlabelled  : {fmt_num(n_uci_unknown)}"
              f"   <-- folder name did not say control/parkinson")
    if json_info.get("n_unidentified"):
        print(f"\n  [NOTE] {fmt_num(json_info['n_unidentified'])} app drawing(s) have no "
              f"participant id and are NOT counted above;")
        print("         the true healthy total is between "
              f"{fmt_num(n_uci_hc + n_json)} and "
              f"{fmt_num(n_uci_hc + n_json + json_info['n_unidentified'])}.")
    print(rule("="))

    if collisions:
        print(f"\n  [INFO] {fmt_num(collisions)} id collision(s) between the two sources "
              f"(e.g. UCI subject '1' vs JSON participant '1').")
        print("         Namespaced counting keeps them apart; the naive sum would have")
        print(f"         reported {fmt_num(naive_total)} instead of {fmt_num(total)}.")

    if n_uci_pd and (n_uci_hc + n_json):
        ratio = n_uci_pd / max(1, (n_uci_hc + n_json))
        print(f"\n  Class balance / Cân bằng lớp: PD : Healthy = {ratio:,.2f} : 1")
        if ratio > 2:
            print("  [WARN] Heavy imbalance — use class weights / stratified group splits.")


def print_recommendations(uci: Dict[str, Any], uci_subjects: Dict[str, Any],
                          js: Dict[str, Any], json_info: Dict[str, Any]) -> None:
    section("[7] WARNINGS & NEXT ACTIONS / CẢNH BÁO & VIỆC CẦN LÀM")
    actions: List[str] = []

    if not uci["exists"]:
        actions.append("UCI root not found — verify the Drive path / kiểm tra lại đường dẫn.")
    if not js["exists"]:
        actions.append("JSON root not found — verify the Drive path / kiểm tra lại đường dẫn.")

    if uci["n_failed"]:
        actions.append(f"{uci['n_failed']} UCI file(s) failed to parse — inspect them "
                       f"(corrupt upload?) before trusting any statistic.")
    if js["n_failed"]:
        actions.append(f"{js['n_failed']} JSON file(s) failed to parse — inspect them "
                       f"(truncated upload / pretty-printed JSON?).")

    n_json_part = json_info.get("n_participants", 0)
    tier = json_info.get("tier", "none")
    if js["exists"] and js["n_ok"]:
        if n_json_part == 0:
            actions.append("JSONs carry NO participant id. Add `participantId` (or "
                           "`userId`) to the capture payload in the React app, then "
                           "re-export — otherwise the healthy class cannot be "
                           "de-duplicated or split safely.")
        elif n_json_part == 1:
            actions.append("The healthy JSONs are ONE person. Recruit more volunteers "
                           "before any supervised training; a single subject cannot "
                           "support a screening model.")
        if tier in ("filename", "weak"):
            actions.append("JSON participant identity is heuristic (filename/weak "
                           "field). Confirm by hand or fix the export schema.")
    n_dupe = js["n_ok"] - len(js["hashes"]) if js["exists"] else 0
    if n_dupe > 0:
        actions.append(f"{n_dupe} byte-identical JSON file(s) — de-duplicate before "
                       f"training or they act as free leakage between splits.")
    if js.get("n_empty_drawings"):
        actions.append(f"{js['n_empty_drawings']} JSON drawing(s) contain 0 points — "
                       f"drop them.")

    expected = UCI_EXPECTED["total"]
    if uci["exists"] and uci_subjects["recommended"] and expected and \
            uci_subjects["recommended"] != expected:
        actions.append(f"UCI subject count is {uci_subjects['recommended']}, but UCI-395 "
                       f"documents {expected} (62 PD + 15 controls). Confirm the Drive "
                       f"copy is complete (both hw_dataset/ and new_dataset/).")
    if uci_subjects.get("ambiguous"):
        actions.append(f"{len(uci_subjects['ambiguous'])} subject code(s) sit in both the "
                       f"control and parkinson folders — resolve their label by hand.")

    actions.append("Split train/val/test by PARTICIPANT (GroupKFold), never by drawing "
                   "or by row — otherwise the same hand is on both sides.")
    actions.append("Keep this audit's CSV/JSON report next to the data; re-run it after "
                   "every new data drop.")

    if not actions:
        print("      (nothing to flag — the data looks clean)")
    for i, action in enumerate(actions, start=1):
        print(f"      {i:>2}. {action}")


# ===========================================================================
# Optional machine-readable export / Xuất báo cáo (tuỳ chọn)
# ===========================================================================
def export_reports(out_dir: Path, uci: Dict[str, Any], uci_subjects: Dict[str, Any],
                   js: Dict[str, Any], json_info: Dict[str, Any]) -> List[Path]:
    """Write CSV/JSON summaries into a NEW folder. Never touches the data roots."""
    written: List[Path] = []
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # --- per-file UCI table ---
        uci_csv = out_dir / f"uci_files_{stamp}.csv"
        with open(uci_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["file", "dataset", "subject_raw", "subject_norm", "group",
                             "group_source", "rows", "delimiter", "encoding",
                             "test_ids", "trials", "duration_s", "error"])
            for f in uci["files"]:
                writer.writerow([
                    f.rel, f.dataset, f.subject_raw, f.subject_norm, f.group,
                    f.group_source, f.n_rows, f.delimiter, f.encoding,
                    ";".join(f"{k}:{v}" for k, v in sorted(f.test_ids.items())),
                    ";".join(f"{k}:{v}" for k, v in sorted(f.test_runs.items())),
                    "" if f.duration_s is None else f"{f.duration_s:.3f}",
                    f.error or "",
                ])
        written.append(uci_csv)

        # --- per-file JSON table ---
        json_csv = out_dir / f"json_files_{stamp}.csv"
        with open(json_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["file", "records", "identity_tier", "identity_key",
                             "participant_ids", "drawing_ids", "tasks", "points",
                             "has_time", "error"])
            for f in js["files"]:
                writer.writerow([
                    f.rel, f.n_records, f.identity_tier, f.identity_key,
                    ";".join(f.participants), ";".join(f.drawing_ids[:5]),
                    ";".join(f.tasks), sum(f.points), f.has_time, f.error or "",
                ])
        written.append(json_csv)

        # --- headline summary ---
        summary = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "uci_root": str(uci["root"]),
            "json_root": str(js["root"]),
            "uci": {
                "n_files": uci["n_files"],
                "n_ok": uci["n_ok"],
                "n_failed": uci["n_failed"],
                "n_samples": uci["n_samples"],
                "unique_subjects_recommended": uci_subjects["recommended"],
                "strategy": uci_subjects["recommended_key"],
                "subjects_by_class": uci_subjects["groups"],
            },
            "json": {
                "n_files": js["n_files"],
                "n_ok": js["n_ok"],
                "n_failed": js["n_failed"],
                "n_records": js["n_records"],
                "unique_participants": json_info.get("n_participants", 0),
                "identity_tier": json_info.get("tier", "none"),
                "verdict": json_info.get("verdict", ""),
                "participants": dict(json_info.get("participants", {})),
            },
        }
        summary_json = out_dir / f"audit_summary_{stamp}.json"
        with open(summary_json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        written.append(summary_json)
    except Exception as exc:  # noqa: BLE001 - exporting is a nice-to-have
        print(f"\n[export][WARN] could not write reports / không ghi được báo cáo: {exc}")
    return written


# ===========================================================================
# CLI / Giao diện dòng lệnh
# ===========================================================================
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="READ-ONLY audit of the real NeuroTrace datasets "
                    "(no training) / Kiểm toán CHỈ ĐỌC dữ liệu thật (không huấn luyện).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT),
                        help="project root on Drive / thư mục gốc trên Drive")
    parser.add_argument("--uci-root", default=None,
                        help=f"override UCI .txt root (default: <project-root>/{UCI_REL.as_posix()})")
    parser.add_argument("--json-root", default=None,
                        help=f"override JSON root (default: <project-root>/{JSON_REL.as_posix()})")
    parser.add_argument("--out-dir", default=None,
                        help=f"where audit reports are written (default: <project-root>/{REPORT_REL})")
    parser.add_argument("--no-export", action="store_true",
                        help="print only, write no files / chỉ in, không ghi file")
    parser.add_argument("--no-mount", action="store_true",
                        help="skip Google Drive mounting (for local runs)")
    parser.add_argument("--limit", type=int, default=None,
                        help="audit at most N files per source (smoke test)")
    parser.add_argument("--show-errors", type=int, default=10,
                        help="how many failed files to list")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress per-file progress lines")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    # Roots derive from --project-root unless explicitly overridden, so moving
    # the Drive folder only requires changing one flag.
    # Các thư mục suy ra từ --project-root trừ khi bị ghi đè rõ ràng.
    project_root = Path(args.project_root)
    uci_root = Path(args.uci_root) if args.uci_root else (project_root / UCI_REL)
    json_root = Path(args.json_root) if args.json_root else (project_root / JSON_REL)
    out_dir = Path(args.out_dir) if args.out_dir else (project_root / REPORT_REL)

    print(rule("="))
    print(" NEUROTRACE · REAL-DATA AUDIT (read-only · no training)")
    print(" KIỂM TOÁN DỮ LIỆU THẬT — chỉ đọc, KHÔNG huấn luyện")
    print(rule("="))
    print(f" run at        : {datetime.now().isoformat(timespec='seconds')}")
    print(f" python        : {sys.version.split()[0]}   pandas: {'yes' if HAS_PANDAS else 'no (stdlib csv)'}")
    print(f" project root  : {project_root}")
    print(f" UCI root      : {uci_root}")
    print(f" JSON root     : {json_root}")

    if not args.no_mount:
        mount_google_drive()

    # ---------------------------------------------------------------- UCI
    print("\n  Scanning UCI .txt files recursively ...")
    uci = audit_uci_root(uci_root, limit=args.limit, quiet=args.quiet)
    uci_subjects = summarise_uci_subjects(uci)
    uci_subjects["_codes"] = {
        (f.infile_subject or f.subject_norm) for f in uci["files"] if f.ok
    }
    print_uci_report(uci, uci_subjects, args.show_errors)

    # --------------------------------------------------------------- JSON
    print("\n  Scanning .json files recursively ...")
    js = audit_json_root(json_root, limit=args.limit, quiet=args.quiet)
    json_info = print_json_report(js, args.show_errors)

    # ------------------------------------------------------------- TOTALS
    print_total_report(uci, uci_subjects, json_info)

    # ------------------------------------------------------ WARNINGS/NEXT
    print_recommendations(uci, uci_subjects, js, json_info)

    # ------------------------------------------------------------- EXPORT
    if not args.no_export and (uci["exists"] or js["exists"]):
        try:
            written = export_reports(out_dir, uci, uci_subjects, js, json_info)
            if written:
                subsection("Audit reports written / Báo cáo đã ghi")
                for path in written:
                    print(f"      {path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[export][WARN] {exc}")

    print("\n" + rule("="))
    print(" AUDIT COMPLETE — no model was trained, no data file was modified.")
    print(" KIỂM TOÁN HOÀN TẤT — không huấn luyện, không sửa dữ liệu.")
    print(rule("=") + "\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:  # pragma: no cover
        print("\n[INTERRUPTED] audit cancelled by user / đã huỷ.")
        sys.exit(130)
    except Exception:  # pragma: no cover - last-resort net
        print("\n[FATAL] unexpected error / lỗi không mong đợi:")
        traceback.print_exc()
        sys.exit(1)
