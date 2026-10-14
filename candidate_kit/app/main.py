"""
Employee Attendance & Analytics API

Run:  uvicorn app.main:app --port 8000
Env:  MONGO_URI, MONGO_DB (a local .env is loaded for convenience; real environment variables win)

Conventions used throughout this file
  * MongoDB stores instants as BSON datetimes (UTC). The client is created with tz_aware=True so every
    datetime that comes back is timezone-aware. The API speaks epoch milliseconds (to_ms / from_ms).
  * Calendar dates ("YYYY-MM-DD", IST) and shift times ("HH:MM", IST) are plain strings.
  * Derived values (work_hours, late_minutes, overtime_minutes, half_day) are computed here, stored, and
    analytics read the stored values.
"""
import calendar
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from typing import Annotated, Any, Literal, Optional
from dotenv import load_dotenv
from pymongo import MongoClient

from bson import json_util
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, Field, field_validator, model_validator
from pymongo import MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

load_dotenv()  # does not override variables that are already set in the environment

log = logging.getLogger("attendance")

client = MongoClient(
    os.getenv("MONGO_URI", "mongodb://localhost:27017"),
    tz_aware=True,
    serverSelectionTimeoutMS=5000,
)
db = client[os.getenv("MONGO_DB", "attendance_db")]

IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

PRESENCE = ["PRESENT", "WFH", "ON_DUTY"]  # R6
Status = Literal["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"]
PresenceStatus = Literal["PRESENT", "WFH", "ON_DUTY"]

HHMM = r"^([01][0-9]|2[0-3]):[0-5][0-9]$"
MONTH = r"^[0-9]{4}-(0[1-9]|1[0-2])$"
DATE_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")

# An instant in the API: an integer number of epoch milliseconds. strict=True rejects floats, strings
# and booleans; the lower bound rejects values that look like seconds.
EpochMs = Annotated[int, Field(strict=True, ge=100_000_000_000, le=4_102_444_800_000)]

MAX_TREND_DAYS = 92


# --------------------------------------------------------------------------- #
# Indexes (created at startup, idempotent)
# --------------------------------------------------------------------------- #
def ensure_indexes() -> None:
    # employees
    db.employees.create_index("emp_code", unique=True, name="uniq_emp_code")  # identity + 409 on duplicates
    db.employees.create_index([("department", 1), ("emp_code", 1)], name="dept_emp_code")  # list/filter, trend
    db.employees.create_index([("joined_on", 1), ("department", 1)], name="joined_dept")  # headcount (R9)
    # attendance_logs
    db.attendance_logs.create_index(
        [("emp_code", 1), ("date", 1)], unique=True, name="uniq_emp_date"
    )  # one record per employee+day (race safety), monthly, $lookup
    db.attendance_logs.create_index([("date", -1), ("emp_code", 1)], name="date_emp")  # list order, month scans
    db.attendance_logs.create_index(
        [("status", 1), ("date", -1), ("emp_code", 1)], name="status_date_emp"
    )  # list filtered by status
    db.attendance_logs.create_index([("emp_code", 1), ("punch_in", -1)], name="emp_punch_in")  # punch-out lookup


_indexes_ready = False


def _try_indexes() -> None:
    global _indexes_ready
    try:
        ensure_indexes()
        _indexes_ready = True
    except PyMongoError as exc:  # MongoDB not reachable yet; /health will retry
        log.warning("could not create indexes yet: %s", exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _try_indexes()
    yield


app = FastAPI(title="Employee Attendance & Analytics API", version="2.0.0", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Time helpers
# --------------------------------------------------------------------------- #
def from_ms(ms: int) -> datetime:
    """Epoch milliseconds -> aware UTC datetime, truncated to whole seconds (R1)."""
    return EPOCH + timedelta(milliseconds=(ms // 1000) * 1000)


def to_ms(dt: Optional[datetime]) -> Optional[int]:
    """Datetime from MongoDB -> epoch milliseconds (naive values are treated as UTC)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return (dt - EPOCH) // timedelta(milliseconds=1)


def now_utc() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def parse_date(value: str, name: str = "date") -> date:
    if not DATE_RE.fullmatch(value):
        raise HTTPException(422, f"{name} must be YYYY-MM-DD")
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise HTTPException(422, f"{name} is not a valid calendar date")


def month_bounds(month: str) -> tuple:
    """'YYYY-MM' -> (first day, last day) as date objects."""
    year, mon = int(month[:4]), int(month[5:7])
    try:
        first = date(year, mon, 1)
    except ValueError:
        raise HTTPException(422, "month is not valid")
    return first, date(year, mon, calendar.monthrange(year, mon)[1])


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def is_overnight(emp: dict) -> bool:
    return _minutes(emp["shift_end"]) <= _minutes(emp["shift_start"])


def ist_at(d: date, hhmm: str) -> datetime:
    h, m = hhmm.split(":")
    return datetime(d.year, d.month, d.day, int(h), int(m), tzinfo=IST)


def attendance_date_for(emp: dict, punch_in: datetime) -> date:
    """R1: IST calendar date of the punch-in; an overnight shift's early-morning punch-in belongs to the
    shift that started the previous day."""
    ist = punch_in.astimezone(IST)
    d = ist.date()
    if is_overnight(emp):
        seconds_of_day = ist.hour * 3600 + ist.minute * 60 + ist.second
        if seconds_of_day < _minutes(emp["shift_end"]) * 60:
            d -= timedelta(days=1)
    return d


def shift_end_dt(emp: dict, d: date) -> datetime:
    end = ist_at(d, emp["shift_end"])
    return end + timedelta(days=1) if is_overnight(emp) else end


def half_up_div(n: int, d: int) -> int:
    """round(n / d) with ties going up, for integers n >= 0, d > 0 (no float involved)."""
    return (2 * n + d) // (2 * d)


# --------------------------------------------------------------------------- #
# Business rules R2 - R5
# --------------------------------------------------------------------------- #
def compute_late_minutes(punch_in: datetime, shift_start: str, att_date: date) -> int:
    """R2: late only if strictly more than 10:00 after shift_start; then whole minutes since shift_start."""
    delta = int((punch_in - ist_at(att_date, shift_start)).total_seconds())
    return delta // 60 if delta > 600 else 0


def compute_work_hours(punch_in: datetime, punch_out: datetime) -> float:
    """R4: seconds / 3600 rounded to 2 decimals half-up, using integer arithmetic."""
    seconds = int((punch_out - punch_in).total_seconds())
    return half_up_div(seconds, 36) / 100  # seconds/3600 in hundredths == seconds/36


def compute_overtime(punch_out: datetime, emp: dict, att_date: date) -> int:
    """R3: whole minutes after shift_end, counted only if at least 30."""
    minutes = int((punch_out - shift_end_dt(emp, att_date)).total_seconds()) // 60
    return minutes if minutes >= 30 else 0


def derive(emp: dict, att_date: date, status: str, punch_in: Optional[datetime], punch_out: Optional[datetime]) -> dict:
    """All stored derived values for a record in its final state."""
    out = {"work_hours": None, "late_minutes": 0, "overtime_minutes": 0, "half_day": False}
    if status not in PRESENCE or punch_in is None:
        return out
    out["late_minutes"] = compute_late_minutes(punch_in, emp["shift_start"], att_date)
    if punch_out is not None:
        wh = compute_work_hours(punch_in, punch_out)
        out["work_hours"] = wh
        out["overtime_minutes"] = compute_overtime(punch_out, emp, att_date)
        out["half_day"] = round(wh * 100) < 450  # R5, on the rounded value
    return out


# --------------------------------------------------------------------------- #
# Serialisation (MongoDB document -> API shape)
# --------------------------------------------------------------------------- #
def _api_value(v: Any) -> Any:
    return to_ms(v) if isinstance(v, datetime) else v


def history_out(h: dict) -> dict:
    return {
        "at": to_ms(h.get("at")),
        "by": h.get("by"),
        "reason": h.get("reason"),
        "changes": {
            k: {"from": _api_value(c.get("from")), "to": _api_value(c.get("to"))}
            for k, c in (h.get("changes") or {}).items()
        },
    }


def record_out(d: dict) -> dict:
    return {
        "emp_code": d["emp_code"],
        "date": d["date"],
        "status": d["status"],
        "punch_in": to_ms(d.get("punch_in")),
        "punch_out": to_ms(d.get("punch_out")),
        "work_hours": d.get("work_hours"),
        "late_minutes": d.get("late_minutes") or 0,
        "overtime_minutes": d.get("overtime_minutes") or 0,
        "half_day": bool(d.get("half_day", False)),
        "history": [history_out(h) for h in (d.get("history") or [])],
    }


def employee_out(d: dict) -> dict:
    return {
        "emp_code": d["emp_code"],
        "name": d["name"],
        "email": d["email"],
        "department": d["department"],
        "shift_start": d["shift_start"],
        "shift_end": d["shift_end"],
        "joined_on": d["joined_on"],
        "created_at": to_ms(d.get("created_at")),
    }


def get_employee(emp_code: str) -> dict:
    emp = db.employees.find_one({"emp_code": emp_code})
    if emp is None:
        raise HTTPException(404, "employee not found")
    return emp


def page_slice(page: int, page_size: int) -> int:
    skip = (page - 1) * page_size
    return skip


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class EmployeeIn(BaseModel):
    emp_code: str = Field(pattern=r"^EMP[0-9]{4,6}$")
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(min_length=1, max_length=50)
    shift_start: str = Field("09:30", pattern=HHMM)
    shift_end: str = Field("18:30", pattern=HHMM)
    joined_on: str

    @field_validator("joined_on")
    @classmethod
    def _joined_on_is_a_date(cls, v: str) -> str:
        if not DATE_RE.fullmatch(v):
            raise ValueError("joined_on must be YYYY-MM-DD")
        date.fromisoformat(v)  # ValueError -> 422
        return v

    @model_validator(mode="after")
    def _shift_times_differ(self):
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start must differ from shift_end")
        return self


class PunchInIn(BaseModel):
    emp_code: str = Field(min_length=1)
    punched_at: Optional[EpochMs] = None
    status: PresenceStatus = "PRESENT"


class PunchOutIn(BaseModel):
    emp_code: str = Field(min_length=1)
    punched_at: Optional[EpochMs] = None


class RegularizeIn(BaseModel):
    # Default None on a non-Optional annotation: omitted is fine, an explicit null is a 422.
    status: Status = None  # type: ignore[assignment]
    punch_in: EpochMs = None  # type: ignore[assignment]
    punch_out: EpochMs = None  # type: ignore[assignment]
    reason: str = Field(min_length=5, max_length=200)
    regularized_by: str = Field(min_length=1, max_length=50)
    # Derived fields sent by a client are simply not part of the model, so they are ignored.


# --------------------------------------------------------------------------- #
# System + employees
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    try:
        client.admin.command("ping")
        if not _indexes_ready:
            _try_indexes()
    except PyMongoError:
        raise HTTPException(503, "database unavailable")
    return {"status": "ok"}


@app.post("/employees", status_code=201)
def create_employee(body: EmployeeIn):
    doc = body.model_dump()
    doc["created_at"] = now_utc()
    try:
        db.employees.insert_one(doc)  # the unique index on emp_code decides, not a prior find()
    except DuplicateKeyError:
        raise HTTPException(409, "emp_code already exists")
    return employee_out(doc)


@app.get("/employees")
def list_employees(
    department: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    q: dict = {}
    if department:
        q["department"] = department
    skip = page_slice(page, page_size)
    total = db.employees.count_documents(q)
    items = []
    if skip < 10**15:
        items = [
            employee_out(d)
            for d in db.employees.find(q).sort("emp_code", 1).skip(skip).limit(page_size)
        ]
    return {"items": items, "total": total, "page": page, "page_size": page_size}


# --------------------------------------------------------------------------- #
# Attendance
# --------------------------------------------------------------------------- #
@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInIn):
    emp = get_employee(body.emp_code)
    ts = from_ms(body.punched_at) if body.punched_at is not None else now_utc()
    att_date = attendance_date_for(emp, ts)
    doc = {
        "emp_code": body.emp_code,
        "date": att_date.isoformat(),
        "status": body.status,
        "punch_in": ts,
        "punch_out": None,
        "work_hours": None,
        "late_minutes": compute_late_minutes(ts, emp["shift_start"], att_date),
        "overtime_minutes": 0,
        "half_day": False,
        "history": [],
    }
    try:
        db.attendance_logs.insert_one(doc)  # unique (emp_code, date) index makes this race-safe
    except DuplicateKeyError:
        raise HTTPException(409, "already punched in for this date")
    return record_out(doc)


@app.post("/attendance/punch-out")
def punch_out(body: PunchOutIn):
    emp = get_employee(body.emp_code)
    ts = from_ms(body.punched_at) if body.punched_at is not None else now_utc()
    # Most recent record whose punch_in is at or before the punch-out (works across midnight).
    rec = db.attendance_logs.find_one(
        {"emp_code": body.emp_code, "punch_in": {"$lte": ts}}, sort=[("punch_in", -1)]
    )
    if rec is None:
        raise HTTPException(404, "no punch-in found")
    if rec.get("punch_out") is not None:
        raise HTTPException(409, "already punched out")
    p_in = rec["punch_in"].astimezone(UTC) if rec["punch_in"].tzinfo else rec["punch_in"].replace(tzinfo=UTC)
    if ts <= p_in:
        raise HTTPException(422, "punched_at must be after the punch-in")
    if ts - p_in > timedelta(hours=24):
        raise HTTPException(422, "punch-out is more than 24 hours after the punch-in")
    derived = derive(emp, date.fromisoformat(rec["date"]), rec["status"], p_in, ts)
    updated = db.attendance_logs.find_one_and_update(
        # Compare-and-set: only an open record that nobody changed in the meantime can be closed.
        {"_id": rec["_id"], "punch_out": None, "punch_in": rec["punch_in"], "status": rec["status"]},
        {"$set": {"punch_out": ts, **derived}},
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise HTTPException(409, "record was modified or already punched out")
    return record_out(updated)


def build_attendance_query(
    emp_code: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
    status: Optional[str],
    page: int,
    page_size: int,
) -> tuple:
    if date_from:
        parse_date(date_from, "date_from")
    if date_to:
        parse_date(date_to, "date_to")
    if date_from and date_to and date_from > date_to:
        raise HTTPException(422, "date_from must not be after date_to")
    q: dict = {}
    if emp_code:
        q["emp_code"] = emp_code
    rng = {}
    if date_from:
        rng["$gte"] = date_from
    if date_to:
        rng["$lte"] = date_to
    if rng:
        q["date"] = rng
    if status:
        q["status"] = status
    sort = [("date", -1), ("emp_code", 1)]
    return q, sort, page_slice(page, page_size), page_size


@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[Status] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    q, sort, skip, limit = build_attendance_query(emp_code, date_from, date_to, status, page, page_size)
    total = db.attendance_logs.count_documents(q) if q else db.attendance_logs.estimated_document_count()
    items = []
    if skip < 10**15:
        items = [record_out(d) for d in db.attendance_logs.find(q).sort(sort).skip(skip).limit(limit)]
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@app.patch("/attendance/{emp_code}/{date}")
def regularize(body: RegularizeIn, emp_code: str, rec_date: str = Path(alias="date")):
    parse_date(rec_date, "date")
    emp = get_employee(emp_code)
    rec = db.attendance_logs.find_one({"emp_code": emp_code, "date": rec_date})
    if rec is None:
        raise HTTPException(404, "no attendance record for that date")

    old = {
        "status": rec["status"],
        "punch_in": rec.get("punch_in"),
        "punch_out": rec.get("punch_out"),
        "work_hours": rec.get("work_hours"),
        "late_minutes": rec.get("late_minutes") or 0,
        "overtime_minutes": rec.get("overtime_minutes") or 0,
        "half_day": bool(rec.get("half_day", False)),
    }
    new_status = body.status or old["status"]
    new_in, new_out = old["punch_in"], old["punch_out"]

    if new_status in ("ABSENT", "LEAVE"):
        if body.punch_in is not None or body.punch_out is not None:
            raise HTTPException(422, "ABSENT/LEAVE clear the punch times; do not supply punch_in/punch_out")
        new_in = new_out = None
    else:
        if body.punch_in is not None:
            new_in = from_ms(body.punch_in)
            if attendance_date_for(emp, new_in).isoformat() != rec_date:
                raise HTTPException(422, "punch_in must stay on the record's attendance date")
        if body.punch_out is not None:
            new_out = from_ms(body.punch_out)
        if new_in is None:
            raise HTTPException(422, "a presence status requires a punch_in")
        if new_out is not None and (body.punch_in is not None or body.punch_out is not None):
            a = new_in if new_in.tzinfo else new_in.replace(tzinfo=UTC)
            if new_out <= a:
                raise HTTPException(422, "punch_out must be after punch_in")
            if new_out - a > timedelta(hours=24):
                raise HTTPException(422, "punch_out must be within 24 hours of punch_in")

    new = {"status": new_status, "punch_in": new_in, "punch_out": new_out}
    new.update(derive(emp, date.fromisoformat(rec_date), new_status, new_in, new_out))

    changes = {}
    for key in ("status", "punch_in", "punch_out", "work_hours", "late_minutes", "overtime_minutes", "half_day"):
        if _api_value(old[key]) != _api_value(new[key]):
            changes[key] = {"from": old[key], "to": new[key]}
    if not changes:
        raise HTTPException(422, "the request changes nothing")

    entry = {"at": now_utc(), "by": body.regularized_by, "reason": body.reason, "changes": changes}
    updated = db.attendance_logs.find_one_and_update(
        {
            "_id": rec["_id"],
            "status": rec["status"],
            "punch_in": rec.get("punch_in"),
            "punch_out": rec.get("punch_out"),
            # nobody appended to the history since we read the record
            "$expr": {"$eq": [{"$size": {"$ifNull": ["$history", []]}}, len(rec.get("history") or [])]},
        },
        {"$set": new, "$push": {"history": entry}},
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise HTTPException(409, "the record was modified concurrently; retry")
    return record_out(updated)


# --------------------------------------------------------------------------- #
# Analytics: pipeline builders
# --------------------------------------------------------------------------- #
def _present_value(ref):
    """Contribution of one log to present_days (R5-R7): presence status on Mon-Fri; half day = 0.5."""
    return {
        "$cond": [
            {
                "$and": [
                    {"$in": [ref("status"), PRESENCE]},
                    {"$lte": [{"$isoDayOfWeek": {"$dateFromString": {"dateString": ref("date")}}}, 5]},
                ]
            },
            {"$cond": [{"$eq": [{"$ifNull": [ref("half_day"), False]}, True]}, 0.5, 1]},
            0,
        ]
    }


def _late_flag(ref):
    return {"$cond": [{"$gt": [{"$ifNull": [ref("late_minutes"), 0]}, 0]}, 1, 0]}


def _doc_field(name):
    return "$" + name


def _log_field(name):
    return "$$l." + name


def build_monthly_pipeline(emp_code: str, start: str, end: str) -> list:
    return [
        {"$match": {"emp_code": emp_code, "date": {"$gte": start, "$lte": end}}},
        {
            "$group": {
                "_id": None,
                "present_days": {"$sum": _present_value(_doc_field)},
                "leave_days": {"$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}},
                "late_count": {"$sum": _late_flag(_doc_field)},
                "total_late_minutes": {"$sum": {"$ifNull": ["$late_minutes", 0]}},
                "total_overtime_minutes": {"$sum": {"$ifNull": ["$overtime_minutes", 0]}},
            }
        },
    ]


def build_department_summary_pipeline(start: str, end: str, department: Optional[str]) -> list:
    emp_match: dict = {"joined_on": {"$lte": end}}  # R9: joined on or before the end of the month
    if department:
        emp_match["department"] = department
    present_logs_with_hours = {
        "$filter": {
            "input": "$logs",
            "as": "l",
            "cond": {"$and": [{"$in": ["$$l.status", PRESENCE]}, {"$isNumber": "$$l.work_hours"}]},
        }
    }

    def count_status(name):
        return {"$size": {"$filter": {"input": "$logs", "as": "l", "cond": {"$eq": ["$$l.status", name]}}}}

    return [
        {"$match": emp_match},
        # Start from employees so people with zero logs still count; the lookup uses the (emp_code, date) index.
        {
            "$lookup": {
                "from": "attendance_logs",
                "localField": "emp_code",
                "foreignField": "emp_code",
                "pipeline": [
                    {"$match": {"date": {"$gte": start, "$lte": end}}},
                    {"$project": {"_id": 0, "status": 1, "date": 1, "half_day": 1, "late_minutes": 1, "work_hours": 1}},
                ],
                "as": "logs",
            }
        },
        {
            "$project": {
                "department": 1,
                "present_days": {"$sum": {"$map": {"input": "$logs", "as": "l", "in": _present_value(_log_field)}}},
                "late_count": {
                    "$size": {
                        "$filter": {
                            "input": "$logs",
                            "as": "l",
                            "cond": {"$gt": [{"$ifNull": ["$$l.late_minutes", 0]}, 0]},
                        }
                    }
                },
                "total_late_minutes": {
                    "$sum": {"$map": {"input": "$logs", "as": "l", "in": {"$ifNull": ["$$l.late_minutes", 0]}}}
                },
                "leave_count": count_status("LEAVE"),
                "on_duty_count": count_status("ON_DUTY"),
                "wh_n": {"$size": present_logs_with_hours},
                # work_hours in hundredths, so the average is computed from exact integers
                "wh_centi": {
                    "$sum": {
                        "$map": {
                            "input": present_logs_with_hours,
                            "as": "w",
                            "in": {"$round": [{"$multiply": ["$$w.work_hours", 100]}, 0]},
                        }
                    }
                },
            }
        },
        {
            "$group": {
                "_id": "$department",
                "headcount": {"$sum": 1},
                "present_days": {"$sum": "$present_days"},
                "late_count": {"$sum": "$late_count"},
                "total_late_minutes": {"$sum": "$total_late_minutes"},
                "leave_count": {"$sum": "$leave_count"},
                "on_duty_count": {"$sum": "$on_duty_count"},
                "wh_n": {"$sum": "$wh_n"},
                "wh_centi": {"$sum": "$wh_centi"},
            }
        },
        {"$sort": {"_id": 1}},
    ]


def build_leaderboard_pipeline(start: str, end: str, limit: int, department: Optional[str]) -> list:
    pipeline: list = [
        {"$match": {"date": {"$gte": start, "$lte": end}, "late_minutes": {"$gt": 0}}},
        {"$group": {"_id": "$emp_code", "total_late_minutes": {"$sum": "$late_minutes"}, "late_count": {"$sum": 1}}},
        {"$lookup": {"from": "employees", "localField": "_id", "foreignField": "emp_code", "as": "emp"}},
        {"$unwind": "$emp"},  # logs of unknown employees are dropped *before* ranking
    ]
    if department:
        pipeline.append({"$match": {"emp.department": department}})  # rank inside the department only
    pipeline += [
        # $rank = standard competition ranking (1, 2, 2, 4); the limit is applied to the rank afterwards.
        {"$setWindowFields": {"sortBy": {"total_late_minutes": -1}, "output": {"rank": {"$rank": {}}}}},
        {"$match": {"rank": {"$lte": limit}}},
        {"$sort": {"total_late_minutes": -1, "_id": 1}},
        {
            "$project": {
                "_id": 0,
                "rank": 1,
                "emp_code": "$_id",
                "name": "$emp.name",
                "department": "$emp.department",
                "total_late_minutes": 1,
                "late_count": 1,
            }
        },
    ]
    return pipeline


def build_trend_pipeline(department: str, codes: list, joins: list, d_from: date, d_to: date) -> list:
    lo = datetime(d_from.year, d_from.month, d_from.day, tzinfo=UTC)
    hi = datetime(d_to.year, d_to.month, d_to.day, tzinfo=UTC) + timedelta(days=1)  # $densify upper bound is exclusive
    rate_window = {"documents": [-6, 0]}  # current row + up to 6 preceding rows, inside the range only
    return [
        {"$match": {"emp_code": {"$in": codes}, "date": {"$gte": d_from.isoformat(), "$lte": d_to.isoformat()}}},
        # One neutral row for the first day guarantees $densify has an input even when the department has no
        # logs at all in the range. It counts as 0 present / 0 late and is built from an indexed employees lookup.
        {
            "$unionWith": {
                "coll": "employees",
                "pipeline": [
                    {"$match": {"department": department}},
                    {"$limit": 1},
                    {"$project": {"_id": 0, "date": {"$literal": d_from.isoformat()}, "status": {"$literal": "ABSENT"}}},
                ],
            }
        },
        {
            "$group": {
                "_id": "$date",
                "present_count": {"$sum": _presence_any_day(_doc_field)},
                "late_count": {"$sum": _late_flag(_doc_field)},
            }
        },
        {"$set": {"day": {"$dateFromString": {"dateString": "$_id"}}}},
        {"$densify": {"field": "day", "range": {"step": 1, "unit": "day", "bounds": [lo, hi]}}},  # gap filling
        {
            "$set": {
                "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$day"}},
                "present_count": {"$ifNull": ["$present_count", 0]},
                "late_count": {"$ifNull": ["$late_count", 0]},
                "is_working_day": {"$lte": [{"$isoDayOfWeek": "$day"}, 5]},
            }
        },
        {
            "$set": {
                "headcount": {
                    "$size": {"$filter": {"input": joins, "as": "j", "cond": {"$lte": ["$$j", "$date"]}}}
                }
            }
        },
        {
            # attendance_rate in units of 1e-4, rounded half-up with integer maths:
            # floor((2*present*10000 + headcount) / (2*headcount))
            "$set": {
                "rate4": {
                    "$cond": [
                        {"$and": ["$is_working_day", {"$gt": ["$headcount", 0]}]},
                        {
                            "$floor": {
                                "$divide": [
                                    {"$add": [{"$multiply": [{"$multiply": ["$present_count", 2]}, 10000]}, "$headcount"]},
                                    {"$multiply": [2, "$headcount"]},
                                ]
                            }
                        },
                        None,
                    ]
                }
            }
        },
        {"$set": {"has_rate": {"$cond": [{"$eq": ["$rate4", None]}, 0, 1]}}},
        {"$sort": {"day": 1}},
        {
            "$setWindowFields": {
                "sortBy": {"day": 1},
                "output": {
                    "win_sum": {"$sum": "$rate4", "window": rate_window},
                    "win_cnt": {"$sum": "$has_rate", "window": rate_window},
                },
            }
        },
        {
            "$project": {
                "_id": 0,
                "date": 1,
                "is_working_day": 1,
                "headcount": 1,
                "present_count": 1,
                "late_count": 1,
                "attendance_rate": {"$cond": [{"$eq": ["$rate4", None]}, None, {"$divide": ["$rate4", 10000]}]},
                "moving_avg_7d": {
                    "$cond": [
                        {"$eq": ["$win_cnt", 0]},
                        None,
                        {
                            "$divide": [
                                {
                                    "$floor": {
                                        "$divide": [
                                            {"$add": [{"$multiply": [2, "$win_sum"]}, "$win_cnt"]},
                                            {"$multiply": [2, "$win_cnt"]},
                                        ]
                                    }
                                },
                                10000,
                            ]
                        },
                    ]
                },
            }
        },
    ]


# The trend counts presence on any day of the week (weekends included), so it needs its own value expression.
def _presence_any_day(ref):
    return {
        "$cond": [
            {"$in": [ref("status"), PRESENCE]},
            {"$cond": [{"$eq": [{"$ifNull": [ref("half_day"), False]}, True]}, 0.5, 1]},
            0,
        ]
    }


# --------------------------------------------------------------------------- #
# Analytics: endpoints
# --------------------------------------------------------------------------- #
def working_days_between(first: date, last: date, joined_on: date) -> int:
    """R7: Monday-Friday days in [first, last], counted from joined_on for mid-month joiners."""
    d = max(first, joined_on)
    n = 0
    while d <= last:
        if d.weekday() < 5:
            n += 1
        d += timedelta(days=1)
    return n


@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(emp_code: str, month: str = Query(..., pattern=MONTH)):
    first, last = month_bounds(month)
    emp = get_employee(emp_code)
    rows = list(db.attendance_logs.aggregate(build_monthly_pipeline(emp_code, first.isoformat(), last.isoformat())))
    r = rows[0] if rows else {}
    working_days = working_days_between(first, last, date.fromisoformat(emp["joined_on"]))
    present = float(r.get("present_days", 0))
    pct = None
    if working_days > 0:
        pct = half_up_div(round(present * 2) * 5000, working_days) / 100  # present/working*100, 2 dp half-up
    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": working_days,
        "present_days": present,
        "leave_days": int(r.get("leave_days", 0)),
        "late_count": int(r.get("late_count", 0)),
        "total_late_minutes": int(r.get("total_late_minutes", 0)),
        "total_overtime_minutes": int(r.get("total_overtime_minutes", 0)),
        "attendance_pct": pct,
    }


@app.get("/analytics/departments/summary")
def department_summary(month: str = Query(..., pattern=MONTH), department: Optional[str] = None):
    first, last = month_bounds(month)
    rows = db.employees.aggregate(build_department_summary_pipeline(first.isoformat(), last.isoformat(), department))
    items = []
    for r in rows:
        n = int(r["wh_n"])
        items.append(
            {
                "department": r["_id"],
                "headcount": int(r["headcount"]),
                "present_days": float(r["present_days"]),
                "avg_work_hours": half_up_div(round(r["wh_centi"]), n) / 100 if n else None,
                "late_count": int(r["late_count"]),
                "total_late_minutes": int(r["total_late_minutes"]),
                "leave_count": int(r["leave_count"]),
                "on_duty_count": int(r["on_duty_count"]),
            }
        )
    return {"month": month, "items": items}


@app.get("/analytics/leaderboard/late")
def late_leaderboard(
    month: str = Query(..., pattern=MONTH),
    limit: int = Query(10, ge=1, le=50),
    department: Optional[str] = None,
):
    first, last = month_bounds(month)
    items = list(
        db.attendance_logs.aggregate(build_leaderboard_pipeline(first.isoformat(), last.isoformat(), limit, department))
    )
    return {"month": month, "items": items}


def load_trend_inputs(department: str, d_from: str, d_to: str) -> tuple:
    start, end = parse_date(d_from, "from"), parse_date(d_to, "to")
    if end < start:
        raise HTTPException(422, "to must not be before from")
    if (end - start).days + 1 > MAX_TREND_DAYS:
        raise HTTPException(422, f"range must not exceed {MAX_TREND_DAYS} days")
    emps = list(db.employees.find({"department": department}, {"_id": 0, "emp_code": 1, "joined_on": 1}))
    if not emps:
        raise HTTPException(404, "unknown department")
    codes = [e["emp_code"] for e in emps]
    joins = sorted(e["joined_on"] for e in emps)
    return start, end, codes, joins


@app.get("/analytics/departments/{department}/trend")
def department_trend(department: str, from_: str = Query(..., alias="from"), to: str = Query(...)):
    start, end, codes, joins = load_trend_inputs(department, from_, to)
    items = list(db.attendance_logs.aggregate(build_trend_pipeline(department, codes, joins, start, end)))
    return {"department": department, "items": items}


# --------------------------------------------------------------------------- #
# Admin: explain
# --------------------------------------------------------------------------- #
def _explain(command: dict) -> dict:
    out = db.command({"explain": command, "verbosity": "executionStats"})
    return json.loads(json_util.dumps(out, json_options=json_util.RELAXED_JSON_OPTIONS))


def _require(**params):
    missing = [k for k, v in params.items() if v is None]
    if missing:
        raise HTTPException(422, "missing query parameter(s): " + ", ".join(missing))


@app.get("/admin/explain/{endpoint}")
def explain_endpoint(
    endpoint: Literal["attendance_list", "employee_monthly", "department_summary", "late_leaderboard", "department_trend"],
    emp_code: Optional[str] = None,
    month: Optional[str] = Query(None, pattern=MONTH),
    department: Optional[str] = None,
    limit: int = Query(10, ge=1, le=50),
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[Status] = None,
    from_: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    if endpoint == "attendance_list":
        q, sort, skip, lim = build_attendance_query(emp_code, date_from, date_to, status, page, page_size)
        command = {"find": "attendance_logs", "filter": q, "sort": dict(sort), "skip": skip, "limit": lim}
        collection = "attendance_logs"
    elif endpoint == "employee_monthly":
        _require(emp_code=emp_code, month=month)
        first, last = month_bounds(month)
        pipeline = build_monthly_pipeline(emp_code, first.isoformat(), last.isoformat())
        collection = "attendance_logs"
        command = {"aggregate": collection, "pipeline": pipeline, "cursor": {}}
    elif endpoint == "department_summary":
        _require(month=month)
        first, last = month_bounds(month)
        pipeline = build_department_summary_pipeline(first.isoformat(), last.isoformat(), department)
        collection = "employees"
        command = {"aggregate": collection, "pipeline": pipeline, "cursor": {}}
    elif endpoint == "late_leaderboard":
        _require(month=month)
        first, last = month_bounds(month)
        pipeline = build_leaderboard_pipeline(first.isoformat(), last.isoformat(), limit, department)
        collection = "attendance_logs"
        command = {"aggregate": collection, "pipeline": pipeline, "cursor": {}}
    else:  # department_trend
        _require(department=department, **{"from": from_, "to": to})
        start, end, codes, joins = load_trend_inputs(department, from_, to)
        pipeline = build_trend_pipeline(department, codes, joins, start, end)
        collection = "attendance_logs"
        command = {"aggregate": collection, "pipeline": pipeline, "cursor": {}}
    try:
        explained = _explain(command)
    except PyMongoError as exc:
        raise HTTPException(503, f"explain failed: {exc}")
    return {"endpoint": endpoint, "collection": collection, "explain": explained}
