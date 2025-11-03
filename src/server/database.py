from __future__ import annotations

import datetime as dt
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import os

from loguru import logger
from passlib.context import CryptContext
from sqlalchemy import Column, JSON, String, UniqueConstraint
from sqlmodel import Field, Session, SQLModel, create_engine, select

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "app.db"
_ENGINE = None
_PWD_CONTEXT = CryptContext(schemes=["argon2", "bcrypt"], deprecated="auto")
DATABASE_URL_ENV = "DATABASE_URL"


def _hash_password(password: str) -> str:
    return _PWD_CONTEXT.hash(password)


def _verify_password(password: str, hashed: str) -> bool:
    try:
        return _PWD_CONTEXT.verify(password, hashed)
    except Exception:  # pragma: no cover - passlib defensive
        return False


class User(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True, index=True)
    email: str = Field(sa_column=Column(String, unique=True, index=True, nullable=False))
    display_name: str = Field(index=True)
    hashed_password: str = Field(sa_column=Column(String, nullable=False))
    avatar_url: Optional[str] = Field(default=None)
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    last_login_at: Optional[dt.datetime] = None
    level: int = Field(default=1)
    experience_points: int = Field(default=0)


class UserPreference(SQLModel, table=True):
    user_id: str = Field(foreign_key="user.id", primary_key=True)
    level: str = Field(default="beginner")
    focus_exercises: List[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False, default="[]"))
    daily_goal: int = Field(default=20)
    smart_goal: bool = Field(default=True)
    timezone: str = Field(default="UTC")
    reminder_hour: int = Field(default=9)
    allow_notifications: bool = Field(default=True)


class DailyProgress(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(foreign_key="user.id")
    date: dt.date
    exercise: str
    reps_completed: int = 0
    target_reps: int = 0
    accuracy: float = 0.0
    consistency: float = 0.0
    warnings_count: int = 0
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)


class SessionRecord(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(foreign_key="user.id")
    exercise: str
    rep_count: int
    accuracy: float
    consistency: float
    warnings_count: int
    xp_earned: int
    started_at: dt.datetime
    ended_at: dt.datetime
    session_metadata: Dict[str, object] = Field(default_factory=dict, sa_column=Column(JSON, default="{}"))


class AdaptiveRecommendation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(foreign_key="user.id")
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    suggestion: str
    category: str = Field(default="general")


class Achievement(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("user_id", "code", name="uq_user_achievement"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(foreign_key="user.id")
    code: str = Field(sa_column=Column(String, nullable=False))
    name: str
    description: str
    earned_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)


class UserReminder(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(foreign_key="user.id")
    message: str
    remind_at: dt.datetime
    completed: bool = Field(default=False)
    created_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)


def _normalize_database_url(url: str) -> str:
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql://") and "+" not in url:
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def _get_engine():
    global _ENGINE
    if _ENGINE is None:
        database_url = os.getenv(DATABASE_URL_ENV)
        if database_url:
            normalized = _normalize_database_url(database_url)
            _ENGINE = create_engine(normalized, echo=False, pool_pre_ping=True)
        else:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            connect_args = {"check_same_thread": False}
            _ENGINE = create_engine(f"sqlite:///{DB_PATH}", echo=False, connect_args=connect_args)
    return _ENGINE


def init_db() -> None:
    engine = _get_engine()
    SQLModel.metadata.create_all(engine)
    database_url = os.getenv(DATABASE_URL_ENV)
    if database_url:
        logger.info("Database initialized using {}", DATABASE_URL_ENV)
    else:
        logger.info("Database initialized at {}", DB_PATH.resolve())


@contextmanager
def session_scope() -> Iterable[Session]:
    engine = _get_engine()
    with Session(engine) as session:
        yield session


def register_user(email: str, password: str, display_name: str) -> User:
    email_normalized = email.strip().lower()
    with session_scope() as session:
        existing = session.exec(select(User).where(User.email == email_normalized)).first()
        if existing:
            raise ValueError("Email already registered")
        user = User(email=email_normalized, display_name=display_name.strip(), hashed_password=_hash_password(password))
        session.add(user)
        session.commit()
        session.refresh(user)
        prefs = UserPreference(user_id=user.id, focus_exercises=["pushup", "pullup", "squat"])
        session.add(prefs)
        session.commit()
        logger.info("Registered new user {}", user.id)
        return user


def _slugify_display_name(display_name: str) -> str:
    """Build a stable local-part for generated email addresses."""
    allowed = [ch.lower() for ch in display_name if ch.isalnum() or ch in {".", "_", "-"}]
    slug = "".join(allowed).strip(".-_")
    return slug or "user"


def create_user(display_name: str) -> User:
    """Backward compatible helper that auto-generates an email for legacy flows."""
    local_part = _slugify_display_name(display_name)
    generated_email = f"{local_part}@example.com"
    with session_scope() as session:
        user = session.exec(select(User).where(User.display_name == display_name)).first()
        if user:
            if user.email != generated_email:
                user.email = generated_email
                session.add(user)
                session.commit()
                session.refresh(user)
            return user
    return register_user(generated_email, password=str(uuid.uuid4()), display_name=display_name)


def authenticate_user(email: str, password: str) -> Optional[User]:
    email_normalized = email.strip().lower()
    with session_scope() as session:
        user = session.exec(select(User).where(User.email == email_normalized)).first()
        if not user or not _verify_password(password, user.hashed_password):
            return None
        user.last_login_at = dt.datetime.utcnow()
        session.add(user)
        session.commit()
        session.refresh(user)
        return user


def get_user(user_id: str) -> Optional[User]:
    with session_scope() as session:
        return session.get(User, user_id)


def get_user_by_email(email: str) -> Optional[User]:
    email_normalized = email.strip().lower()
    with session_scope() as session:
        return session.exec(select(User).where(User.email == email_normalized)).first()


def get_preferences(user_id: str) -> UserPreference:
    with session_scope() as session:
        prefs = session.exec(select(UserPreference).where(UserPreference.user_id == user_id)).first()
        if prefs is None:
            prefs = UserPreference(user_id=user_id, focus_exercises=["pushup", "pullup", "squat"])
            session.add(prefs)
            session.commit()
            session.refresh(prefs)
        return prefs


def update_preferences(
    user_id: str,
    *,
    level: Optional[str] = None,
    focus_exercises: Optional[Sequence[str]] = None,
    daily_goal: Optional[int] = None,
    smart_goal: Optional[bool] = None,
    timezone: Optional[str] = None,
    reminder_hour: Optional[int] = None,
    allow_notifications: Optional[bool] = None,
) -> UserPreference:
    with session_scope() as session:
        prefs = session.exec(select(UserPreference).where(UserPreference.user_id == user_id)).first()
        if prefs is None:
            prefs = UserPreference(user_id=user_id, focus_exercises=["pushup", "pullup", "squat"])
            session.add(prefs)
        if level:
            prefs.level = level
        if focus_exercises is not None:
            prefs.focus_exercises = list(dict.fromkeys([fx.lower() for fx in focus_exercises])) or ["pushup"]
        if daily_goal is not None:
            prefs.daily_goal = max(1, int(daily_goal))
        if smart_goal is not None:
            prefs.smart_goal = bool(smart_goal)
        if timezone:
            prefs.timezone = timezone
        if reminder_hour is not None:
            prefs.reminder_hour = int(max(0, min(23, reminder_hour)))
        if allow_notifications is not None:
            prefs.allow_notifications = bool(allow_notifications)
        session.add(prefs)
        session.commit()
        session.refresh(prefs)
        logger.info("Updated preferences for user {}", user_id)
        return prefs


def _compute_accuracy(rep_count: int, warnings: int) -> float:
    if rep_count <= 0:
        return 0.0
    return max(0.0, 1.0 - (warnings / max(rep_count, 1)))


def _compute_consistency(rep_count: int, duration_seconds: float) -> float:
    if rep_count <= 0 or duration_seconds <= 0:
        return 0.0
    reps_per_minute = rep_count / (duration_seconds / 60.0)
    return min(1.0, reps_per_minute / 60.0)


def _xp_for_next_level(level: int) -> int:
    return 100 + (level - 1) * 50


def _grant_experience(session: Session, user: User, xp_earned: int) -> None:
    if xp_earned <= 0:
        return
    user.experience_points += xp_earned
    leveled_up = False
    while user.experience_points >= _xp_for_next_level(user.level):
        user.experience_points -= _xp_for_next_level(user.level)
        user.level += 1
        leveled_up = True
    session.add(user)
    if leveled_up:
        unlock_achievement(
            user.id,
            code=f"level_{user.level}",
            name=f"Level {user.level}",
            description="You levelled up! Keep crushing those reps.",
            session=session,
        )


def record_session(
    *,
    user_id: str,
    exercise: str,
    rep_count: int,
    warnings: int,
    started_at: dt.datetime,
    ended_at: dt.datetime,
    session_metadata: Optional[Dict[str, object]] = None,
) -> SessionRecord:
    accuracy = _compute_accuracy(rep_count, warnings)
    consistency = _compute_consistency(rep_count, (ended_at - started_at).total_seconds())
    xp_earned = max(0, int(rep_count * (0.6 + accuracy + consistency)))
    record = SessionRecord(
        user_id=user_id,
        exercise=exercise,
        rep_count=rep_count,
        accuracy=accuracy,
        consistency=consistency,
        warnings_count=warnings,
        xp_earned=xp_earned,
        started_at=started_at,
        ended_at=ended_at,
        session_metadata=session_metadata or {},
    )
    with session_scope() as session:
        session.add(record)
        session.commit()
        session.refresh(record)
        _update_daily_progress(session, record)
        _grant_experience(session, session.get(User, user_id), xp_earned)
        _maybe_create_recommendation(session, record)
        _maybe_unlock_achievements(session, record)
    logger.info(
        "Session recorded for user {} | exercise={} reps={} accuracy={:.2f} consistency={:.2f} xp={}",
        user_id,
        exercise,
        rep_count,
        record.accuracy,
        record.consistency,
        xp_earned,
    )
    return record


def _update_daily_progress(session: Session, record: SessionRecord) -> None:
    day = record.ended_at.date()
    progress = session.exec(
        select(DailyProgress).where(
            DailyProgress.user_id == record.user_id,
            DailyProgress.date == day,
            DailyProgress.exercise == record.exercise,
        )
    ).first()
    target_reps = _compute_target_for_day(record.user_id, record.exercise, session)
    if progress is None:
        progress = DailyProgress(
            user_id=record.user_id,
            date=day,
            exercise=record.exercise,
            reps_completed=record.rep_count,
            target_reps=target_reps,
            accuracy=record.accuracy,
            consistency=record.consistency,
            warnings_count=record.warnings_count,
        )
    else:
        progress.reps_completed += record.rep_count
        progress.target_reps = target_reps
        progress.accuracy = (progress.accuracy + record.accuracy) / 2.0
        progress.consistency = (progress.consistency + record.consistency) / 2.0
        progress.warnings_count += record.warnings_count
    session.add(progress)
    session.commit()


def _compute_target_for_day(user_id: str, exercise: str, session: Session) -> int:
    prefs = session.exec(select(UserPreference).where(UserPreference.user_id == user_id)).first()
    if prefs is None:
        return 20
    base = prefs.daily_goal
    streak = compute_streak(user_id, exercise, session=session)
    bonus = min(10, streak * 2)
    return base + bonus


def compute_streak(user_id: str, exercise: str, *, session: Optional[Session] = None) -> int:
    owns_session = session is None
    if owns_session:
        session = Session(_get_engine())
    assert session is not None
    try:
        today = dt.date.today()
        streak = 0
        for offset in range(0, 30):
            day = today - dt.timedelta(days=offset)
            entry = session.exec(
                select(DailyProgress).where(
                    DailyProgress.user_id == user_id,
                    DailyProgress.exercise == exercise,
                    DailyProgress.date == day,
                )
            ).first()
            if entry and entry.reps_completed > 0:
                streak += 1
            else:
                break
        return streak
    finally:
        if owns_session:
            session.close()


def unlock_achievement(
    user_id: str,
    *,
    code: str,
    name: str,
    description: str,
    session: Optional[Session] = None,
) -> Optional[Achievement]:
    owns_session = session is None
    if owns_session:
        session = Session(_get_engine())
    assert session is not None
    try:
        existing = session.exec(
            select(Achievement).where(Achievement.user_id == user_id, Achievement.code == code)
        ).first()
        if existing:
            return None
        achievement = Achievement(user_id=user_id, code=code, name=name, description=description)
        session.add(achievement)
        session.commit()
        session.refresh(achievement)
        logger.info("Achievement unlocked for user {}: {}", user_id, code)
        return achievement
    finally:
        if owns_session:
            session.close()


def _maybe_unlock_achievements(session: Session, record: SessionRecord) -> None:
    if record.rep_count <= 0:
        return
    unlock_achievement(
        record.user_id,
        code="first_session",
        name="First Steps",
        description="Completed the first tracked workout session.",
        session=session,
    )
    if record.rep_count >= 50:
        unlock_achievement(
            record.user_id,
            code="fifty_reps",
            name="Rep Machine",
            description="Logged 50+ reps in a single session.",
            session=session,
        )
    if record.accuracy >= 0.9:
        unlock_achievement(
            record.user_id,
            code="form_master",
            name="Form Master",
            description="Maintained 90%+ form accuracy.",
            session=session,
        )
    prefs = session.exec(select(UserPreference).where(UserPreference.user_id == record.user_id)).first()
    if prefs and record.rep_count >= prefs.daily_goal:
        unlock_achievement(
            record.user_id,
            code="goal_crusher",
            name="Goal Crusher",
            description="Crushed the daily goal in a single run.",
            session=session,
        )
    streak = compute_streak(record.user_id, record.exercise, session=session)
    if streak >= 5:
        unlock_achievement(
            record.user_id,
            code=f"streak_{streak}",
            name=f"{streak}-Day Streak",
            description=f"Maintained a {streak}-day streak on {record.exercise}.",
            session=session,
        )


def _maybe_create_recommendation(session: Session, record: SessionRecord) -> None:
    if record.rep_count < 1:
        return
    prefs = session.exec(select(UserPreference).where(UserPreference.user_id == record.user_id)).first()
    if not prefs:
        return
    if record.accuracy > 0.85 and record.consistency > 0.6 and prefs.smart_goal:
        prefs.daily_goal = int(prefs.daily_goal * 1.1) + 1
        session.add(prefs)
        recommendation = AdaptiveRecommendation(
            user_id=record.user_id,
            suggestion=f"Great work! Increase {record.exercise} goal to {prefs.daily_goal} reps.",
            category="progression",
        )
        session.add(recommendation)
        session.commit()


def list_achievements(user_id: str) -> List[Dict[str, object]]:
    with session_scope() as session:
        rows = session.exec(
            select(Achievement).where(Achievement.user_id == user_id).order_by(Achievement.earned_at.desc())
        ).all()
        return [
            {
                "code": row.code,
                "name": row.name,
                "description": row.description,
                "earned_at": row.earned_at,
            }
            for row in rows
        ]


def list_reminders(user_id: str, include_completed: bool = False) -> List[Dict[str, object]]:
    with session_scope() as session:
        query = select(UserReminder).where(UserReminder.user_id == user_id)
        if not include_completed:
            query = query.where(UserReminder.completed.is_(False))
        rows = session.exec(query.order_by(UserReminder.remind_at.asc())).all()
        return [
            {
                "id": row.id,
                "message": row.message,
                "remind_at": row.remind_at,
                "completed": row.completed,
            }
            for row in rows
        ]


def schedule_reminder(user_id: str, message: str, remind_at: dt.datetime) -> UserReminder:
    reminder = UserReminder(user_id=user_id, message=message, remind_at=remind_at)
    with session_scope() as session:
        session.add(reminder)
        session.commit()
        session.refresh(reminder)
        return reminder


def complete_reminder(reminder_id: int, user_id: str) -> None:
    with session_scope() as session:
        reminder = session.get(UserReminder, reminder_id)
        if reminder and reminder.user_id == user_id:
            reminder.completed = True
            session.add(reminder)
            session.commit()


def get_dashboard(user_id: str) -> Dict[str, object]:
    with session_scope() as session:
        user = session.get(User, user_id)
        if not user:
            raise ValueError("User not found")
        prefs = session.exec(select(UserPreference).where(UserPreference.user_id == user_id)).first()
        progress_rows = session.exec(
            select(DailyProgress)
            .where(DailyProgress.user_id == user_id)
            .order_by(DailyProgress.date.desc())
        ).all()
        recommendations = session.exec(
            select(AdaptiveRecommendation)
            .where(AdaptiveRecommendation.user_id == user_id)
            .order_by(AdaptiveRecommendation.created_at.desc())
            .limit(10)
        ).all()
        streak_targets = prefs.focus_exercises if prefs else ["pushup", "pullup", "squat"]
        streaks = {exercise: compute_streak(user_id, exercise, session=session) for exercise in streak_targets}
        timeline = [
            {
                "date": row.date.isoformat(),
                "exercise": row.exercise,
                "reps": row.reps_completed,
                "target": row.target_reps,
                "accuracy": round(row.accuracy, 2),
                "consistency": round(row.consistency, 2),
            }
            for row in progress_rows
        ]
        xp_to_next = max(0, _xp_for_next_level(user.level) - user.experience_points)
        achievements = session.exec(
            select(Achievement)
            .where(Achievement.user_id == user_id)
            .order_by(Achievement.earned_at.desc())
        ).all()
        reminders = list_reminders(user_id)
        social_suggestions = [
            "Share your streak with friends for bonus accountability.",
            "Invite a workout buddy to challenge your best rep count.",
        ]
        return {
            "user": {
                "id": user.id,
                "email": user.email,
                "display_name": user.display_name,
                "avatar_url": user.avatar_url,
                "member_since": user.created_at,
                "level": user.level,
                "experience": user.experience_points,
                "xp_to_next_level": xp_to_next,
            },
            "preferences": {
                "level": prefs.level if prefs else "beginner",
                "focus_exercises": streak_targets,
                "daily_goal": prefs.daily_goal if prefs else 20,
                "smart_goal": prefs.smart_goal if prefs else True,
                "timezone": prefs.timezone if prefs else "UTC",
                "reminder_hour": prefs.reminder_hour if prefs else 9,
                "allow_notifications": prefs.allow_notifications if prefs else True,
            },
            "streaks": streaks,
            "progress_timeline": timeline,
            "recommendations": [
                {
                    "id": rec.id,
                    "message": rec.suggestion,
                    "category": rec.category,
                    "created_at": rec.created_at,
                }
                for rec in recommendations
            ],
            "achievements": [
                {
                    "code": ach.code,
                    "name": ach.name,
                    "description": ach.description,
                    "earned_at": ach.earned_at,
                }
                for ach in achievements
            ],
            "reminders": reminders,
            "suggestions": social_suggestions,
        }


def list_sessions(user_id: str, limit: int = 20) -> List[Dict[str, object]]:
    with session_scope() as session:
        records = session.exec(
            select(SessionRecord)
            .where(SessionRecord.user_id == user_id)
            .order_by(SessionRecord.ended_at.desc())
            .limit(limit)
        ).all()
        return [
            {
                "exercise": rec.exercise,
                "rep_count": rec.rep_count,
                "accuracy": round(rec.accuracy, 2),
                "consistency": round(rec.consistency, 2),
                "warnings": rec.warnings_count,
                "xp": rec.xp_earned,
                "started_at": rec.started_at,
                "ended_at": rec.ended_at,
            }
            for rec in records
        ]

