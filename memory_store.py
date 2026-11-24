from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MEMORY_FILE = Path("data/leo_student_memory.json")


def load_student_memory() -> dict[str, Any]:
    """Load application-level student memory from JSON."""
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    return {
        "students": {}
    }


def get_student_history(student_name: str) -> dict[str, Any]:
    memory = load_student_memory()
    return memory.get("students", {}).get(
        student_name.strip().lower(),
        {
            "display_name": student_name.strip(),
            "last_topic": None,
            "last_level": None,
            "previous_score": None,
            "sessions": [],
        },
    )


def update_student_memory(
    student_name: str,
    topic: str,
    level: str,
    score: int,
    total_questions: int,
    percentage: float,
) -> dict[str, Any]:
    """Persist the student's latest session."""
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    memory = load_student_memory()
    key = student_name.strip().lower()

    student = memory.setdefault("students", {}).setdefault(
        key,
        {
            "display_name": student_name.strip(),
            "last_topic": None,
            "last_level": None,
            "previous_score": None,
            "sessions": [],
        },
    )

    student["display_name"] = student_name.strip()
    student["last_topic"] = topic
    student["last_level"] = level
    student["previous_score"] = percentage
    student["sessions"].append(
        {
            "topic": topic,
            "level": level,
            "score": score,
            "total_questions": total_questions,
            "percentage": percentage,
        }
    )

    MEMORY_FILE.write_text(
        json.dumps(memory, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return student
