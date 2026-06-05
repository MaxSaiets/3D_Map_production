"""
Клас для відстеження статусу задачі генерації
"""
from dataclasses import dataclass, field
from typing import Optional, Dict
from datetime import datetime


@dataclass
class GenerationTask:
    """Задача генерації 3D моделі"""
    task_id: str
    request: object
    status: str = "pending"  # pending, processing, completed, failed, cancelled
    progress: int = 0  # 0-100
    message: str = ""
    # Основний файл, який повертається за замовчуванням (наприклад, 3MF або STL)
    output_file: Optional[str] = None
    # Набір доступних файлів по форматах: {"3mf": "...", "stl": "..."}
    output_files: Dict[str, str] = field(default_factory=dict)
    # Набір хмарних посилань: {"base_stl": "...", "3mf": "..."}
    firebase_outputs: Dict[str, str] = field(default_factory=dict)
    firebase_url: Optional[str] = None
    # Print-quality / QA outcome surfaced to the client:
    # {"status": "ok"|"warning", "warnings": [..], "report": "path"}
    print_quality: Optional[Dict] = None
    error: Optional[str] = None
    # Скасування та TTL
    cancelled: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)

    def update_status(self, status: str, progress: int, message: str = ""):
        """Оновлює статус задачі. Якщо задача скасована — ігнорує оновлення."""
        if self.cancelled:
            return
        self.status = status
        self.progress = progress
        self.message = message

    def cancel(self):
        """Позначає задачу як скасовану"""
        self.cancelled = True
        if self.status not in ("completed", "failed"):
            self.status = "cancelled"
            self.message = "Скасовано користувачем"

    def is_stale(self, max_age_hours: float = 2.0) -> bool:
        """Перевіряє чи задача застаріла (старша за max_age_hours)"""
        age = (datetime.utcnow() - self.created_at).total_seconds() / 3600
        return age > max_age_hours

    def complete(self, output_file: str):
        """Позначає задачу як виконану"""
        if self.cancelled:
            return
        self.status = "completed"
        self.progress = 100
        self.output_file = output_file

    def set_output(self, fmt: str, path: str):
        """Зберігає шлях до вихідного файлу для конкретного формату"""
        self.output_files[fmt.lower()] = path

    def fail(self, error: str):
        """Позначає задачу як невдалу"""
        if self.cancelled:
            return
        self.status = "failed"
        self.error = error
        self.message = f"Помилка: {error}"
