"""Синтез и вставка контента: планирование, генерация, inpainting, композитинг."""
from .planner import InsertionPlanner, InsertionEvent
from .generator import ContentGenerator
from .composite import CompositePipeline

__all__ = ["InsertionPlanner", "InsertionEvent", "ContentGenerator", "CompositePipeline"]
