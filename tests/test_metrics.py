"""Tests for the evaluation metrics."""

import pytest
from eval.metrics import evaluate_checkpoint, compute_cvr, compute_task_accuracy


class TestMetrics:
    def test_keyword_check_positive(self):
        assert evaluate_checkpoint(
            "I used PostgreSQL for the database",
            "Does response use PostgreSQL?",
            True
        )

    def test_keyword_check_negative(self):
        assert not evaluate_checkpoint(
            "I used MySQL for the database",
            "Does response use PostgreSQL?",
            True
        )

    def test_bullet_point_check(self):
        assert evaluate_checkpoint(
            "• Item 1\n• Item 2\n• Item 3",
            "Is response in bullet points?",
            True
        )

    def test_no_bullet_points(self):
        assert not evaluate_checkpoint(
            "This is a plain paragraph response.",
            "Is response in bullet points?",
            True
        )

    def test_empty_conversations(self):
        assert compute_cvr([]) == 0.0
        assert compute_task_accuracy([]) == 0.0
