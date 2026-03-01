"""Tests for the Constraint Store."""

import pytest
from core.constraint_store import ConstraintStore, Constraint


class TestConstraintStore:
    def setup_method(self):
        self.store = ConstraintStore(max_constraints=5)

    def test_add_constraint(self):
        cid = self.store.add("Never modify auth.py", category="RULE", priority=5)
        assert len(self.store) == 1
        assert cid is not None

    def test_get_all_active(self):
        self.store.add("Rule 1", category="RULE", priority=5)
        self.store.add("Rule 2", category="PREFERENCE", priority=3)
        active = self.store.get_all_active()
        assert len(active) == 2
        # Should be sorted by priority (highest first)
        assert active[0].priority >= active[1].priority

    def test_deactivate(self):
        cid = self.store.add("Temp rule", category="RULE", priority=3)
        assert len(self.store) == 1
        self.store.deactivate(cid)
        assert len(self.store) == 0

    def test_violation_check(self):
        self.store.add("Never modify auth.py", category="RULE", priority=5,
                       keywords=["auth.py", "modify"])
        violations = self.store.check_violation("I will modify auth.py now")
        assert len(violations) == 1

    def test_no_false_violation(self):
        self.store.add("Never modify auth.py", category="RULE", priority=5,
                       keywords=["auth.py"])
        violations = self.store.check_violation("I updated the database schema")
        assert len(violations) == 0

    def test_capacity_eviction(self):
        # Add max_constraints + 1
        for i in range(6):
            self.store.add(f"Rule {i}", category="RULE", priority=i)
        # Should not exceed max
        assert len(self.store) <= 5

    def test_display_text(self):
        self.store.add("Use Python 3.11+", category="FACT", priority=4)
        text = self.store.get_display_text()
        assert "Python 3.11" in text
        assert "FACT" in text

    def test_keyword_extraction(self):
        cid = self.store.add("Never modify 'config.py' in the project",
                             category="RULE", priority=5)
        constraint = self.store.constraints[cid]
        assert "config.py" in constraint.keywords
