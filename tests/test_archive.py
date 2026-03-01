"""Tests for the Hierarchical Archive."""

import pytest
from core.archive import HierarchicalArchive
from memory.vector_store import VectorStore


class TestHierarchicalArchive:
    def setup_method(self):
        self.vs = VectorStore(collection_name="test_archive")
        self.archive = HierarchicalArchive(
            vector_store=self.vs,
            max_l0_entries=5,
            segment_size=3  # small for testing
        )

    def teardown_method(self):
        self.vs.clear()

    def test_add_turn(self):
        self.archive.add_turn(1, "Hello", "Hi there", "Greeting exchange", "Greeting")
        assert self.archive.total_turns == 1

    def test_l0_directory_text(self):
        self.archive.add_turn(1, "Setup project", "Done", "Project setup", "Setup")
        text = self.archive.get_l0_directory_text()
        assert "seg_001" in text

    def test_segment_rotation(self):
        # With segment_size=3, adding 4 turns should create 2 segments
        for i in range(4):
            self.archive.add_turn(i+1, f"msg {i}", f"resp {i}", f"summary {i}", f"bullet {i}")
        assert len(self.archive.l0_directory) == 2

    def test_l1_retrieval(self):
        self.archive.add_turn(1, "Setup Flask", "Done", "Flask setup", "Set up Flask app")
        text = self.archive.get_l1_segment("seg_001")
        assert text is not None
        assert "Flask" in text

    def test_l0_stays_bounded(self):
        # With max_l0_entries=5 and segment_size=3, after 20 turns
        # we should have created ~7 segments but merged to stay ≤5
        for i in range(20):
            self.archive.add_turn(
                i+1, f"user {i}", f"assistant {i}",
                f"summary of turn {i}", f"bullet {i}"
            )
        assert len(self.archive.l0_directory) <= 6  # 5 + 1 current

    def test_semantic_search(self):
        self.archive.add_turn(1, "Set up PostgreSQL database", "Done",
                              "PostgreSQL database was configured", "DB setup")
        results = self.archive.semantic_search("database", top_k=1)
        assert len(results) >= 1
