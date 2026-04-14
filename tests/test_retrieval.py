"""Tests for the Retrieval Router."""

import pytest
from core.curator import CuratorDecision
from core.archive import HierarchicalArchive
from retrieval.router import RetrievalRouter
from memory.vector_store import VectorStore


class TestRetrievalRouter:
    def setup_method(self):
        self.vs = VectorStore(collection_name="test_router")
        self.archive = HierarchicalArchive(
            vector_store=self.vs, max_l0_entries=10, segment_size=5
        )
        self.router = RetrievalRouter(archive=self.archive, vector_store=self.vs)

    def teardown_method(self):
        self.vs.clear()

    def test_none_strategy(self):
        decision = CuratorDecision.empty()
        chunks, peripheral = self.router.retrieve(decision)
        assert len(chunks) == 0
        assert len(peripheral) == 0

    def test_hierarchy_retrieval(self):
        # Add some data
        self.archive.add_turn(1, "Setup Flask", "Done", "Flask setup", "Set up Flask")
        
        decision = CuratorDecision(
            retrieval_strategy="HIERARCHY",
            segments_to_fetch=["seg_001"],
            semantic_queries=[],
            peripheral_segments=[],
            fetch_full_turns=[],
            reasoning="test"
        )
        chunks, peripheral = self.router.retrieve(decision)
        assert len(chunks) >= 1

    def test_semantic_retrieval(self):
        self.archive.add_turn(1, "PostgreSQL database setup", "Done",
                              "Database configured with PostgreSQL", "DB setup")
        
        decision = CuratorDecision(
            retrieval_strategy="SEMANTIC",
            segments_to_fetch=[],
            semantic_queries=["database configuration"],
            peripheral_segments=[],
            fetch_full_turns=[],
            reasoning="test"
        )
        chunks, peripheral = self.router.retrieve(decision)
        # Should find something related to database
        assert len(chunks) >= 1

    def test_deduplication(self):
        assert not self.router._is_duplicate("completely different text", [])
