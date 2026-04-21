"""Tests for the Context Assembler."""

import pytest
from hiermem.core.assembler import ContextAssembler, ContextChunk


class TestContextAssembler:
    def setup_method(self):
        self.assembler = ContextAssembler(total_budget=2000)

    def test_basic_assembly(self):
        result = self.assembler.assemble(
            constraints_text="[RULE] Never modify auth.py",
            relevant_chunks=[
                ContextChunk.from_text("Previous discussion about auth", "L1:seg_001")
            ],
            peripheral_summaries=["seg_002: Database work"],
            current_prompt="Update the login endpoint"
        )
        assert "auth.py" in result.full_text
        assert "Update the login" in result.full_text
        assert result.total_tokens_est > 0

    def test_constraints_before_prompt(self):
        result = self.assembler.assemble(
            constraints_text="[RULE] Use Python 3.11",
            relevant_chunks=[],
            peripheral_summaries=[],
            current_prompt="Write a function"
        )
        text = result.full_text
        constraint_pos = text.index("Python 3.11")
        prompt_pos = text.index("Write a function")
        assert constraint_pos < prompt_pos

    def test_empty_constraints(self):
        result = self.assembler.assemble(
            constraints_text="",
            relevant_chunks=[],
            peripheral_summaries=[],
            current_prompt="Hello"
        )
        assert "Hello" in result.full_text

    def test_peripheral_included(self):
        result = self.assembler.assemble(
            constraints_text="",
            relevant_chunks=[],
            peripheral_summaries=["Topic A summary", "Topic B summary"],
            current_prompt="Question"
        )
        assert "Topic A" in result.full_text
        assert "Topic B" in result.full_text

    def test_sources_tracked(self):
        result = self.assembler.assemble(
            constraints_text="",
            relevant_chunks=[
                ContextChunk.from_text("chunk 1", "L1:seg_001"),
                ContextChunk.from_text("chunk 2", "L2:semantic"),
            ],
            peripheral_summaries=[],
            current_prompt="Query"
        )
        assert "L1:seg_001" in result.sources_used
        assert "L2:semantic" in result.sources_used
