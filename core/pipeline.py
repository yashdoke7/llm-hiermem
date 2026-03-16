"""
HierMem Pipeline — Main orchestration loop.

This is the heart of the system. For each user turn:
  1. Curator selects context (reads L0 index + constraints)
  2. Retrieval engine fetches from L1/L2/L3 based on curator's decision
  3. Assembler builds attention-optimized context
  4. Main LLM generates response
  5. Post-processor updates memory stores
"""

import re
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

from llm.client import LLMClient
from llm.token_counter import count_tokens
from core.constraint_store import ConstraintStore
from core.archive import HierarchicalArchive
from core.curator import ContextCurator, CuratorDecision
from core.assembler import ContextAssembler, ContextChunk, AssembledContext
from core.post_processor import PostProcessor
from retrieval.router import RetrievalRouter
import config

logger = logging.getLogger(__name__)


def _strip_hallucinated_continuation(text: str) -> str:
    """Remove fake User:/Assistant: exchanges the LLM hallucinates."""
    match = re.search(r'\n\s*User\s*:', text)
    if match:
        return text[:match.start()].rstrip()
    return text


@dataclass
class TurnResult:
    """Result of processing one conversation turn."""
    turn_number: int
    user_message: str
    assistant_response: str
    curator_decision: Optional[CuratorDecision]
    assembled_context: Optional[AssembledContext]
    warnings: List[str] = field(default_factory=list)
    tokens_used: int = 0


class HierMemPipeline:
    """
    Main pipeline orchestrating all components.
    
    Usage:
        pipeline = HierMemPipeline.create()
        for turn, user_msg in enumerate(conversation):
            result = pipeline.process_turn(user_msg)
            print(result.assistant_response)
    """

    def __init__(self, main_llm: LLMClient, curator: ContextCurator,
                 assembler: ContextAssembler, post_processor: PostProcessor,
                 constraint_store: ConstraintStore, archive: HierarchicalArchive,
                 retrieval_router: RetrievalRouter):
        self.main_llm = main_llm
        self.curator = curator
        self.assembler = assembler
        self.post_processor = post_processor
        self.constraint_store = constraint_store
        self.archive = archive
        self.retrieval_router = retrieval_router
        self.turn_count = 0
        self.history: List[TurnResult] = []

    @classmethod
    def create(cls, main_model: str = None, curator_model: str = None,
               provider: str = None) -> "HierMemPipeline":
        """Factory method to create a fully initialized pipeline."""
        from memory.vector_store import VectorStore
        
        provider = provider or config.MAIN_PROVIDER
        
        # Initialize LLM clients — each role can use a different provider
        # e.g. main on Groq, curator/summarizer on local ollama
        main_llm = LLMClient(provider=config.MAIN_PROVIDER)
        curator_llm = LLMClient(provider=config.CURATOR_PROVIDER)
        summarizer_llm = LLMClient(provider=config.SUMMARIZER_PROVIDER)
        
        # Initialize stores
        vector_store = VectorStore()
        constraint_store = ConstraintStore(max_constraints=config.MAX_CONSTRAINTS)
        archive = HierarchicalArchive(
            vector_store=vector_store,
            max_l0_entries=config.MAX_L0_ENTRIES,
            segment_size=config.SEGMENT_SIZE
        )
        
        # Initialize agents
        curator = ContextCurator(llm_client=curator_llm)
        assembler = ContextAssembler(total_budget=config.TOTAL_CONTEXT_BUDGET)
        post_processor = PostProcessor(
            llm_client=summarizer_llm,
            constraint_store=constraint_store,
            archive=archive
        )
        retrieval_router = RetrievalRouter(archive=archive, vector_store=vector_store)
        
        return cls(
            main_llm=main_llm,
            curator=curator,
            assembler=assembler,
            post_processor=post_processor,
            constraint_store=constraint_store,
            archive=archive,
            retrieval_router=retrieval_router
        )

    def process_turn(self, user_message: str,
                     system_prompt: str = "You are a helpful assistant.") -> TurnResult:
        """
        Process one conversation turn.

        Uses adaptive passthrough: if full history fits within
        PASSTHROUGH_THRESHOLD, send everything to the LLM directly
        (avoiding information loss from curation on short conversations).
        Once history exceeds the threshold, switch to the full HierMem
        pipeline with curated context assembly.

        Post-processing (constraint extraction, archiving) always runs
        so memory is ready when the pipeline activates.
        """
        self.turn_count += 1
        turn_num = self.turn_count

        # Check if we should use passthrough mode
        history_tokens = self._estimate_history_tokens(user_message)
        use_passthrough = history_tokens < config.PASSTHROUGH_THRESHOLD

        if use_passthrough:
            # --- Passthrough: send full history like raw_llm ---
            context = self._build_full_history(user_message)
            assembled = AssembledContext(
                full_text=context,
                total_tokens_est=count_tokens(context),
                zone_breakdown={"passthrough": count_tokens(context)},
                sources_used=["passthrough:full_history"],
            )
            curator_decision = None

            response = self.main_llm.call(
                system_prompt=system_prompt,
                user_prompt=context,
                model=config.MAIN_LLM_MODEL,
                temperature=config.TEMPERATURE_MAIN,
            )
            response = _strip_hallucinated_continuation(response)
        else:
            # --- Full HierMem pipeline ---
            # Step 1: Curator selects context
            curator_decision = self.curator.select_context(
                user_prompt=user_message,
                l0_directory_text=self.archive.get_l0_directory_text(),
                constraints_text=self.constraint_store.get_display_text(),
                total_turns=self.turn_count
            )

            # Step 2: Retrieve based on curator's decision
            relevant_chunks, peripheral_summaries = self.retrieval_router.retrieve(
                decision=curator_decision
            )

            # Step 2.5: Tag pre-constraint chunks (Fix 2 + Fix 5)
            relevant_chunks = self._tag_pre_constraint_chunks(relevant_chunks)

            # Step 3: Get recent turns (always included)
            recent_chunks = self._get_recent_turn_chunks()

            # Step 4: Assemble context
            assembled = self.assembler.assemble(
                constraints_text=self.constraint_store.get_display_text(),
                relevant_chunks=relevant_chunks,
                peripheral_summaries=peripheral_summaries,
                current_prompt=user_message,
                recent_turns=recent_chunks,
                system_prompt=system_prompt
            )

            # Step 5: Call Main LLM
            response = self.main_llm.call(
                system_prompt=system_prompt,
                user_prompt=assembled.full_text,
                model=config.MAIN_LLM_MODEL,
                temperature=config.TEMPERATURE_MAIN,
            )
            response = _strip_hallucinated_continuation(response)

        # Step 6: Post-process (always runs — keeps memory updated)
        # Pass retry info so post-processor can re-call on violation
        retry_context = context if use_passthrough else assembled.full_text
        final_response, warnings = self.post_processor.process(
            turn_number=turn_num,
            user_msg=user_message,
            assistant_msg=response,
            retry_llm=self.main_llm,
            retry_context=retry_context,
            system_prompt=system_prompt,
        )

        if use_passthrough:
            warnings.append(f"passthrough_mode (history {history_tokens} < {config.PASSTHROUGH_THRESHOLD} threshold)")

        result = TurnResult(
            turn_number=turn_num,
            user_message=user_message,
            assistant_response=final_response,
            curator_decision=curator_decision,
            assembled_context=assembled,
            warnings=warnings,
            tokens_used=assembled.total_tokens_est
        )
        self.history.append(result)
        return result

    def _estimate_history_tokens(self, current_message: str) -> int:
        """Estimate total tokens of conversation history including current message."""
        total = count_tokens(current_message)
        for r in self.history:
            total += count_tokens(r.user_message) + count_tokens(r.assistant_response)
        return total

    def _build_full_history(self, current_message: str) -> str:
        """Build full conversation history as plain text (passthrough mode).
        
        Prepends any extracted constraints at the top so they remain prominent
        even before the curator/HYBRID pipeline activates. This is defense-in-depth:
        constraints in the conversation history are already present as raw text,
        but surfacing them explicitly at the top mirrors Zone 1 behaviour.
        """
        lines = []
        constraints_text = self.constraint_store.get_display_text()
        if constraints_text.strip() and constraints_text != "No active constraints.":
            lines.append(f"=== Active Rules ===\n{constraints_text}\n")
        for r in self.history:
            lines.append(f"User: {r.user_message}")
            lines.append(f"Assistant: {r.assistant_response}")
        lines.append(f"User: {current_message}")
        return "\n\n".join(lines)

    def _get_recent_turn_chunks(self) -> List[ContextChunk]:
        """Get the last N turns as ContextChunks."""
        if self.turn_count <= 1:
            return []
        
        recent = self.archive.get_recent_turns(n=config.RECENT_TURNS_ALWAYS_INCLUDED)
        return [
            ContextChunk.from_text(text=r.get("text", ""), source=f"recent:turn_{r.get('turn', '?')}")
            for r in recent
        ]

    def _tag_pre_constraint_chunks(self, chunks: List[ContextChunk]) -> List[ContextChunk]:
        """
        Tag retrieved chunks that predate active constraints.

        If a chunk comes from a segment whose turns ended before the earliest
        active constraint was set, prepend a warning note so the main LLM
        knows to follow the rules instead of the old patterns.
        """
        active = self.constraint_store.get_all_active()
        if not active:
            return chunks

        earliest_constraint_turn = min(c.source_turn for c in active)
        if earliest_constraint_turn <= 0:
            return chunks

        NOTE = (
            "[⚠ PRE-RULE CONTEXT: This was recorded BEFORE current rules were set. "
            "Always follow ACTIVE RULES & CONSTRAINTS above instead of patterns shown here.]\n"
        )

        tagged = []
        for chunk in chunks:
            end_turn = self._get_chunk_end_turn(chunk)
            if end_turn is not None and end_turn < earliest_constraint_turn:
                chunk = ContextChunk(
                    text=NOTE + chunk.text,
                    source=chunk.source,
                    tokens_est=chunk.tokens_est + len(NOTE) // 4
                )
                logger.debug(
                    "Tagged pre-constraint chunk %s (end_turn=%d < constraint_turn=%d)",
                    chunk.source, end_turn, earliest_constraint_turn
                )
            tagged.append(chunk)
        return tagged

    def _get_chunk_end_turn(self, chunk: ContextChunk) -> Optional[int]:
        """Determine the latest turn number in a chunk from its source metadata."""
        source = chunk.source

        # L1:seg_XXX → look up segment turn range in L0 directory
        if source.startswith("L1:"):
            seg_id = source[3:]
            entry = self.archive.l0_directory.get(seg_id)
            if entry:
                return entry.turn_end

        # L3:turn_N or recent:turn_N → direct turn number
        if ":turn_" in source:
            try:
                turn_str = source.split(":turn_")[1]
                return int(turn_str)
            except (ValueError, IndexError):
                pass

        # Semantic results → try to extract turn number from text
        if "Turn " in chunk.text:
            match = re.search(r'Turn\s+(\d+)', chunk.text)
            if match:
                return int(match.group(1))

        return None

    def get_conversation_trace(self) -> List[dict]:
        """Get the full conversation trace for logging/evaluation."""
        return [
            {
                "turn": r.turn_number,
                "user": r.user_message,
                "assistant": r.assistant_response,
                "curator_strategy": r.curator_decision.retrieval_strategy if r.curator_decision else None,
                "context_tokens": r.tokens_used,
                "warnings": r.warnings,
                "sources": r.assembled_context.sources_used if r.assembled_context else []
            }
            for r in self.history
        ]

    def reset(self):
        """Reset the pipeline for a new conversation."""
        self.turn_count = 0
        self.history.clear()
        # Note: stores are NOT reset — they persist across conversations
