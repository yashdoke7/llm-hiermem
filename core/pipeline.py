"""
HierMem Pipeline — Main orchestration loop.

This is the heart of the system. For each user turn:
  1. Curator selects context (reads L0 index + constraints)
  2. Retrieval engine fetches from L1/L2/L3 based on curator's decision
  3. Assembler builds attention-optimized context
  4. Main LLM generates response
  5. Post-processor updates memory stores
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass, field

from llm.client import LLMClient
from core.constraint_store import ConstraintStore
from core.archive import HierarchicalArchive
from core.curator import ContextCurator, CuratorDecision
from core.assembler import ContextAssembler, ContextChunk, AssembledContext
from core.post_processor import PostProcessor
from retrieval.router import RetrievalRouter
import config


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
        
        provider = provider or config.DEFAULT_PROVIDER
        
        # Initialize LLM clients
        main_llm = LLMClient(provider=provider)
        curator_llm = LLMClient(provider=provider)
        summarizer_llm = LLMClient(provider=provider)
        
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
        Process one conversation turn through the full pipeline.
        
        Args:
            user_message: The user's input
            system_prompt: Optional system prompt for the main LLM
            
        Returns:
            TurnResult with the response and metadata
        """
        self.turn_count += 1
        turn_num = self.turn_count

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

        # Step 6: Post-process (constraint check, extract constraints, summarize, archive)
        final_response, warnings = self.post_processor.process(
            turn_number=turn_num,
            user_msg=user_message,
            assistant_msg=response
        )

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

    def _get_recent_turn_chunks(self) -> List[ContextChunk]:
        """Get the last N turns as ContextChunks."""
        if self.turn_count <= 1:
            return []
        
        recent = self.archive.get_recent_turns(n=config.RECENT_TURNS_ALWAYS_INCLUDED)
        return [
            ContextChunk.from_text(text=r.get("text", ""), source=f"recent:turn_{r.get('turn', '?')}")
            for r in recent
        ]

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
