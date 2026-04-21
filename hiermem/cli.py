"""
HierMem CLI — Interactive chat and one-shot query interface.

Usage:
    hiermem chat                           # Interactive chat (Ollama default)
    hiermem chat --provider openai         # Chat with OpenAI
    hiermem ask "What is Python?"          # One-shot query
    hiermem config                         # Show current configuration
"""

import argparse
import sys
import os


def cmd_chat(args):
    """Interactive chat session with HierMem pipeline."""
    # Set env vars before importing config (which reads them at import time)
    if args.provider:
        os.environ["MAIN_PROVIDER"] = args.provider
    if args.model:
        os.environ["MAIN_LLM_MODEL"] = args.model
    if args.curator_model:
        os.environ["CURATOR_MODEL"] = args.curator_model
    if args.budget:
        os.environ["TOTAL_CONTEXT_BUDGET"] = str(args.budget)
        os.environ["HIERMEM_CONTEXT_BUDGET"] = str(args.budget)

    from hiermem.core.pipeline import HierMemPipeline

    print("\n" + "=" * 60)
    print("  🧠 HierMem — Hierarchical Context Memory")
    print("=" * 60)
    print(f"  Provider : {os.getenv('MAIN_PROVIDER', 'ollama')}")
    print(f"  Model    : {os.getenv('MAIN_LLM_MODEL', 'ollama/llama3.1:8b')}")
    print(f"  Budget   : {os.getenv('HIERMEM_CONTEXT_BUDGET', '8192')} tokens")
    print("-" * 60)
    print("  Type your message. Commands:")
    print("    /constraints  — Show active constraints")
    print("    /stats        — Show pipeline statistics")
    print("    /quit         — Exit")
    print("=" * 60 + "\n")

    try:
        pipeline = HierMemPipeline.create()
    except Exception as e:
        print(f"\n❌ Failed to create pipeline: {e}")
        print("   Check your provider settings and API keys.")
        sys.exit(1)

    # Pre-load any --constraint flags
    if args.constraint:
        for c in args.constraint:
            pipeline.constraint_store.add_constraint(c, source_turn=0)
            print(f"  📌 Pre-loaded constraint: {c}")
        print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("\nGoodbye! 👋")
            break

        if user_input.lower() == "/constraints":
            cs = pipeline.constraint_store
            if cs.count == 0:
                print("  No active constraints.\n")
            else:
                print(f"  Active constraints ({cs.count}):")
                for c in cs.get_all():
                    text = c.get("text", c) if isinstance(c, dict) else str(c)
                    print(f"    • {text}")
                print()
            continue

        if user_input.lower() == "/stats":
            print(f"  Turn count   : {pipeline.turn_count}")
            print(f"  Constraints  : {pipeline.constraint_store.count}")
            print(f"  L0 segments  : {len(pipeline.archive.l0_directory)}")
            print(f"  Context budget: {pipeline.context_budget}")
            if pipeline.history:
                last = pipeline.history[-1]
                print(f"  Last ctx tokens: {last.tokens_used}")
                pt = last.pipeline_telemetry
                if pt:
                    print(f"  Last passthrough: {pt.get('passthrough_mode', '?')}")
                    print(f"  Last curator tok: {pt.get('curator_total_tokens', '?')}")
            print()
            continue

        # Process turn
        try:
            result = pipeline.process_turn(user_input)
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            continue

        print(f"\nAssistant: {result.assistant_response}\n")

        # Show telemetry summary
        pt = result.pipeline_telemetry
        mode = "passthrough" if pt.get("passthrough_mode") else "curated"
        constraints = pt.get("constraint_count", 0)
        if result.warnings:
            for w in result.warnings:
                print(f"  ⚠️  {w}")
        print(f"  [{mode} | {result.tokens_used} tok | {constraints} constraints]")
        print()


def cmd_ask(args):
    """One-shot query — process a single message and exit."""
    if args.provider:
        os.environ["MAIN_PROVIDER"] = args.provider
    if args.model:
        os.environ["MAIN_LLM_MODEL"] = args.model

    from hiermem.core.pipeline import HierMemPipeline

    pipeline = HierMemPipeline.create()
    if args.constraint:
        for c in args.constraint:
            pipeline.constraint_store.add_constraint(c, source_turn=0)

    result = pipeline.process_turn(args.message)
    print(result.assistant_response)


def cmd_config(args):
    """Display current configuration."""
    import hiermem.config as cfg
    print("\n🧠 HierMem Configuration")
    print("=" * 50)
    print(f"  MAIN_PROVIDER        : {cfg.MAIN_PROVIDER}")
    print(f"  CURATOR_PROVIDER     : {cfg.CURATOR_PROVIDER}")
    print(f"  SUMMARIZER_PROVIDER  : {cfg.SUMMARIZER_PROVIDER}")
    print(f"  MAIN_LLM_MODEL      : {cfg.MAIN_LLM_MODEL}")
    print(f"  CURATOR_MODEL        : {cfg.CURATOR_MODEL}")
    print(f"  SUMMARIZER_MODEL     : {cfg.SUMMARIZER_MODEL}")
    print(f"  TOTAL_CONTEXT_BUDGET : {cfg.TOTAL_CONTEXT_BUDGET}")
    print(f"  HIERMEM_CONTEXT_BUDGET: {cfg.HIERMEM_CONTEXT_BUDGET}")
    print(f"  PASSTHROUGH_THRESHOLD: {cfg.PASSTHROUGH_THRESHOLD}")
    print(f"  MAX_CONSTRAINTS      : {cfg.MAX_CONSTRAINTS}")
    print(f"  MAX_L0_ENTRIES       : {cfg.MAX_L0_ENTRIES}")
    print(f"  SEGMENT_SIZE         : {cfg.SEGMENT_SIZE}")
    print(f"  OLLAMA_BASE_URL      : {cfg.OLLAMA_BASE_URL}")
    print(f"  OLLAMA_CONTEXT_SIZE  : {cfg.OLLAMA_CONTEXT_SIZE}")
    print("=" * 50)

    # Check API keys
    keys = {
        "GROQ_API_KEY": bool(cfg.GROQ_API_KEY),
        "OPENAI_API_KEY": bool(cfg.OPENAI_API_KEY),
        "ANTHROPIC_API_KEY": bool(cfg.ANTHROPIC_API_KEY),
        "GOOGLE_API_KEY": bool(cfg.GOOGLE_API_KEY),
    }
    print("\n  API Keys:")
    for k, v in keys.items():
        status = "✅ set" if v else "❌ not set"
        print(f"    {k:25s}: {status}")
    print()


def main():
    parser = argparse.ArgumentParser(
        prog="hiermem",
        description="HierMem — Hierarchical Paged Context Management for LLMs",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- chat ---
    chat_parser = subparsers.add_parser("chat", help="Interactive chat session")
    chat_parser.add_argument("--provider", "-p", help="LLM provider (ollama/openai/anthropic/google/groq)")
    chat_parser.add_argument("--model", "-m", help="Main LLM model name")
    chat_parser.add_argument("--curator-model", help="Curator model name")
    chat_parser.add_argument("--budget", "-b", type=int, help="Context token budget")
    chat_parser.add_argument("--constraint", "-c", action="append", help="Pre-load a constraint (repeatable)")
    chat_parser.set_defaults(func=cmd_chat)

    # --- ask ---
    ask_parser = subparsers.add_parser("ask", help="One-shot query")
    ask_parser.add_argument("message", help="Message to send")
    ask_parser.add_argument("--provider", "-p", help="LLM provider")
    ask_parser.add_argument("--model", "-m", help="Main LLM model name")
    ask_parser.add_argument("--constraint", "-c", action="append", help="Pre-load a constraint (repeatable)")
    ask_parser.set_defaults(func=cmd_ask)

    # --- config ---
    config_parser = subparsers.add_parser("config", help="Show current configuration")
    config_parser.set_defaults(func=cmd_config)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
