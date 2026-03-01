"""
HierMem Demo — Interactive Streamlit app for testing the pipeline.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pipeline import HierMemPipeline


def init_pipeline():
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = HierMemPipeline.create()
    if "messages" not in st.session_state:
        st.session_state.messages = []


def main():
    st.set_page_config(page_title="HierMem Demo", page_icon="🧠", layout="wide")
    st.title("🧠 HierMem — Hierarchical Paged Context Management")
    st.caption("A research demo for long-context conversation with constraint tracking")
    
    init_pipeline()
    
    # Sidebar: system info
    with st.sidebar:
        st.header("Pipeline State")
        pipeline = st.session_state.pipeline
        st.metric("Turns processed", pipeline.turn_count)
        st.metric("Active constraints", len(pipeline.constraint_store))
        st.metric("L0 segments", len(pipeline.archive.l0_directory))
        
        if st.button("🗑️ Reset conversation"):
            st.session_state.pipeline = HierMemPipeline.create()
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        st.subheader("Active Constraints")
        constraints = pipeline.constraint_store.get_all_active()
        if constraints:
            for c in constraints:
                st.text(f"[{c.category}] {c.text[:60]}...")
        else:
            st.text("No active constraints.")
    
    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # User input
    if user_input := st.chat_input("Type your message..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                result = pipeline.process_turn(user_input)
                response = result.assistant_response if hasattr(result, 'assistant_response') else str(result)
            st.write(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


if __name__ == "__main__":
    main()
