from pathlib import Path
import re

import streamlit as st

st.set_page_config(page_title="Semantic Pipeline Help/Guide", layout="wide")
st.title("Semantic Pipeline Help/Guide")
st.caption("Reference page for the context-aware semantic audio workflow.")

doc_path = Path(__file__).resolve().parent.parent / "SemanticPipeline.md"

if not doc_path.exists():
    st.error(f"Could not find {doc_path.name}.")
else:
    doc_text = doc_path.read_text(encoding="utf-8")
    parts = re.split(r"(?m)^##\s+", doc_text)
    preamble = parts[0].strip()
    if preamble:
        st.markdown(preamble)

    for part in parts[1:]:
        title, _, body = part.partition("\n")
        title = title.strip()
        body = body.strip()
        if title == "Complete Rule-Based Adjustment Guide":
            st.markdown(f"## {title}")
            guide_intro, *section_chunks = body.split("\n### ")
            if guide_intro.strip():
                st.markdown(guide_intro.strip())
            for chunk in section_chunks:
                section_title, _, section_body = chunk.partition("\n")
                section_title = section_title.strip()
                with st.expander(section_title, expanded=(section_title == "Complete Rule Table")):
                    st.markdown(section_body.strip())
        else:
            with st.expander(title, expanded=(title == "Purpose")):
                st.markdown(body)
