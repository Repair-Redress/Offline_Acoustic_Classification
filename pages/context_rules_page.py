import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from semantic_pipeline import DEFAULT_CONTEXT_RULES_PATH, validate_context_rules_config


st.set_page_config(page_title="Context Adjustment Rules", layout="wide")
st.title("Context Adjustment Rules")
st.caption("Edit the repo-backed rule file used by semantic classification.")

rules_path = DEFAULT_CONTEXT_RULES_PATH


def load_rules() -> dict:
    with open(rules_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    validate_context_rules_config(config)
    return config


def parse_list(text: str) -> list[str]:
    return [item.strip() for item in str(text or "").split(",") if item.strip()]


def parse_json_object(text: str, field_name: str) -> dict:
    if text in ("", None) or (isinstance(text, float) and pd.isna(text)):
        return {}
    try:
        value = json.loads(str(text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON: {exc}") from None
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object.")
    return value


def patterns_to_df(config: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"group": group, "patterns": ", ".join(patterns)}
            for group, patterns in sorted(config["label_pattern_sets"].items())
        ]
    )


def rules_to_df(config: dict) -> pd.DataFrame:
    rows = []
    for rule in config["context_rules"]:
        rows.append(
            {
                "name": rule["name"],
                "signals": ", ".join(rule.get("signals", [])),
                "boost": json.dumps(rule.get("boost", {}), sort_keys=True),
                "suppress": json.dumps(rule.get("suppress", {}), sort_keys=True),
                "suppress_labels": json.dumps(rule.get("suppress_labels", {}), sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def dfs_to_config(patterns_df: pd.DataFrame, rules_df: pd.DataFrame) -> dict:
    label_pattern_sets = {}
    for _, row in patterns_df.fillna("").iterrows():
        group = str(row.get("group", "")).strip()
        if not group:
            continue
        label_pattern_sets[group] = parse_list(row.get("patterns", ""))

    context_rules = []
    for _, row in rules_df.fillna("").iterrows():
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        context_rules.append(
            {
                "name": name,
                "signals": parse_list(row.get("signals", "")),
                "boost": parse_json_object(row.get("boost", "{}"), f"{name}.boost"),
                "suppress": parse_json_object(row.get("suppress", "{}"), f"{name}.suppress"),
                "suppress_labels": parse_json_object(row.get("suppress_labels", "{}"), f"{name}.suppress_labels"),
            }
        )

    config = {"label_pattern_sets": label_pattern_sets, "context_rules": context_rules}
    validate_context_rules_config(config)
    return config


def save_config(config: dict) -> None:
    validate_context_rules_config(config)
    with open(rules_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


try:
    current_config = load_rules()
except Exception as exc:
    st.error(f"Could not load {rules_path.name}: {exc}")
    st.stop()

st.info(f"Rules file: {rules_path}")

upload_col, download_col = st.columns(2)
with upload_col:
    uploaded_json = st.file_uploader("Import replacement rules JSON", type=["json"])
    if uploaded_json is not None:
        try:
            imported = json.loads(uploaded_json.getvalue().decode("utf-8"))
            validate_context_rules_config(imported)
            st.session_state["context_rules_imported_config"] = imported
            st.success("Imported JSON is valid. Review it below, then save.")
        except Exception as exc:
            st.error(f"Imported rules are invalid: {exc}")
with download_col:
    st.download_button(
        "Download current rules JSON",
        data=json.dumps(current_config, indent=2).encode("utf-8"),
        file_name="context_rules.json",
        mime="application/json",
        width="stretch",
    )

editing_config = st.session_state.get("context_rules_imported_config", current_config)

st.markdown("**Label Pattern Groups**")
patterns_df = st.data_editor(
    patterns_to_df(editing_config),
    width="stretch",
    num_rows="dynamic",
    hide_index=True,
    column_config={
        "group": st.column_config.TextColumn("Group", required=True),
        "patterns": st.column_config.TextColumn("Comma-separated label patterns", width="large"),
    },
)

st.markdown("**Context Rules**")
rules_df = st.data_editor(
    rules_to_df(editing_config),
    width="stretch",
    num_rows="dynamic",
    hide_index=True,
    column_config={
        "name": st.column_config.TextColumn("Rule name", required=True),
        "signals": st.column_config.TextColumn("Comma-separated context signals", width="large"),
        "boost": st.column_config.TextColumn("Boost JSON", width="medium"),
        "suppress": st.column_config.TextColumn("Suppress JSON", width="medium"),
        "suppress_labels": st.column_config.TextColumn("Suppress labels JSON", width="medium"),
    },
)

try:
    proposed_config = dfs_to_config(patterns_df, rules_df)
    st.success("Edited rules are valid.")
except Exception as exc:
    proposed_config = None
    st.error(f"Edited rules are invalid: {exc}")

with st.expander("Structured JSON preview", expanded=False):
    st.code(json.dumps(proposed_config or editing_config, indent=2), language="json")

save_col, reset_col = st.columns([1, 1])
with save_col:
    if st.button("Save rules to repo file", type="primary", width="stretch", disabled=proposed_config is None):
        save_config(proposed_config)
        st.session_state.pop("context_rules_imported_config", None)
        st.cache_data.clear()
        st.success(f"Saved {rules_path.name}. Semantic classifications will use these rules on the next run.")
with reset_col:
    if st.button("Reload from disk", width="stretch"):
        st.rerun()
