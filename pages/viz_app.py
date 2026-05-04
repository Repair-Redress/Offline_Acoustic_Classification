import json
import pandas as pd
import streamlit as st
import plotly.express as px

@st.cache_data
def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            t = float(obj["t_sec"])
            top = obj["top"]
            # store top-k in columns for easy plotting
            for rank, item in enumerate(top):
                rows.append({
                    "t_sec": t,
                    "rank": rank + 1,
                    "label": item["label"],
                    "score": float(item["score"]),
                })
    df = pd.DataFrame(rows).sort_values(["t_sec", "rank"])
    return df

def make_segments(top1_df, min_score=0.0):
    # Collapse consecutive frames with same label into segments
    top1_df = top1_df[top1_df["score1"] >= min_score].copy()
    if top1_df.empty:
        return pd.DataFrame(columns=["label","t_start","t_end","dur_s","mean_score"])
    segs = []
    cur_label = None
    t_start = None
    scores = []
    prev_t = None
    for _, r in top1_df.iterrows():
        t, lab, sc = r["t_sec"], r["top1"], r["score1"]
        if cur_label is None:
            cur_label, t_start, prev_t, scores = lab, t, t, [sc]
            continue
        # new segment if label changes OR time jumps (handles gaps)
        if lab != cur_label or (t - prev_t) > 2.0*(top1_df["t_sec"].diff().median() or 1e-6):
            segs.append((cur_label, t_start, prev_t, prev_t - t_start, sum(scores)/len(scores)))
            cur_label, t_start, scores = lab, t, [sc]
        else:
            scores.append(sc)
        prev_t = t
    segs.append((cur_label, t_start, prev_t, prev_t - t_start, sum(scores)/len(scores)))
    out = pd.DataFrame(segs, columns=["label","t_start","t_end","dur_s","mean_score"])
    return out.sort_values(["t_start"])

st.title("Offline YAMNet / AudioSet Timeline Viewer")

pred_path = st.text_input("predictions.jsonl path", "predictions.jsonl")
df = load_jsonl(pred_path)

# derive top-1 per frame
top1 = df[df["rank"] == 1][["t_sec","label","score"]].rename(columns={"label":"top1","score":"score1"}).sort_values("t_sec")

threshold = st.slider("Min confidence (top-1)", 0.0, 1.0, 0.3, 0.01)

st.subheader("Top-1 label over time")
top1_f = top1[top1["score1"] >= threshold].copy()
fig = px.scatter(top1_f, x="t_sec", y="top1", color="top1", hover_data=["score1"], height=450)
fig.update_layout(showlegend=False, xaxis_title="Time (s)", yaxis_title="Top-1 label")
st.plotly_chart(fig, width="stretch")

st.subheader("Event segments (collapsed runs)")
segs = make_segments(top1, min_score=threshold)
st.dataframe(segs, width="stretch")

st.subheader("Inspect predictions at a time")
tmin, tmax = float(top1["t_sec"].min()), float(top1["t_sec"].max())
t_pick = st.slider("Time (s)", tmin, tmax, tmin)
# nearest frame
idx = (top1["t_sec"] - t_pick).abs().idxmin()
t_near = float(top1.loc[idx, "t_sec"])
st.write(f"Nearest frame: t = {t_near:.3f}s, top1 = {top1.loc[idx,'top1']} ({top1.loc[idx,'score1']:.3f})")

frame = df[df["t_sec"] == t_near].sort_values("rank")[["rank","label","score"]]
st.dataframe(frame, width="stretch")
