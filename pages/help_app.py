import streamlit as st

st.set_page_config(page_title="Help", layout="wide")
st.title("Help / Reference")

st.header("Dashboard overview")
st.write("This dashboard compares offline YAMNet predictions with uploaded annotations.")
st.markdown("[Link to Offline Classification Github](https://github.com/Repair-Redress/Offline_Acoustic_Classification)")

st.header("Model")
st.write("YAMNet is an AudioSet-trained audio event classifier.")
st.markdown("[Link to Model Overview](https://www.tensorflow.org/hub/tutorials/yamnet)")

st.header("AudioSet Ontology")
st.markdown("AudioSet is a hierarchical ontology of audio events.")
st.markdown("[Link to AudioSet Ontology Overview](https://research.google.com/audioset/ontology/index.html)")

st.header("Parameter definitions")
with st.expander("sample_hz"):
    st.write("Predictions per second.")
with st.expander("avg_frames"):
    st.write("Temporal smoothing window over frames.")
with st.expander("channel_mode"):
    st.write("How stereo channels are handled." \
             "Stereo audio can be handled as two separate channels or the average of the channels."
            )

st.divider() 
with st.expander("Metric definitions", expanded=True): 
    with st.expander("Frames compared", expanded=False): 
        st.write( "How many frame-to-frame comparisons were made. " 
                 "This equals the number of unique frame times in the real-time dataset, " 
                 "because the code iterates over RT frames and matches each one to the nearest OFFLINE frame." 
                )
    with st.expander("Avg same in Top-10", expanded=False): 
        st.write( "For each matched pair of frames, take the Top-10 label sets from RT and OFFLINE and count " 
                 "how many labels overlap. This metric is the average overlap count across all compared frames " 
                 "(range: 0 to 10)." 
                ) 
    with st.expander("Top-1 match rate", expanded=False): 
        st.write( "The fraction of matched frames where the highest-scoring (rank=1) label is exactly the same " 
                 "between RT and OFFLINE. Reported as a percentage."
                ) 
    with st.expander("Top-2 match rate", expanded=False): 
        st.write( "The fraction of matched frames where the second-highest (rank=2) label is exactly the same "
                 "between RT and OFFLINE. Reported as a percentage." 
                )
    with st.expander("Avg Jaccard(Top-10)", expanded=False): 
        st.write( "For each matched pair of frames, compute the Jaccard similarity between the two Top-10 label sets." ) 
        st.latex(r"J = \frac{|A \cap B|}{|A \cup B|}") 
        st.write("Then average across frames. Range: 0 to 1 (higher means more similar sets).") 
    with st.expander("Avg |Δscore| (common)", expanded=False): 
        st.write( "For each matched pair of frames, look only at labels that appear in both Top-10 sets. " 
                 "For those common labels, compute the absolute score difference |s_RT - s_OFF|, average within " 
                 "the frame, then average across frames. Lower means better agreement on shared labels’ confidence scores." 
                ) 
    with st.expander("Std dev of per-label mean scores (Top-10 approx)", expanded=False):
        st.write( "For each label, compute its mean score only over frames where it appears in the Top-10 " 
                 "Then compute the standard deviation across labels. Higher values mean the model" 
                 "concentrates probability mass more unevenly across labels; lower values mean label" 
                 "scores are more uniformly distributed (within the Top-K approximation)."
                )