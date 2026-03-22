import streamlit as st

st.set_page_config(
    page_title="Audio Dashboard",
    layout="wide",
)

pg = st.navigation(
    [
        st.Page("pages/compare_app.py", title="Comparison Dashboard", icon=":material/compare_arrows:"),
        st.Page("pages/help_app.py", title="Help / Reference", icon=":material/help:"),
    ],
    position="sidebar",
)

pg.run()