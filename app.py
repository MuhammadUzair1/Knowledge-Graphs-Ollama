import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    initial_sidebar_state="expanded"
)

pg = st.navigation(
    [
        st.Page("pgs/home.py", title="Home", icon="🏠"),
        st.Page("pgs/upload.py", title="Upload File", icon="🗳️"), 
        st.Page("pgs/chat.py", title="Chat with Graph" , icon="🦜"),
        st.Page("pgs/display.py", title="Display Graph", icon="🕸️"), 
        st.Page("pgs/config.py", title="Settings", icon="⚙️"),
    ]
)

pg.run()
