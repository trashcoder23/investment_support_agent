import os
import streamlit as st

def apply_theme():
    """Injects custom CSS from style.css into the Streamlit app to create a premium FinTech UI."""
    css_file = os.path.join(os.path.dirname(__file__), 'style.css')
    
    if os.path.exists(css_file):
        with open(css_file, 'r', encoding='utf-8') as f:
            css = f.read()
            st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Theme file (style.css) not found.")
