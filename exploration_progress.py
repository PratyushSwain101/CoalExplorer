"""
exploration_progress.py
Charts for drilling progress and metrics.
"""
import streamlit as st
import plotly.express as px

def render_exploration_progress(data_model) -> None:
    st.markdown("## 🏗️ Exploration Progress")
    
    collar = data_model.collar
    
    # Progress by Status
    status_counts = collar['STATUS'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']
    
    fig = px.bar(status_counts, x='Status', y='Count', color='Status', 
                 title="Borehole Status Distribution")
    st.plotly_chart(fig, use_container_width=True)