"""
statistics.py
Renders coal statistics and distribution data.
"""
import streamlit as st
import pandas as pd
import plotly.express as px

def render_statistics(data_model) -> None:
    """Calculates and renders coal statistics."""
    st.markdown("## 📈 Coal Statistics")
    
    litho = data_model.litho
    coal_data = litho[litho['IS_COAL'] == True]
    
    if coal_data.empty:
        st.warning("No coal data detected in the provided files.")
        return
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Seam-wise Averages")
        seam_stats = coal_data.groupby('SEAM').agg(
            Count=('SEAM', 'count'),
            Avg_Thickness=('WIDTH', 'mean'),
            Max_Thickness=('WIDTH', 'max'),
            Min_Thickness=('WIDTH', 'min')
        ).reset_index().round(2)
        st.dataframe(seam_stats, use_container_width=True)
        
    with col2:
        st.markdown("### Coal Depth Distribution")
        depth_dist = coal_data.groupby('DEPTH_INTERVAL')['WIDTH'].sum().reset_index()
        fig = px.pie(depth_dist, values='WIDTH', names='DEPTH_INTERVAL', 
                     title="Total Coal Volume by Depth Interval", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)