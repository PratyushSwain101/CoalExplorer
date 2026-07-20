"""
borehole_details.py
Tabular views and data grids for boreholes.
"""
import streamlit as st

def render_borehole_details(data_model) -> None:
    st.markdown("## 📋 Borehole Database")
    
    status_filter = st.multiselect("Filter by Status", data_model.collar['STATUS'].unique())
    
    filtered_df = data_model.collar
    if status_filter:
        filtered_df = filtered_df[filtered_df['STATUS'].isin(status_filter)]
        
    st.dataframe(filtered_df, use_container_width=True, height=500)
    
    st.download_button(
        label="Download Filtered CSV",
        data=filtered_df.to_csv(index=False).encode('utf-8'),
        file_name='filtered_boreholes.csv',
        mime='text/csv'
    )