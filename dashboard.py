"""
dashboard.py
Renders the high-level KPI dashboard.
"""
import streamlit as st
import pandas as pd

def render_dashboard(data_model) -> None:
    """Renders main dashboard KPIs and progress bars."""
    st.markdown("## 📊 Exploration Overview")
    
    collar = data_model.collar
    
    total_bh = len(collar)
    closed = len(collar[collar['STATUS'] == 'Closed'])
    running = len(collar[collar['STATUS'] == 'Running'])
    pending = len(collar[collar['STATUS'] == 'Pending'])
    
    total_meterage_prop = collar['DEPTH'].sum() if 'DEPTH' in collar.columns else 0
    drilled = collar[collar['STATUS'] == 'Closed']['DEPTH'].sum() + \
              (collar[collar['STATUS'] == 'Running']['DEPTH'].sum() * 0.5) # Estimate for running
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Proposed BH", total_bh)
    col2.metric("Closed Boreholes", closed)
    col3.metric("Running Rigs", running)
    col4.metric("Pending Locations", pending)
    
    st.markdown("---")
    
    col5, col6, col7 = st.columns(3)
    col5.metric("Total Proposed Meterage (m)", f"{total_meterage_prop:,.2f}")
    col6.metric("Estimated Drilled (m)", f"{drilled:,.2f}")
    
    completion_pct = (drilled / total_meterage_prop * 100) if total_meterage_prop > 0 else 0
    col7.metric("Completion Percentage", f"{completion_pct:.1f}%")
    
    st.progress(min(completion_pct / 100.0, 1.0))
    
    if data_model.seams:
        st.markdown("### ⛏️ Coal Highlights")
        c1, c2, c3 = st.columns(3)
        avg_coal = collar[collar['TOTAL_COAL'] > 0]['TOTAL_COAL'].mean()
        max_coal = collar['TOTAL_COAL'].max()
        deepest = collar['DEPTH'].max() if 'DEPTH' in collar.columns else 0
        
        c1.metric("Average Cumulative Coal Thickness (m)", f"{avg_coal:.2f}" if pd.notna(avg_coal) else "0")
        c2.metric("Max Cumulative Coal Thickness (m)", f"{max_coal:.2f}")
        c3.metric("Deepest Borehole (m)", f"{deepest:.2f}")