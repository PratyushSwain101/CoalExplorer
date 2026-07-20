"""
app.py
Main entry point for the Streamlit Coal Exploration Dashboard.
"""
import streamlit as st
from config import APP_TITLE, LAYOUT
from loader import ProjectData
from dashboard import render_dashboard
from map import render_map
from graphic_logs import render_graphic_logs
from statistics import render_statistics
from borehole_details import render_borehole_details
from exploration_progress import render_exploration_progress

st.set_page_config(page_title=APP_TITLE, layout=LAYOUT)

@st.cache_resource
def load_project_data(collar_file, boundary_file, litho_file) -> ProjectData:
    """Caches the instantiation of the master data model to prevent recalculation."""
    return ProjectData(collar_file, boundary_file, litho_file)

def main():
    st.sidebar.title("⛰️ Coal Explorer Pro")
    st.sidebar.markdown("Upload project CSVs to dynamically generate the dashboard.")
    
    # File uploaders
    st.sidebar.markdown("### Data Upload")
    collar_file = st.sidebar.file_uploader("Upload COLLAR.csv", type=['csv'])
    boundary_file = st.sidebar.file_uploader("Upload BOUNDARY.csv", type=['csv'])
    litho_file = st.sidebar.file_uploader("Upload LITHO.csv", type=['csv'])
    
    if collar_file and boundary_file and litho_file:
        try:
            # Initialize the master data model
            data_model = load_project_data(collar_file, boundary_file, litho_file)
            
            # Application Navigation
            st.sidebar.markdown("### Navigation")
            pages = {
                "Dashboard": render_dashboard,
                "Map & Spatial": render_map,
                "Borehole Details": render_borehole_details,
                "Graphic Logs": render_graphic_logs,
                "Coal Statistics": render_statistics,
                "Exploration Progress": render_exploration_progress,
            }
            
            selection = st.sidebar.radio("Go to", list(pages.keys()))
            
            # Execute selected page function, passing the cached master data model
            pages[selection](data_model)
            
        except Exception as e:
            st.error(f"Error processing files. Please ensure CSV schemas match requirements. Details: {e}")
    else:
        st.info("👈 Please upload COLLAR, BOUNDARY, and LITHO CSV files to initialize the project engine.")
        
        # Placeholder styling while waiting for files
        st.markdown(
            """
            ### Welcome to Coal Explorer Pro
            A completely data-driven, modular exploration dashboard.
            *   **No Hardcoded References:** Adapts to any project.
            *   **Automated Analytics:** Calculates net coal, detects seams, creates depth intervals.
            *   **Architectural Efficiency:** Built around a single, cached `ProjectData` model.
            
            Ready to process up to 1000+ boreholes instantly.
            """
        )

if __name__ == "__main__":
    main()