"""
graphic_logs.py
Renders interactive graphic geological logs.
"""
import streamlit as st
import plotly.graph_objects as go
from config import COAL_COLOR, COAL_LINE_COLOR, LITHO_COLORS
from utils import generate_seam_colors

def render_graphic_logs(data_model) -> None:
    """Generates a graphical strip log for a selected borehole."""
    st.markdown("## 📜 Graphic Logs")
    
    bhid_list = data_model.litho['BHID'].dropna().unique()
    if len(bhid_list) == 0:
        st.warning("No lithology data available.")
        return
        
    selected_bhid = st.selectbox("Select Borehole ID", sorted(bhid_list))
    
    log_data = data_model.litho[data_model.litho['BHID'] == selected_bhid].copy()
    
    if log_data.empty:
        st.info("No lithology intervals found for this borehole.")
        return
        
    seam_colors = generate_seam_colors(data_model.seams)
    
    fig = go.Figure()
    
    for _, row in log_data.iterrows():
        is_coal = row.get('IS_COAL', False)
        litho_name = str(row.get('LITHO', 'Unknown'))
        seam_name = str(row.get('SEAM', ''))
        
        # Determine color logic dynamically
        if is_coal:
            fill_color = seam_colors.get(seam_name, COAL_COLOR)
            line_color = COAL_LINE_COLOR
            line_width = 2
            hover_text = f"<b>Seam: {seam_name}</b><br>Depth: {row['FROM']} - {row['TO']}m<br>Width: {row['WIDTH']}m"
        else:
            fill_color = LITHO_COLORS.get(litho_name, '#D3D3D3')
            line_color = '#A9A9A9'
            line_width = 1
            hover_text = f"<b>{litho_name}</b><br>Depth: {row['FROM']} - {row['TO']}m<br>Width: {row['WIDTH']}m"
            
        fig.add_trace(go.Bar(
            y=[f"BH: {selected_bhid}"], 
            x=[row['WIDTH']],
            base=[row['FROM']],
            orientation='h',
            marker=dict(
                color=fill_color,
                line=dict(color=line_color, width=line_width)
            ),
            name=seam_name if is_coal else litho_name,
            hoverinfo='text',
            hovertext=hover_text,
            showlegend=False
        ))

    # Reverse X axis so zero is at the top (standard for depth logs)
    fig.update_layout(
        title=f"Graphic Log - {selected_bhid}",
        xaxis=dict(title="Depth (m)", autorange="reversed"),
        yaxis=dict(title=""),
        height=300,
        barmode='stack',
        plot_bgcolor='white',
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detail Table
    st.markdown(f"**Lithology Details for {selected_bhid}**")
    st.dataframe(log_data[['FROM', 'TO', 'WIDTH', 'LITHO', 'SEAM']], use_container_width=True)