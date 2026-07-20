"""
map.py
Renders the interactive exploration map.
"""
import streamlit as st
import plotly.graph_objects as go
from config import STATUS_COLORS

def render_map(data_model) -> None:
    """Plots the lease boundary and boreholes dynamically."""
    st.markdown("## 🗺️ Plan View & Status Map")
    
    collar = data_model.collar
    boundary = data_model.boundary
    
    fig = go.Figure()

    # Plot Boundary Polygon
    if not boundary.empty and 'X' in boundary.columns and 'Y' in boundary.columns:
        fig.add_trace(go.Scatter(
            x=boundary['X'].tolist() + [boundary['X'].iloc[0]], 
            y=boundary['Y'].tolist() + [boundary['Y'].iloc[0]],
            fill='toself',
            fillcolor='rgba(135, 206, 250, 0.2)',
            line=dict(color='blue', width=2),
            name='Lease Boundary',
            hoverinfo='none'
        ))

    # Plot Boreholes grouped by Status
    for status in collar['STATUS'].unique():
        subset = collar[collar['STATUS'] == status]
        color = STATUS_COLORS.get(status, STATUS_COLORS['Unknown'])
        
        hover_text = [
            f"<b>{row['DISPLAY_ID']}</b><br>"
            f"Status: {row['STATUS']}<br>"
            f"Depth: {row.get('DEPTH', 'N/A')} m<br>"
            f"RL: {row.get('RL', 'N/A')} m<br>"
            f"Total Coal: {row.get('TOTAL_COAL', 0):.2f} m<br>"
            f"Seams: {row.get('NO_OF_SEAMS', 0)}"
            for _, row in subset.iterrows()
        ]
        
        fig.add_trace(go.Scatter(
            x=subset['X'],
            y=subset['Y'],
            mode='markers+text',
            marker=dict(size=10, color=color, line=dict(width=1, color='black')),
            text=subset['DISPLAY_ID'],
            textposition="top center",
            textfont=dict(size=9),
            name=status,
            hoverinfo='text',
            hovertext=hover_text
        ))

    fig.update_layout(
        title="Exploration Plan View",
        xaxis_title="Easting (X)",
        yaxis_title="Northing (Y)",
        yaxis=dict(scaleanchor="x", scaleratio=1),  # Forces equal aspect ratio for spatial accuracy
        legend_title="Status",
        height=700,
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor='white'
    )
    
    # Add minor grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    st.plotly_chart(fig, use_container_width=True)