import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import random
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, A3
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
import streamlit.components.v1 as components

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & CONSTANTS
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Exploration Monitoring Dashboard", layout="wide")

STATUS_COLORS = {
    'Closed': '#00CC96',       # Green
    'Running': '#FFA15A',      # Orange
    'Under Shifting': '#FFFF00', # Blue
    'Breakdown': '#E74C3C',    # Red
    'Pending': '#AB63FA',      # Purple/Grey
    'Unknown': '#B6E880'
}

LITHO_COLORS = {
    'Soil': '#8B4513',
    'Sandstone': '#F4A460',
    'Shale': '#708090',
    'Siltstone': '#A0522D',
    'Mudstone': '#696969',
    'Carbonaceous Shale': '#2F4F4F'
}

COAL_COLOR = '#000000'
PLOT_TEXT_COLOR = 'black'

def generate_seam_colors(seams):
    cleaned_seams = sorted([str(s) for s in seams if s and str(s).strip().lower() != 'nan'])
    colors = px.colors.qualitative.Dark24 + px.colors.qualitative.Alphabet
    seam_color_map = {}
    for i, seam in enumerate(cleaned_seams):
        seam_color_map[seam] = colors[i % len(colors)]
    return seam_color_map

# -----------------------------------------------------------------------------
# 2. DATA PROCESSING & COMPOSITING ENGINE
# -----------------------------------------------------------------------------

def load_csv_safe(file_obj) -> pd.DataFrame:
    try:
        file_obj.seek(0)
        return pd.read_csv(file_obj, encoding='utf-8')
    except UnicodeDecodeError:
        file_obj.seek(0)
        return pd.read_csv(file_obj, encoding='latin1')


@st.cache_data(show_spinner=False)
def process_collar(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [col.upper().strip() for col in df.columns]
    
    # Drop rows that are completely empty (fixes the 51 vs 48 issue)
    df.dropna(how='all', inplace=True)
    
    if 'STATUS' in df.columns:
        df['STATUS'] = df['STATUS'].fillna('Pending').astype(str).str.title()
    else:
        df['STATUS'] = 'Unknown'
        
    if 'ED' not in df.columns:
        df['ED'] = df.get('DEPTH', 0)
        
    df['ED'] = pd.to_numeric(df['ED'], errors='coerce').fillna(0)
    df['DEPTH'] = pd.to_numeric(df.get('DEPTH', 0), errors='coerce').fillna(0)
    df['RL'] = pd.to_numeric(df.get('RL', 0), errors='coerce').fillna(0)
    
    for date_col in ['CLOSING DATE', 'DOC']:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
        else:
            df[date_col] = pd.NaT
            
    # Ensure text columns (including SID) exist to avoid KeyError later
    # Ensure text columns exist (Added 'TYPE' to the list)
    for text_col in ['HQ', 'RIG NO', 'RIG MODEL', 'GPL STATUS', 'REMARKS', 'POINT', 'SID', 'TYPE']:
        if text_col not in df.columns:
            df[text_col] = 'CMPDI' if text_col == 'TYPE' else ''
            
    # Clean the TYPE column to ensure exact matching (uppercase, no spaces)
    df['TYPE'] = df['TYPE'].fillna('CMPDI').astype(str).str.strip().str.upper()
        
    def make_display_id(row):
        bhid = str(row.get('BHID', '')).strip()
        point = str(row.get('POINT', '')).strip()
        
        if bhid.lower() in ['nan', 'none', 'null', '']: bhid = ''
        if point.lower() in ['nan', 'none', 'null', '']: point = ''
            
        if bhid and point: return f"{bhid}/{point}"
        elif bhid: return bhid
        elif point: return point
        else: return "Unknown"
            
    df['DISPLAY_ID'] = df.apply(make_display_id, axis=1)
    
    # Filter out empty rows that couldn't generate a DISPLAY_ID
    df = df[df['DISPLAY_ID'] != "Unknown"].reset_index(drop=True)
    
    return df


@st.cache_data(show_spinner=False)
def process_boundary(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [col.upper().strip() for col in df.columns]
    df.dropna(subset=['X', 'Y'], inplace=True)
    return df


@st.cache_data(show_spinner=False)
def process_litho(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [col.upper().strip() for col in df.columns]
    
    df['SEAM'] = df.get('SEAM', '').fillna('')
    df['LITHO'] = df.get('LITHO', 'Unknown').fillna('Unknown')
    df['FROM'] = pd.to_numeric(df['FROM'], errors='coerce')
    df['TO'] = pd.to_numeric(df['TO'], errors='coerce')
    
    df.dropna(subset=['BHID', 'FROM', 'TO'], inplace=True)
    df = df.sort_values(['BHID', 'FROM']).reset_index(drop=True)
    
    df['shift_group'] = (
        (df['BHID'] != df['BHID'].shift()) |
        (df['LITHO'] != df['LITHO'].shift()) |
        (df['SEAM'] != df['SEAM'].shift())
    ).cumsum()
    
    agg_dict = {
        'BHID': 'first',
        'FROM': 'min',
        'TO': 'max',
        'LITHO': 'first',
        'SEAM': 'first'
    }
    
    if 'RECOVERY' in df.columns:
        df['RECOVERY'] = pd.to_numeric(df['RECOVERY'], errors='coerce')
        agg_dict['RECOVERY'] = 'sum' 
        
    df_merged = df.groupby('shift_group').agg(agg_dict).reset_index(drop=True)
    
    df_merged['WIDTH'] = df_merged['TO'] - df_merged['FROM']
    df_merged['IS_COAL'] = df_merged['SEAM'].astype(str).str.strip() != ''
    df_merged['MID_DEPTH'] = (df_merged['FROM'] + df_merged['TO']) / 2
    
    return df_merged

# -----------------------------------------------------------------------------
# 3. REPORTLAB PDF DRAWING ENGINE
# -----------------------------------------------------------------------------

def draw_pattern_rect(c, x, y, w, h, litho, is_coal):
    c.saveState()
    path = c.beginPath()
    path.rect(x, y, w, h)
    c.clipPath(path, stroke=0)
    
    base_color = colors.white
    litho_u = str(litho).upper()
    
    if is_coal: 
        base_color = colors.black
    elif 'CARB' in litho_u: 
        base_color = colors.Color(0.8, 0.8, 0.8) 
    elif 'SHALE' in litho_u: 
        base_color = colors.Color(0.95, 0.95, 0.95)
    elif 'SAND' in litho_u or 'SST' in litho_u: 
        base_color = colors.Color(1, 1, 0.85)
    elif 'MUD' in litho_u:
        base_color = colors.Color(0.9, 0.92, 0.9)
    
    c.setFillColor(base_color)
    c.rect(x, y, w, h, fill=1, stroke=1)

    if is_coal:
        c.setFillColor(colors.black)
        c.rect(x, y, w, h, fill=1, stroke=0)
        
    elif 'SAND' in litho_u or 'SST' in litho_u:
        c.setFillColor(colors.black)
        rows, cols = int(h / (1.5*mm)), int(w / (1.5*mm))
        for r in range(rows + 2):
            for col in range(cols + 2):
                dot_x = x + (col * 1.5*mm) + random.uniform(-0.5, 0.5)
                dot_y = y + (r * 1.5*mm) + random.uniform(-0.5, 0.5)
                if random.choice([True, False]):
                    c.circle(dot_x, dot_y, 0.2, fill=1, stroke=0)
                else:
                    c.circle(dot_x, dot_y, 0.4, fill=0, stroke=1)
                    
    elif 'SHALE' in litho_u and 'CARB' not in litho_u:
        c.setStrokeColor(colors.black)
        c.setLineWidth(0.5)
        c.setDash([4, 2])
        curr_y = y + 1*mm
        while curr_y < y + h:
            c.line(x, curr_y, x + w, curr_y)
            curr_y += 1.5*mm
        c.setDash([])
        
    elif 'CARB' in litho_u:
        c.setStrokeColor(colors.black)
        c.setLineWidth(0.8)
        c.setDash([3, 2])
        curr_y = y + 1*mm
        while curr_y < y + h:
            c.line(x, curr_y, x + w, curr_y)
            curr_y += 1.5*mm
        c.setDash([])
        c.setLineWidth(0.3)
        curr_x = x + 2*mm
        while curr_x < x + w:
            c.line(curr_x, y, curr_x, y+h)
            curr_x += 3*mm
            
    elif 'SOIL' in litho_u or 'ALLUVIUM' in litho_u or 'DIRT' in litho_u:
        c.setStrokeColor(colors.black)
        c.setLineWidth(0.5)
        step_y, step_x = 3 * mm, 3 * mm
        rows, cols = int(h / step_y), int(w / step_x)
        for r in range(rows + 2):
            for col in range(cols + 2):
                sx = x + (col * step_x) + random.uniform(-1, 1)
                sy = y + (r * step_y) + random.uniform(-1, 1)
                p = c.beginPath()
                p.moveTo(sx, sy)
                p.lineTo(sx + 1.2, sy - 2); p.lineTo(sx + 2.4, sy)
                c.drawPath(p, stroke=1, fill=0)
                
    elif 'MUD' in litho_u:
        c.setStrokeColor(colors.black)
        c.setLineWidth(0.3)
        curr_y = y + 1*mm
        while curr_y < y + h:
            c.line(x, curr_y, x + w, curr_y)
            curr_y += 1.0*mm
            
    c.restoreState()

def generate_graphic_log_pdf(df_litho, df_collar, selected_bhids):
    buffer = io.BytesIO()
    
    # 1:500 Scale (1 meter = 2 mm)
    SCALE_FACTOR = 2.0 * mm 
    MARGIN_TOP, MARGIN_BOTTOM, MARGIN_LEFT, MARGIN_RIGHT = 3*cm, 3*cm, 4*cm, 2*cm
    COL_WIDTH = 1.0 * cm 
    
    max_depth = df_collar[df_collar['BHID'].isin(selected_bhids)]['DEPTH'].max()
    if pd.isna(max_depth) or max_depth <= 0: max_depth = 100
    
    REQUIRED_HEIGHT = (max_depth * SCALE_FACTOR) + MARGIN_TOP + MARGIN_BOTTOM
    PAGE_HEIGHT = max(A3[1], REQUIRED_HEIGHT)
    
    num_bhs = len(selected_bhids)
    REQUIRED_WIDTH = MARGIN_LEFT + MARGIN_RIGHT + (num_bhs * 4 * cm)
    PAGE_WIDTH = max(A3[0], REQUIRED_WIDTH)
    SPACING = (PAGE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT - (num_bhs * COL_WIDTH)) / num_bhs if num_bhs > 1 else 6*cm
    
    c = canvas.Canvas(buffer, pagesize=(PAGE_WIDTH, PAGE_HEIGHT))
    Y_REF = PAGE_HEIGHT - MARGIN_TOP

    legend_x, legend_y = PAGE_WIDTH - 4*cm, PAGE_HEIGHT - 2*cm
    c.setFont("Helvetica-Bold", 10)
    c.drawString(legend_x, legend_y + 0.5*cm, "LITHOLOGY LEGEND")
    
    unique_lithos = set(df_litho[~df_litho['IS_COAL']]['LITHO'])
    has_coal = df_litho[df_litho['IS_COAL']].shape[0] > 0
    legend_items = sorted(list(unique_lithos))
    if has_coal: legend_items.insert(0, 'COAL SEAM')
    
    curr_ly = legend_y
    for item in legend_items:
        draw_pattern_rect(c, legend_x, curr_ly - 0.6*cm, 1*cm, 0.6*cm, item, item == 'COAL SEAM')
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 8)
        c.drawString(legend_x + 1.2*cm, curr_ly - 0.4*cm, item)
        curr_ly -= 0.8*cm

    tick_step_major = 10 if max_depth > 100 else (5 if max_depth > 50 else 2)
    tick_step_minor = 2 if max_depth > 100 else 1

    for i, bhid in enumerate(selected_bhids):
        x_origin = MARGIN_LEFT + (i * (COL_WIDTH + SPACING))
        
        collar_row = df_collar[df_collar['BHID'] == bhid]
        if collar_row.empty: continue
        
        collar_rl = collar_row.iloc[0]['RL']
        total_depth = collar_row.iloc[0]['DEPTH']
        display_id = collar_row.iloc[0]['DISPLAY_ID'] if 'DISPLAY_ID' in collar_row.columns else bhid
        if pd.isna(total_depth) or total_depth <= 0: total_depth = 100
        
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 11)
        c.drawCentredString(x_origin + COL_WIDTH/2, Y_REF + 2.0*cm, display_id)
        c.setFont("Helvetica", 9)
        c.drawCentredString(x_origin + COL_WIDTH/2, Y_REF + 1.5*cm, f"RL: {collar_rl:.2f}m")
        c.drawCentredString(x_origin + COL_WIDTH/2, Y_REF + 1.1*cm, f"TD: {total_depth:.2f}m")
        
        c.setLineWidth(1)
        c.line(x_origin, Y_REF, x_origin, Y_REF - (total_depth * SCALE_FACTOR))
        c.line(x_origin + COL_WIDTH, Y_REF, x_origin + COL_WIDTH, Y_REF - (total_depth * SCALE_FACTOR))
        
        for d in range(0, int(total_depth) + 1):
            y_pos = Y_REF - (d * SCALE_FACTOR)
            if d % tick_step_major == 0:
                c.line(x_origin - 3*mm, y_pos, x_origin, y_pos)
                c.setFont("Helvetica", 7)
                c.drawRightString(x_origin - 4*mm, y_pos - 2, f"{d}")
            elif d % tick_step_minor == 0:
                c.line(x_origin - 1.5*mm, y_pos, x_origin, y_pos)

        bh_data = df_litho[df_litho['BHID'] == bhid].sort_values('FROM')
        for _, row in bh_data.iterrows():
            if row['WIDTH'] <= 0: continue
            rect_y = Y_REF - (row['TO'] * SCALE_FACTOR)
            rect_h = row['WIDTH'] * SCALE_FACTOR
            draw_pattern_rect(c, x_origin, rect_y, COL_WIDTH, rect_h, row['LITHO'], row['IS_COAL'])
            
            if not row['IS_COAL'] and rect_h > 0.5*cm:
                c.setFillColor(colors.black)
                c.setFont("Helvetica", 6)
                c.drawCentredString(x_origin + (COL_WIDTH / 2), rect_y + (rect_h / 2) - 2, row['LITHO'])

        coal_data = bh_data[bh_data['IS_COAL']].copy()
        if not coal_data.empty:
            coal_data = coal_data.sort_values('FROM')
            coal_data['PREV_TO'] = coal_data.groupby('SEAM')['TO'].shift(1)
            coal_data['NEW_INSTANCE'] = ((coal_data['FROM'] - coal_data['PREV_TO']) >= 0.50).astype(int).fillna(0)
            coal_data['INSTANCE'] = coal_data.groupby('SEAM')['NEW_INSTANCE'].cumsum()

            seam_groups = coal_data.groupby(['SEAM', 'INSTANCE']).agg({
                'FROM': 'min', 'TO': 'max'
            }).reset_index()
            seam_groups['ENVELOPE_WIDTH'] = seam_groups['TO'] - seam_groups['FROM']

            for _, s_row in seam_groups.iterrows():
                bx = x_origin + COL_WIDTH + 2*mm
                bt = Y_REF - (s_row['FROM'] * SCALE_FACTOR)
                bb = Y_REF - (s_row['TO'] * SCALE_FACTOR)
                
                c.setLineWidth(1)
                c.line(bx, bt, bx + 2*mm, bt) 
                c.line(bx + 2*mm, bt, bx + 2*mm, bb)
                c.line(bx, bb, bx + 2*mm, bb)
                
                c.setFillColor(colors.black)
                c.setFont("Helvetica-Bold", 8)
                c.drawString(bx + 4*mm, (bt + bb)/2 - 3, f"{s_row['SEAM']} - {s_row['ENVELOPE_WIDTH']:.2f}m")
                
                c.setFont("Helvetica-Bold", 7)
                top_y, bot_y = bt - 2, bb + 2
                if (bt - bb) < 12: 
                    top_y, bot_y = bt + 3, bb - 7
                
                # Offset shifted to 0.8 cm to clear the y-axis ticks
                c.drawRightString(x_origin - 0.8*cm, top_y, f"{s_row['FROM']:.2f}")
                c.drawRightString(x_origin - 0.8*cm, bot_y, f"{s_row['TO']:.2f}")

        close_y = Y_REF - (total_depth * SCALE_FACTOR) - 1*cm
        c.setLineWidth(1.5)
        c.line(x_origin, close_y + 0.8*cm, x_origin + COL_WIDTH, close_y + 0.8*cm)
        c.setFont("Helvetica-Bold", 9)
        c.drawCentredString(x_origin + COL_WIDTH/2, close_y, f"TD: {total_depth:.2f}m")
    c.save()
    buffer.seek(0)
    return buffer


# -----------------------------------------------------------------------------
# VIEW RENDERING: PLAN VIEW MAP
# -----------------------------------------------------------------------------
def render_map(key_suffix="map"):
    st.markdown("### 🗺️ Status Map")
        
    collar = st.session_state['df_collar']
    boundary = st.session_state['df_boundary']
    
    fig = go.Figure()

    # 1. Plot Boundary
    if not boundary.empty:
        fig.add_trace(go.Scatter(
            x=boundary['X'].tolist() + [boundary['X'].iloc[0]], 
            y=boundary['Y'].tolist() + [boundary['Y'].iloc[0]],
            fill='toself', fillcolor='rgba(135, 206, 250, 0.1)',
            line=dict(color='blue', width=2),
            name='Lease Boundary', hoverinfo='none'
        ))

    y_range = collar['Y'].max() - collar['Y'].min()
    y_offset = (y_range * 0.03) if y_range > 0 else 50

    # Ensure TYPE column exists for safety before filtering
    if 'TYPE' not in collar.columns:
        collar['TYPE'] = 'CMPDI'

    # Split data by TYPE for plotting markers
    cmpdi_collar = collar[collar['TYPE'] == 'CMPDI']
    gsi_collar = collar[collar['TYPE'] == 'GSI']

    # Safely handle NaN values for the Map Text Labels
    def format_map_label(row):
        disp = str(row.get('DISPLAY_ID', 'Unknown'))
    
        rl = row.get('RL')
        rl_str = f"RL: {rl}" if pd.notna(rl) and str(rl).lower().strip() != 'nan' else "RL: N/A"
        
        cl = row.get('DEPTH')
        cl_str = f"CD: {cl}" if pd.notna(cl) and str(cl).lower().strip() != 'nan' else "CD: N/A"
        
        sid = row.get('SID')
        if pd.isna(sid) or str(sid).lower().strip() in ['nan', 'none', '']:
            sid_str = ""
        else:
            try:
                sid_str = f"SID: {int(float(sid))}"
            except (ValueError, TypeError):
                sid_str = f"SID: {sid}"
                
        return f"<b>{disp}</b><br>{rl_str}<br>{cl_str}<br>{sid_str}"

    ordered_statuses = ['Closed', 'Running', 'Under Shifting', 'Breakdown', 'Pending', 'Unknown']
    
    # 2. Plot CMPDI Boreholes (Markers and Labels linked by Legendgroup)
    available_statuses = cmpdi_collar['STATUS'].unique()
    for status in [s for s in ordered_statuses if s in available_statuses]:
        subset = cmpdi_collar[cmpdi_collar['STATUS'] == status]
        if subset.empty: continue
        
        color = STATUS_COLORS.get(status, STATUS_COLORS['Unknown'])
        marker_symbol = 'x' if status == 'Breakdown' else 'circle'
        marker_size = 12 if status == 'Breakdown' else 10
        
        hover_text = [
            f"<b>{row.get('DISPLAY_ID', '')}</b><br>"
            f"Type: CMPDI<br>"
            f"Status: {row.get('STATUS', '')}<br>"
            f"Depth: {row.get('DEPTH', 0)} m / ED: {row.get('ED', 0)} m<br>"
            f"RL: {row.get('RL', 'N/A')} m<br>"
            f"SID: {row.get('SID', 'N/A')} m"
            for _, row in subset.iterrows()
        ]
        
        # Add the points
        fig.add_trace(go.Scatter(
            x=subset['X'], y=subset['Y'], mode='markers',
            marker=dict(size=marker_size, color=color, symbol=marker_symbol, line=dict(width=0.5, color='black')),
            name=status, hoverinfo='text', hovertext=hover_text,
            legendgroup=status  # Ties this trace to the group
        ))
        
        # Add the text labels tied to the SAME group so they toggle together
        subset_labels = subset.apply(format_map_label, axis=1)
        fig.add_trace(go.Scatter(
            x=subset['X'], y=subset['Y'] - y_offset, mode='text',
            text=subset_labels, 
            textposition="bottom center", 
            textfont=dict(size=6, color=PLOT_TEXT_COLOR),
            showlegend=False, hoverinfo='skip',
            legendgroup=status  # Ties this label trace to the marker trace above
        ))

    # 3. Plot GSI Boreholes separately
    if not gsi_collar.empty:
        hover_text_gsi = [
            f"<b>{row.get('DISPLAY_ID', '')}</b><br>"
            f"Type: GSI<br>"
            f"Depth: {row.get('DEPTH', 0)} m<br>"
            f"RL: {row.get('RL', 'N/A')} m<br>"
            f"SID: {row.get('SID', 'N/A')} m"
            for _, row in gsi_collar.iterrows()
        ]
        
        fig.add_trace(go.Scatter(
            x=gsi_collar['X'], y=gsi_collar['Y'], mode='markers',
            marker=dict(size=10, color='#FF0055', symbol='triangle-up', line=dict(width=0.5, color='black')),
            name="GSI Bhs", hoverinfo='text', hovertext=hover_text_gsi,
            legendgroup="GSI Bhs"
        ))
        
        gsi_labels = gsi_collar.apply(format_map_label, axis=1)
        fig.add_trace(go.Scatter(
            x=gsi_collar['X'], y=gsi_collar['Y'] - y_offset, mode='text',
            text=gsi_labels, 
            textposition="bottom center", 
            textfont=dict(size=6, color=PLOT_TEXT_COLOR),
            showlegend=False, hoverinfo='skip',
            legendgroup="GSI Bhs"
        ))
    
    # -------------------------------------------------------------------------
    # 🌟 ENHANCEMENTS: NORTH ARROW, FULL GRID BOX, LEGEND STYLING
    # -------------------------------------------------------------------------
    
    # Add a North Arrow Annotation
    fig.add_annotation(
        text="<b>N<br>▲</b>",
        xref="paper", yref="paper",
        x=0.98, y=0.98,
        showarrow=False,
        font=dict(size=16, color="black"),
        bgcolor="rgba(255, 255, 255, 0.8)",
        borderwidth=1.5,
        borderpad=4
    )

    # Scale bar in the lower-right corner, aligned to the same base level
    scale_x0 = 0.74
    scale_x_mid = 0.83
    scale_x1 = 0.92
    scale_y = 0.08

    fig.add_shape(
        type="line",
        x0=scale_x0, x1=scale_x1, y0=scale_y, y1=scale_y,
        xref="paper", yref="paper",
        line=dict(color="black", width=2),
    )

    for tick_x in [scale_x0, scale_x_mid, scale_x1]:
        fig.add_shape(
            type="line",
            x0=tick_x, x1=tick_x, y0=scale_y, y1=scale_y - 0.012,
            xref="paper", yref="paper",
            line=dict(color="black", width=2.5),
        )

    # fig.add_annotation(
    #     text="SCALE",
    #     x=scale_x0, y=scale_y + 0.018,
    #     xref="paper", yref="paper",
    #     showarrow=False,
    #     font=dict(size=10, color="black"),
    #     xanchor="left",
    #     yanchor="bottom"
    # )

    for x_value, label in [(scale_x0, "0"), (scale_x_mid, "250m"), (scale_x1, "500m")]:
        fig.add_annotation(
            text=label,
            x=x_value, y=scale_y - 0.018,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=12, color="black"),
            xanchor="center",
            yanchor="top"
        )

    fig.update_layout(
        title="Exploration Plan View",
        autosize=True, # Allows the map to naturally shrink/grow based on width and scaleanchor
        margin=dict(l=40, r=40, t=40, b=40),
        # plot_bgcolor='white',
        
        # Borderless Legend
        legend=dict(
            title="Legend",
            # bgcolor="rgba(255, 255, 255, 0.9)",
            borderwidth=0, 
            x=1.01, y=1,
            xanchor="left", yanchor="top"
        ),
        
        # PRIMARY AXES (Bottom and Left)
        xaxis=dict(
            title="Easting (X)",
            showgrid=True, gridwidth=1, gridcolor='LightGray',
            showline=True, linewidth=1.5, linecolor='black',
            ticks='outside', tickcolor='black',
            mirror=False,
            tickformat=".0f" # Removes decimals and prevents 'k' or 'M' abbreviations
        ),
        yaxis=dict(
            title="Northing (Y)",
            scaleanchor="x", scaleratio=1, # Locks aspect ratio
            showgrid=True, gridwidth=1, gridcolor='LightGray',
            showline=True, linewidth=1.5, linecolor='black',
            ticks='outside', tickcolor='black',
            mirror=False,
            tickformat=".0f" # Removes decimals and prevents 'k' or 'M' abbreviations
        ),
        
        # SECONDARY AXES (Top and Right for matching ticks & labels)
        xaxis2=dict(
            overlaying='x', side='top', matches='x',
            showgrid=False, showline=True, linewidth=1.5, linecolor='black',
            ticks='outside', tickcolor='black', showticklabels=True,
            tickformat=".0f"
        ),
        yaxis2=dict(
            overlaying='y', side='right', matches='y',
            showgrid=False, showline=True, linewidth=1.5, linecolor='black',
            ticks='outside', tickcolor='black', showticklabels=True,
            tickformat=".0f"
        )
    )
    
    # -------------------------------------------------------------------------
    # 📥 NO-LIBRARY HIGH-RESOLUTION DOWNLOAD LOGIC
    # -------------------------------------------------------------------------
    plotly_config = {
        'displayModeBar': True,
        'displaylogo': False,
        'toImageButtonOptions': {
            'format': 'png', 
            'filename': 'Exploration_Map_HighRes',
            'height': 650,
            'width': 1200,
            'scale': 3 
        }
    }
    
    st.plotly_chart(fig, use_container_width=True, config=plotly_config, key=f"plotly_map_{key_suffix}")




# -----------------------------------------------------------------------------
# 4. VIEW RENDERING (TABS)
# -----------------------------------------------------------------------------
def render_dashboard():
    # Strict CSS injection to force metric center alignment, AND handle A4 Printing
    st.markdown("""
<style>
/* Center metrics */
[data-testid="stMetric"] {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
}
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
    width: 100%;
    display: flex;
    justify-content: center;
}

/* =========================================================
   🖨️ PRINT STYLES FOR A4 DASHBOARD EXPORT
   ========================================================= */
@media print {
    @page { size: A4 portrait; margin: 10mm; }
    header, .stSidebar, .stApp > header, .stToolbar, iframe { display: none !important; }
    .print-hide { display: none !important; }
    .stPlotlyChart, .stDataFrame, table, div[data-testid="stVerticalBlock"] > div {
        page-break-inside: avoid !important; break-inside: avoid !important;
    }
    h1, h2, h3, h4, h5, h6 { page-break-after: avoid !important; break-after: avoid !important; }
    * { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
}
</style>
""", unsafe_allow_html=True)

    # --- ⚠️ NEW: Global Alert Input in Sidebar ---
    st.sidebar.markdown("### ⚠️ Daily Operational Alert")
    global_alert = st.sidebar.text_area(
        "Block-wide issues (e.g., Heavy Rain):", 
        help="This text will appear in the News Banner and WhatsApp Report."
    ).strip()
    
    col_title, col_print = st.columns([3, 1])
    with col_title:
        st.markdown(f"### 📊 Exploration Overview")
        
    with col_print:
        import streamlit.components.v1 as components
        components.html(
            """
            <div style="text-align:right;">
                <button onclick="window.parent.print()" style="background-color:#2962ff; color:white; border:none; padding:8px 16px; border-radius:4px; cursor:pointer; font-weight:bold; margin-top:2px; box-shadow: 0 2px 4px rgba(0,0,0,0.2); font-family: sans-serif;">
                    🖨️ Print Dashboard
                </button>
            </div>
            """,
            height=45
        )
        
    collar, litho = st.session_state['df_collar'], st.session_state.get('df_litho', pd.DataFrame())
    
    # --- NEW LOGIC: Filter only CMPDI boreholes for all dashboard calculations ---
    collar = collar[collar['TYPE'] == 'CMPDI'].copy()
    
    total_bh = len(collar)
    closed = len(collar[collar['STATUS'] == 'Closed'])
    running = len(collar[collar['STATUS'] == 'Running'])
    shifting = len(collar[collar['STATUS'] == 'Under Shifting'])
    breakdown = len(collar[collar['STATUS'] == 'Breakdown'])
    pending = len(collar[collar['STATUS'] == 'Pending'])
    
    # Safe check for GPL Status
    gpl = 0
    if 'GPL STATUS' in collar.columns:
        gpl = len(collar[collar['GPL STATUS'].astype(str).str.strip().str.title() == 'Done'])
    
    pending_total = pending + shifting + running + breakdown

    # Calculate Core Meterages early so they can be used in reports and banners
    total_proposed = collar['ED'].sum()
    total_drilled = collar[collar['STATUS'].isin(['Closed', 'Running', 'Under Shifting', 'Breakdown'])]['DEPTH'].sum()
    balance = max(0, total_proposed - total_drilled)

    active_metrics = [
        {"label": "Total Proposed BHs", "value": total_bh},
        {"label": "Closed Boreholes", "value": closed},
        {"label": "Running", "value": running},
        {"label": "GPL Completed", "value": gpl}
    ]
    
    if shifting > 0: active_metrics.append({"label": "Under Shifting", "value": shifting})
    if breakdown > 0: active_metrics.append({"label": "Breakdown", "value": breakdown})
    active_metrics.append({"label": "Pending", "value": pending_total})

    cols = st.columns(len(active_metrics))
    for col, metric in zip(cols, active_metrics):
        col.metric(metric["label"], metric["value"])
    
    # =====================================================================
    # 📢 NEWS UPDATE BANNER LOGIC
    # =====================================================================
    today = pd.to_datetime('today').normalize()
    news_items = []
    
    # --- 🚨 INJECT GLOBAL ALERT INTO BANNER ---
    if global_alert:
        news_items.append(f"<b style='color: #d32f2f;'>{global_alert}</b>")
    
    news_items.append(f"<b>Total Meterage:</b> {total_drilled:,.2f} m")
    news_items.append(f"<b>Closed Bhs:</b> {closed}/{total_bh}")
    
    breakdown_bhs = collar[collar['STATUS'] == 'Breakdown']
    if not breakdown_bhs.empty:
        bd_rigs = []
        for _, r in breakdown_bhs.iterrows():
            rig = str(r.get('RIG NO', 'Unknown'))
            try: rig = str(int(float(rig)))
            except: pass
            bd_rigs.append(f"Rig {rig}")
        news_items.append(f"<b>Rig Status:</b> 🔴 Breakdown ({', '.join(bd_rigs)})")
    else:
        news_items.append("<b>Rig Status:</b> 🟢 All Operational")

    closed_bhs_news = collar[collar['STATUS'] == 'Closed'].copy()
    if not closed_bhs_news.empty and 'CLOSING DATE' in closed_bhs_news.columns:
        closed_bhs_news['CLOSING DATE_DT'] = pd.to_datetime(closed_bhs_news['CLOSING DATE'], errors='coerce')
        closed_bhs_news_sorted = closed_bhs_news.dropna(subset=['CLOSING DATE_DT']).sort_values(by='CLOSING DATE_DT', ascending=False)
        if not closed_bhs_news_sorted.empty:
            last_closed = closed_bhs_news_sorted.iloc[0]
            days_since_close = (today - last_closed['CLOSING DATE_DT'].normalize()).days
            if 0 <= days_since_close <= 4:
                lc_bhid = last_closed.get('DISPLAY_ID', 'N/A')
                lc_depth = last_closed.get('DEPTH', 0)
                lc_date_str = last_closed['CLOSING DATE_DT'].strftime('%d-%b-%Y')
                news_items.append(f"<b>Last Closed:</b> {lc_bhid} at {lc_depth:.2f}m ({lc_date_str})")

    # --- 🔬 SAMPLE DISPATCH BANNER EXTRACTION (LAST 4 DAYS) ---
    recent_dispatches = pd.DataFrame()
    if not closed_bhs_news.empty and 'DESPATCH_DATE' in closed_bhs_news.columns:
        dispatch_df = closed_bhs_news.copy()
        dispatch_df['DESPATCH_DATE_DT'] = pd.to_datetime(dispatch_df['DESPATCH_DATE'], errors='coerce')
        dispatch_df = dispatch_df.dropna(subset=['DESPATCH_DATE_DT'])
        
        # Filter for samples sent in the last 4 days
        recent_dispatches = dispatch_df[(today - dispatch_df['DESPATCH_DATE_DT']).dt.days.between(0, 4)].copy()
        
        if not recent_dispatches.empty:
            for (date, lab), group in recent_dispatches.groupby(['DESPATCH_DATE_DT', 'LAB']):
                num_bhs = len(group)
                lab_name = str(lab).strip().upper() if pd.notna(lab) else "Lab"
                date_str = date.strftime('%d-%b-%Y')
                news_items.append(f"<b>Samples Dispatched ({date_str}):</b> No. of boreholes - {num_bhs}  Lab - {lab_name}")

    news_html = " &nbsp; <span style='color:#ccc;'>|</span> &nbsp; ".join(news_items)

    st.markdown(f"""
<div style="padding: 12px 15px; border-radius: 8px; background-color: #f8faff; color: #1e1e1e; border: 1px solid #d6e4ff; border-left: 6px solid #2962ff; font-size: 15px; margin-bottom: 10px; display: flex; align-items: center; box-shadow: 0 2px 4px rgba(0,0,0,0.04);">
    <span style="font-size: 20px; margin-right: 12px; animation: blink 2s infinite;">📢</span>
    <div style="line-height: 1.5;">
        <strong style="color: #2962ff; letter-spacing: 0.5px;">UPDATE:</strong> &nbsp; {news_html}
    </div>
</div>
<style>
    @keyframes blink {{ 0% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} 100% {{ opacity: 1; }} }}
</style>
""", unsafe_allow_html=True)
    
    # =====================================================================
    # 📋 DAILY DRILLING PROGRESS REPORT GENERATOR (WhatsApp Auto-Format)
    # =====================================================================
    from datetime import timedelta
    yesterday_dt = today - timedelta(days=1)
    
    def safe_format_rig_no(val):
        if pd.isna(val) or str(val).strip().lower() in ['none', 'nan', '']: return ""
        try: return f"{int(float(val))}"
        except: return str(val)

    # Helper function to find the previous point drilled by a specific rig
    def get_last_point_for_rig(rig_no, current_bhid):
        if not rig_no: return "previous point"
        rig_bhs = collar[(collar['RIG NO'].astype(str) == str(rig_no)) & (collar['BHID'] != current_bhid)].copy()
        if rig_bhs.empty: return "previous point"
        # Try to sort by DOC to get the most recent one before this
        rig_bhs['temp_date'] = pd.to_datetime(rig_bhs['DOC'], errors='coerce')
        rig_bhs = rig_bhs.sort_values('temp_date', ascending=False)
        last_pt = rig_bhs.iloc[0].get('POINT', 'previous point')
        return str(last_pt) if pd.notna(last_pt) and str(last_pt).strip() != '' else "previous point"

    # Calculate Total Sent Boreholes globally for the report
    total_sent_bhs = 0
    if 'SAMPLING_STATUS' in collar.columns:
        total_sent_bhs = len(collar[collar['SAMPLING_STATUS'].astype(str).str.strip().str.lower() == 'sent'])

    report_lines = []
    report_lines.append("*DRILLING PROGRESS REPORT*")
    report_lines.append(f"Date: {yesterday_dt.strftime('%d-%m-%Y')}")
    
    # --- 🚨 INJECT GLOBAL ALERT INTO WHATSAPP REPORT ---
    if global_alert:
        report_lines.append(f"🚨 *UPDATE:* _{global_alert}_")
        
    report_lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    active_bhs = collar[collar['STATUS'].isin(['Running', 'Breakdown', 'Under Shifting'])].copy()
    recent_closed_df = pd.DataFrame()
    if not closed_bhs_news.empty and 'CLOSING DATE_DT' in closed_bhs_news.columns:
        recent_closed_df = closed_bhs_news[closed_bhs_news['CLOSING DATE_DT'] >= yesterday_dt].copy()
        
    report_bhs = pd.concat([active_bhs, recent_closed_df]).drop_duplicates(subset=['BHID'])
    report_bhs['RIG_SORT'] = pd.to_numeric(report_bhs['RIG NO'], errors='coerce').fillna(999)
    report_bhs = report_bhs.sort_values('RIG_SORT')
    
    for _, row in report_bhs.iterrows():
        raw_rig_no = row.get('RIG NO', '')
        rig_no = safe_format_rig_no(raw_rig_no)
        rig_model = str(row.get('RIG MODEL', '')).strip()
        
        rig_display = f"*Rig {rig_no}*" if rig_no else "*Unknown Rig*"
        if rig_model and rig_model.lower() not in ['nan', 'none', '']:
            rig_display += f" ({rig_model})"
            
        bhid = row.get('BHID', 'N/A')
        if pd.isna(bhid) or str(bhid).lower() == 'nan': bhid = 'nan'
        point = row.get('POINT', 'N/A')
        bhid_str = f"BHID: {bhid} | {point}" if point != 'N/A' and str(point).lower() != 'nan' else f"BHID: {bhid}"
        
        # Combined single-line Rig + BHID output
        report_lines.append(f"▪️ {rig_display}, {bhid_str}")
        
        status = row['STATUS']
        depth = row.get('DEPTH', 0)
        
        if status == 'Closed':
            report_lines.append(f"  * Borehole Closed at {depth:.2f} m")
            gpl_stat = str(row.get('GPL STATUS', '')).strip().title()
            if gpl_stat == 'Done':
                report_lines.append(f"  * GPL completed up to {depth:.2f} m")
        elif status == 'Running':
            report_lines.append(f"  * Depth: {depth:.2f} m")
            doc = pd.to_datetime(row.get('DOC'), errors='coerce')
            if pd.notna(doc):
                days_running = (yesterday_dt - doc.normalize()).days
                if -1 <= days_running <= 2: 
                    last_point = get_last_point_for_rig(raw_rig_no, bhid)
                    report_lines.append(f"  _Remark: Rig No. {rig_no}, Shifted from {last_point} to {point}, and Drilling operation started at {point}._")
            
            custom_remark = str(row.get('REMARKS', '')).strip()
            if custom_remark and custom_remark.lower() not in ['nan', 'none', '']:
                report_lines.append(f"  _Remark: {custom_remark}_")
                
        elif status == 'Breakdown':
            report_lines.append(f"  * Depth: {depth:.2f} m")
            custom_remark = str(row.get('REMARKS', '')).strip()
            if custom_remark and custom_remark.lower() not in ['nan', 'none', '']:
                report_lines.append(f"  _Remark: {custom_remark}_")
            else:
                report_lines.append(f"  _Remark: Rig is under breakdown condition._")
                
        elif status == 'Under Shifting':
            last_point = get_last_point_for_rig(raw_rig_no, bhid)
            report_lines.append(f"  _Remark: Rig No. {rig_no}, Under shifting from {last_point} to {point}._")

        report_lines.append("")

    # --- 📦 INJECT SAMPLE DISPATCH INTO WHATSAPP REPORT ---
    if not recent_dispatches.empty:
        report_lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━")
        report_lines.append("📦 *Recent Sample Dispatches*")
        for (date, lab), group in recent_dispatches.groupby(['DESPATCH_DATE_DT', 'LAB']):
            num_bhs = len(group)
            lab_name = str(lab).strip().upper() if pd.notna(lab) else "Lab"
            date_str = date.strftime('%d-%m-%Y')
            report_lines.append(f"▪️ Samples Dispatched ({date_str}): No. of boreholes - {num_bhs}, Lab - {lab_name}")
        report_lines.append(f"Total - {total_sent_bhs} BHs")
        # report_lines.append("")

    report_lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━")
    report_lines.append("📌 *Project Summary*")
    report_lines.append(f"✅ Closed Boreholes: {closed} / {total_bh}")
    report_lines.append(f"📈 Cumulative Meterage: {total_drilled:,.2f} m")
    report_lines.append(f"🎯 GPL Completed: {gpl} BHs")
    report_lines.append(f"🚧 Balance Meterage: {balance:,.2f} m")
    report_lines.append(f"Total BHs Sent for analysis - {total_sent_bhs} BHs")
    
    report_text = "\n".join(report_lines)
    
    with st.expander("📋 Generate Daily Drilling Progress Report"):
        st.code(report_text, language="markdown")

   
    # =====================================================================

    c_left, c_right = st.columns([1, 1.5])
    
    with c_left:
        st.markdown("### 🚧 Meterage Completion")
        fig_comp = px.pie(
            names=['Completed (m)', 'Balance (m)'], 
            values=[total_drilled, balance],
            hole=0.5,
            color_discrete_sequence=['#00CC96', '#FFA15A']
        )
        fig_comp.update_traces(textinfo='percent', hoverinfo='label+percent+value')
        fig_comp.update_layout(margin=dict(t=10, b=0, l=0, r=0), height=180, showlegend=True)
        st.plotly_chart(fig_comp, use_container_width=True)
        
        st.markdown(f"""
<div style='text-align: left; font-size: 16px; margin-top: 5px;'>
    <b style='color: #AB63FA;'>Total Meterage Drilled:</b> {total_drilled:,.2f} m <br>
    <b>Total Proposed Meterage:</b> {total_proposed:,.2f} m <br>
    <b>Balance Meterage:</b> {balance:,.2f} m
</div>
""", unsafe_allow_html=True)

    with c_right:
        st.markdown("### 🚧 Running Boreholes Progress")
        running_bhs = collar[collar['STATUS'].isin(['Running', 'Breakdown'])].copy()
        
        if not running_bhs.empty:
            marker_colors = running_bhs['STATUS'].map(lambda s: STATUS_COLORS.get(s, '#FFA15A')).tolist()
            
            fig_prog = go.Figure()
            fig_prog.add_trace(go.Bar(
                x=running_bhs['ED'], y=running_bhs['DISPLAY_ID'],
                orientation='h', name='Estimated Depth',
                marker=dict(color='#f0f2f6', line=dict(color='#d9d9d9', width=1)),
                hoverinfo='text',
                text=running_bhs['ED'].apply(lambda x: f"{x:.2f} m"), textposition='inside', 
                hovertext=running_bhs['ED'].apply(lambda x: f"Target: {x:.2f} m")
            ))
            fig_prog.add_trace(go.Bar(
                x=running_bhs['DEPTH'], y=running_bhs['DISPLAY_ID'],
                orientation='h', name='Current Depth',
                marker=dict(color=marker_colors),
                text=running_bhs['DEPTH'].apply(lambda x: f"{x:.2f} m"), textposition='inside',
                hoverinfo='text', hovertext=running_bhs['DEPTH'].apply(lambda x: f"Current: {x:.2f} m")
            ))
            fig_prog.update_layout(
                barmode='overlay', 
                height=max(180, len(running_bhs) * 45 + 50), 
                margin=dict(l=0, r=20, t=10, b=0),
                xaxis=dict(title="Depth (m)", showgrid=True, zeroline=True),
                yaxis=dict(title="", autorange="reversed"), 
                showlegend=True,
                plot_bgcolor='white'
            )
            st.plotly_chart(fig_prog, use_container_width=True)
        else:
            st.info("No boreholes are currently running.")
    st.markdown("---")
    # =====================================================================
    # Block 1: Rig Deployment in 4 Columns
    st.markdown("### 🏁 Rig Deployment Status")
    active_rigs = collar[collar['STATUS'].isin(['Running', 'Under Shifting', 'Breakdown'])].copy()
    if not active_rigs.empty:
        rig_cols = st.columns(4)
        for i, (_, row) in enumerate(active_rigs.iterrows()):
            rig_name = str(row.get('RIG NO', 'Unknown Rig')).strip()
            if not rig_name or rig_name.lower() == 'nan': rig_name = "Unnamed Rig"
            
            model = str(row.get('RIG MODEL', '')).strip()
            if model and model.lower() != 'nan': 
                rig_name += f" ({model})"
            
            status = row['STATUS']
            loc = row['DISPLAY_ID']
            
            with rig_cols[i % 4]:
                if status == 'Running':
                    st.success(f"🟢 Rig - **{rig_name}**  \nLocation: **{loc}**  \n*(RUNNING)*")
                elif status == 'Under Shifting':
                    st.info(f"🔵 Rig - **{rig_name}**  \nTarget Location: **{loc}**  \n*(UNDER SHIFTING)*")
                elif status == 'Breakdown':
                    st.error(f"🔴 Rig - **{rig_name}**  \nLocation: **{loc}**  \n*(BREAKDOWN)*")
    else:
        st.info("No active, shifting, or breakdown rigs currently recorded in the database.")   
    def format_rig_no(val):
        if pd.isna(val) or str(val).strip().lower() in ['none', 'nan', '']:
            return ""
        try:
            return f"{int(float(val))}"
        except (ValueError, TypeError):
            return str(val)
    def clean_none_values(df):
        df_clean = df.copy()
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].apply(
                    lambda x: '' if pd.isna(x) or str(x).strip().lower() in ['none', 'nan', 'na', '<na>'] else x
                )
        return df_clean

    center_styles = [
        {'selector': 'th', 'props': [('text-align', 'center !important'), ('font-weight', 'bold !important')]},
        {'selector': 'td', 'props': [('text-align', 'center !important')]}
    ]
    
    # Block 2: Active / Ongoing Boreholes Table
    st.markdown("### 🏁 Active Boreholes")
    if not active_rigs.empty:
        if not active_rigs['DOC'].isna().all():
            active_rigs = active_rigs.sort_values(by='DOC', ascending=False)
            
        recent_active = active_rigs.head(4).sort_values(by='DISPLAY_ID', ascending=True)
        
        disp_cols = ['DISPLAY_ID','RL','ED', 'DOC', 'DEPTH', 'HQ' , 'RIG NO', 'RIG MODEL','STATUS', 'REMARKS']
        actual_cols = [c for c in disp_cols if c in recent_active.columns]
        disp_recent_act = recent_active[actual_cols].copy()
        disp_recent_act = disp_recent_act.rename(columns={'DISPLAY_ID': 'BHID/Point'})
        
        for date_col in ['CLOSING DATE', 'DOC']:
            if date_col in disp_recent_act.columns:
                if pd.api.types.is_datetime64_any_dtype(disp_recent_act[date_col]):
                    disp_recent_act[date_col] = disp_recent_act[date_col].dt.strftime('%d-%b-%Y').fillna('')
                else:
                    disp_recent_act[date_col] = disp_recent_act[date_col].astype(str).replace(['nan', 'NaT', 'None'], '')
        
        disp_recent_act = clean_none_values(disp_recent_act)
        
        def highlight_status(row):
            status = str(row.get('STATUS', ''))
            remarks = str(row.get('REMARKS', '')).lower() 
            if status in ['Running', 'Breakdown'] and ("achieved closure depth" in remarks or "near closure depth" in remarks):
                return ['background-color: rgba(255, 215, 0, 0.3)'] * len(row) 
            if status == 'Running':
                return ['background-color: rgba(144, 238, 144, 0.4)'] * len(row)
            elif status == 'Breakdown':
                return ['background-color: rgba(231, 76, 60, 0.3)'] * len(row)
            elif status == 'Under Shifting':
                return ['background-color: rgba(52, 152, 219, 0.3)'] * len(row)
            return [''] * len(row)

        num_cols_act = disp_recent_act.select_dtypes(include=['number']).columns
        format_dict_act = {c: "{:.2f}" for c in num_cols_act if c != 'RIG NO'}
        if 'RIG NO' in disp_recent_act.columns:
            format_dict_act['RIG NO'] = format_rig_no

        styled_act = (
            disp_recent_act.style
            .apply(highlight_status, axis=1)
            .format(format_dict_act, na_rep="")
            .set_properties(**{'text-align': 'center !important'})
            .set_table_styles(center_styles)
        )

        st.dataframe(styled_act, use_container_width=True, hide_index=True)
    else:
        st.info("No running, shifting, or breakdown boreholes found.")
        
    st.write("")
    
    # Block 3: Recently Closed Boreholes Table (Top 5)
    st.markdown("### 🏁 Recently Closed Boreholes")
    closed_bhs = collar[collar['STATUS'] == 'Closed'].copy()
    
    if not closed_bhs.empty:
        if not closed_bhs['CLOSING DATE'].isna().all():
            closed_bhs = closed_bhs.sort_values(by='CLOSING DATE', ascending=False)
            
        most_recent_bhid = closed_bhs.iloc[0]['DISPLAY_ID']
        recent_closed = closed_bhs.head(5).sort_values(by='DISPLAY_ID', ascending=True)
        
        disp_cols = ['DISPLAY_ID','RL','ED', 'DOC', 'DEPTH', 'HQ', 'CLOSING DATE', 'RIG NO', 'RIG MODEL', 'GPL STATUS','STATUS', 'SID', 'REMARKS']
        actual_cols = [c for c in disp_cols if c in recent_closed.columns]
        disp_recent = recent_closed[actual_cols].copy()
        disp_recent = disp_recent.rename(columns={'DISPLAY_ID': 'BHID/Point'})
        
        for date_col in ['CLOSING DATE', 'DOC']:
            if date_col in disp_recent.columns:
                if pd.api.types.is_datetime64_any_dtype(disp_recent[date_col]):
                    disp_recent[date_col] = disp_recent[date_col].dt.strftime('%d-%b-%Y').fillna('')
                else:
                    disp_recent[date_col] = disp_recent[date_col].astype(str).replace(['nan', 'NaT', 'None'], '')

        disp_recent = clean_none_values(disp_recent)

        num_cols_closed = disp_recent.select_dtypes(include=['number']).columns
        format_dict_closed = {c: "{:.2f}" for c in num_cols_closed if c != 'RIG NO'}
        if 'RIG NO' in disp_recent.columns:
            format_dict_closed['RIG NO'] = format_rig_no

        def highlight_newest_closed(row):
            if row['BHID/Point'] == most_recent_bhid:
                return ['background-color: rgba(144, 238, 144, 0.4)'] * len(row)
            return [''] * len(row)

        styled_closed = (
            disp_recent.style
            .apply(highlight_newest_closed, axis=1)
            .format(format_dict_closed, na_rep="")
            .set_properties(**{'text-align': 'center !important'})
            .set_table_styles(center_styles)
        )

        st.dataframe(styled_closed, use_container_width=True, hide_index=True)
    else:
        st.info("No closed boreholes found.")
        
    st.markdown("---")
    render_map() # Assuming render_map() is globally accessible

    # =====================================================================
    # 🧪 Sampling Status & Dispatch Statistics
    # =====================================================================
    st.markdown("---")
    st.markdown("### 🧪 Sampling & Dispatch Status")
    
    if not closed_bhs.empty:
        samp_df = closed_bhs.copy()
        
        if 'SAMPLING_STATUS' in samp_df.columns and 'LAB' in samp_df.columns:
            samp_df['STATUS_CLEAN'] = samp_df['SAMPLING_STATUS'].fillna('Pending').astype(str).str.strip().str.title()
            null_strings = ['', 'Nan', 'None', 'Na', '<Na>']
            samp_df.loc[samp_df['STATUS_CLEAN'].isin(null_strings), 'STATUS_CLEAN'] = 'Pending'
            
            samp_df['LAB_CLEAN'] = samp_df['LAB'].fillna('').astype(str).str.strip().str.upper()
            samp_df.loc[samp_df['LAB_CLEAN'].isin(['NAN', 'NONE', 'NA', '<NA>']), 'LAB_CLEAN'] = ''
            
            def make_metric_label(row):
                status = row['STATUS_CLEAN']
                lab = row['LAB_CLEAN']
                if status.lower() == 'sent' and lab != '':
                    return f"{lab}/Sent"
                return status
                
            samp_df['METRIC_LABEL'] = samp_df.apply(make_metric_label, axis=1)
            sampling_counts = samp_df['METRIC_LABEL'].value_counts().reset_index()
            sampling_counts.columns = ['Status Category', 'Borehole Count']
            
            sent_mask = sampling_counts['Status Category'].str.lower().str.contains('sent')
            total_sent = sampling_counts.loc[sent_mask, 'Borehole Count'].sum()
            
            samp_metrics = []
            
            pending_row = sampling_counts[sampling_counts['Status Category'] == 'Pending']
            if not pending_row.empty:
                samp_metrics.append({"label": "Pending", "value": int(pending_row['Borehole Count'].iloc[0])})
            else:
                samp_metrics.append({"label": "Pending", "value": 0})
                
            packed_row = sampling_counts[sampling_counts['Status Category'] == 'Packed']
            if not packed_row.empty:
                samp_metrics.append({"label": "Packed", "value": int(packed_row['Borehole Count'].iloc[0])})
                
            for _, row in sampling_counts.iterrows():
                status = row['Status Category']
                if status not in ['Pending', 'Packed']:
                    samp_metrics.append({"label": str(status), "value": int(row['Borehole Count'])})
                    
            samp_metrics.append({"label": "Total Sent", "value": int(total_sent)})
            
            samp_cols = st.columns(len(samp_metrics))
            for col, metric in zip(samp_cols, samp_metrics):
                col.metric(metric["label"], metric["value"])
                
        else:
            st.info("Columns 'SAMPLING_STATUS' and/or 'LAB' not found in database.")
            
        st.write("") 
            
        if 'DESPATCH_DATE' in samp_df.columns:
            batch_df = samp_df.dropna(subset=['DESPATCH_DATE']).copy()
            
            if 'LAB_CLEAN' not in batch_df.columns:
                if 'LAB' in batch_df.columns:
                    batch_df['LAB_CLEAN'] = batch_df['LAB'].fillna('').astype(str).str.strip().str.upper()
                    batch_df.loc[batch_df['LAB_CLEAN'].isin(['NAN', 'NONE', 'NA', '<NA>']), 'LAB_CLEAN'] = 'Unknown'
                else:
                    batch_df['LAB_CLEAN'] = 'Unknown'
            
            if pd.api.types.is_datetime64_any_dtype(batch_df['DESPATCH_DATE']):
                batch_df['DATE_STR'] = batch_df['DESPATCH_DATE'].dt.strftime('%d-%b-%Y')
            else:
                batch_df['DATE_STR'] = batch_df['DESPATCH_DATE'].astype(str).str.strip()
                batch_df = batch_df[~batch_df['DATE_STR'].str.lower().isin(['nan', 'none', '', 'nat'])]
                
            if not batch_df.empty:
                batch_stats = batch_df.groupby('DATE_STR').agg(
                    Borehole_Count=('DISPLAY_ID', 'count'),
                    Lab_Destination=('LAB_CLEAN', lambda x: ', '.join(filter(lambda item: item != '', x.astype(str).unique()))),
                    Boreholes=('DISPLAY_ID', lambda x: ', '.join(x.astype(str)))
                ).reset_index()
                
                # --- APPLY SORTING: LATEST DATE ON TOP ---
                try:
                    # Explicitly convert to datetime for robust sorting
                    batch_stats['SortDate'] = pd.to_datetime(batch_stats['DATE_STR'], format='%d-%b-%Y', errors='coerce')
                    batch_stats = batch_stats.sort_values(by='SortDate', ascending=False).drop(columns=['SortDate'])
                except Exception:
                    pass
                
                batch_stats.columns = ['Dispatch Date', 'No. of Boreholes', 'Lab', 'Borehole IDs']
                
                # --- APPLY TEXT WRAPPING TO BOREHOLES COLUMN ---
                import textwrap
                # Dynamically slice the string using textwrap to force linebreaks 
                # (so it won't trail off outside the dataframe view limits)
                batch_stats['Borehole IDs'] = batch_stats['Borehole IDs'].apply(
                    lambda x: "\n".join(textwrap.wrap(x, width=45))
                )
            else:
                batch_stats = pd.DataFrame(columns=['Dispatch Date', 'No. of Boreholes', 'Lab', 'Borehole IDs'])
        else:
            batch_stats = pd.DataFrame(columns=['Dispatch Date', 'No. of Boreholes', 'Lab', 'Borehole IDs'])

        st.markdown("###### Sample Dispatch Batches")
        if not batch_stats.empty:
            styled_batch = (
                batch_stats.style
                .set_properties(**{'text-align': 'center !important', 'white-space': 'pre-wrap'})
                .set_table_styles(center_styles)
            )
            # Use column config to give the Wrapped Borehole column adequate space
            st.dataframe(
                styled_batch, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Borehole IDs": st.column_config.TextColumn(
                        "Borehole IDs",
                        width="large",
                    )
                }
            )
        else:
            st.info("No dispatch dates found in 'DESPATCH_DATE' column yet.")
    else:
        st.info("No closed boreholes available to analyze sampling data.")

    # =====================================================================
    # 📚 Complete Closed Boreholes Archive
    # =====================================================================
    st.markdown("---")
    st.markdown("### 📚 Complete Closed Boreholes Archive")
    
    if not closed_bhs.empty:
        if not closed_bhs['CLOSING DATE'].isna().all():
            full_closed = closed_bhs.sort_values(by='CLOSING DATE', ascending=False)
        else:
            full_closed = closed_bhs.sort_values(by='DISPLAY_ID', ascending=True)

        disp_cols_full = ['DISPLAY_ID','RL','ED', 'DEPTH', 'HQ', 'DOC', 'CLOSING DATE', 'RIG NO', 'RIG MODEL', 'GPL STATUS', 'STATUS', 'SID']
        actual_cols_full = [c for c in disp_cols_full if c in full_closed.columns]
        disp_full = full_closed[actual_cols_full].copy()
        disp_full = disp_full.rename(columns={'DISPLAY_ID': 'BHID/Point'})
        
        for date_col in ['CLOSING DATE', 'DOC', 'SAMPLE_SENT_DATE']:
            if date_col in disp_full.columns:
                if pd.api.types.is_datetime64_any_dtype(disp_full[date_col]):
                    disp_full[date_col] = disp_full[date_col].dt.strftime('%d-%b-%Y').fillna('')
                else:
                    disp_full[date_col] = disp_full[date_col].astype(str).replace(['nan', 'NaT', 'None'], '')

        disp_full = clean_none_values(disp_full)

        num_cols_full = disp_full.select_dtypes(include=['number']).columns
        format_dict_full = {c: "{:.2f}" for c in num_cols_full if c != 'RIG NO'}
        if 'RIG NO' in disp_full.columns:
            format_dict_full['RIG NO'] = format_rig_no

        styled_full = (
            disp_full.style
            .format(format_dict_full, na_rep="")
            .set_properties(**{'text-align': 'center !important'})
            .set_table_styles(center_styles)
        )

        st.dataframe(styled_full, use_container_width=True, hide_index=True)
    else:
        st.info("No closed boreholes found in the database.")



def plot_plotly_graphic_log(df_litho, df_collar, selected_bhids):
    df_plot = df_litho[df_litho['BHID'].isin(selected_bhids)].copy()
    if not df_plot.empty:
        df_plot = pd.merge(df_plot, df_collar[['BHID', 'RL', 'DEPTH']], on='BHID', how='left')
        df_plot['RL'] = df_plot['RL'].fillna(0)
        df_plot['FROM_RL'] = df_plot['RL'] - df_plot['FROM']
        df_plot['TO_RL'] = df_plot['RL'] - df_plot['TO']
        
        seams = df_plot[df_plot['IS_COAL']]['SEAM'].unique()
        seam_colors = generate_seam_colors(seams)
        df_plot['COLOR'] = df_plot.apply(lambda r: seam_colors.get(r['SEAM'], COAL_COLOR) if r['IS_COAL'] else LITHO_COLORS.get(r['LITHO'], '#D3D3D3'), axis=1)
        df_plot['LABEL'] = df_plot.apply(lambda r: r['SEAM'] if r['IS_COAL'] else r['LITHO'], axis=1)

    x_map = {bhid: float(i) for i, bhid in enumerate(selected_bhids)}
    fig = go.Figure()
    BAR_WIDTH = 0.15 
    
    max_depth_overall = df_collar[df_collar['BHID'].isin(selected_bhids)]['DEPTH'].max()
    if pd.isna(max_depth_overall) or max_depth_overall == 0: max_depth_overall = 100

    for bhid in selected_bhids:
        x_idx = x_map[bhid]
        collar_row = df_collar[df_collar['BHID'] == bhid]
        if collar_row.empty: continue
            
        rl = collar_row.iloc[0]['RL']
        td = collar_row.iloc[0]['DEPTH']
        if pd.isna(td): td = 0

        fig.add_trace(go.Bar(
            x=[x_idx], y=[td], base=[rl - td],
            marker=dict(color='rgba(0,0,0,0)', line=dict(color='black', width=1)),
            width=BAR_WIDTH, hoverinfo='skip', showlegend=False
        ))
        
        display_id = collar_row.iloc[0]['DISPLAY_ID'] if 'DISPLAY_ID' in collar_row.columns else bhid

        fig.add_annotation(
            x=x_idx, y=rl + (td * 0.05) if td > 0 else rl + 10, 
            text=f"<b>{display_id}</b><br>RL: {rl:.1f}<br>TD: {td:.1f}m", 
            showarrow=False, yanchor='bottom', font=dict(size=10)
        )

        if not df_plot.empty:
            subset = df_plot[df_plot['BHID'] == bhid]
            if not subset.empty:
                hover_text = (
                    "<b>BHID:</b> " + subset['BHID'] + "<br>" +
                    "<b>" + subset['LABEL'] + "</b><br>" +
                    "Depth: " + subset['FROM'].map("{:.2f}".format) + " to " + subset['TO'].map("{:.2f}".format) + " m<br>" +
                    "Thickness: " + subset['WIDTH'].map("{:.2f}".format) + " m"
                )

                fig.add_trace(go.Bar(
                    x=[x_idx] * len(subset), y=subset['WIDTH'], base=subset['TO_RL'],
                    marker=dict(color=subset['COLOR'], line=dict(color='black', width=1)),
                    text=subset['LABEL'], textposition='inside', textfont=dict(color='black', size=9),
                    width=BAR_WIDTH, hovertext=hover_text, hoverinfo='text', showlegend=False
                ))

    if not df_plot.empty:
        coal_subset = df_plot[df_plot['IS_COAL']].copy()
        if not coal_subset.empty:
            coal_subset['X_POS'] = coal_subset['BHID'].map(x_map)
            coal_subset = coal_subset.sort_values(['BHID', 'FROM'])
            
            coal_subset['PREV_TO'] = coal_subset.groupby(['BHID', 'SEAM'])['TO'].shift(1)
            coal_subset['NEW_INSTANCE'] = ((coal_subset['FROM'] - coal_subset['PREV_TO']) >= 0.50).astype(int).fillna(0)
            coal_subset['INSTANCE'] = coal_subset.groupby(['BHID', 'SEAM'])['NEW_INSTANCE'].cumsum()

            seam_envelopes = coal_subset.groupby(['BHID', 'SEAM', 'INSTANCE', 'X_POS']).agg(
                FROM_RL=('FROM_RL', 'max'), TO_RL=('TO_RL', 'min'),
                FROM_DEPTH=('FROM', 'min'), TO_DEPTH=('TO', 'max')
            ).reset_index()

            seam_envelopes['ENVELOPE_WIDTH'] = seam_envelopes['TO_DEPTH'] - seam_envelopes['FROM_DEPTH']
            seam_envelopes['MID_RL'] = (seam_envelopes['FROM_RL'] + seam_envelopes['TO_RL']) / 2
            
            seam_envelopes['FMT_WIDTH'] = seam_envelopes['ENVELOPE_WIDTH'].apply(lambda x: f"{x:.2f}")
            seam_envelopes['FMT_FROM'] = seam_envelopes['FROM_DEPTH'].apply(lambda x: f"{x:.2f}")
            seam_envelopes['FMT_TO'] = seam_envelopes['TO_DEPTH'].apply(lambda x: f"{x:.2f}")
            
            LABEL_OFFSET = (BAR_WIDTH / 2) + 0.12 
            BRACE_WIDTH = 0.05
            
            brace_x, brace_y = [], []
            for _, row in seam_envelopes.iterrows():
                bx = row['X_POS'] + (BAR_WIDTH/2) + 0.05
                brace_x.extend([bx, bx + BRACE_WIDTH, bx + BRACE_WIDTH, bx, None])
                brace_y.extend([row['FROM_RL'], row['FROM_RL'], row['TO_RL'], row['TO_RL'], None])
                
            fig.add_trace(go.Scatter(
                x=brace_x, y=brace_y, mode='lines', 
                line=dict(color='black', width=1.5), 
                hoverinfo='skip', showlegend=False
            ))

            label_text = "<b>" + seam_envelopes['SEAM'] + "</b> - " + seam_envelopes['FMT_WIDTH'] + "m"
            fig.add_trace(go.Scatter(
                x=seam_envelopes['X_POS'] + (BAR_WIDTH/2) + 0.05 + BRACE_WIDTH + 0.02, y=seam_envelopes['MID_RL'], mode='text',
                text=label_text, textposition='middle right',
                textfont=dict(size=11, color='black'), hoverinfo='skip', showlegend=False
            ))
            
            overlap_threshold = max_depth_overall * 0.02
            pos_from = ['top left' if (r['FROM_RL'] - r['TO_RL']) < overlap_threshold else 'middle left' for _, r in seam_envelopes.iterrows()]
            pos_to = ['bottom left' if (r['FROM_RL'] - r['TO_RL']) < overlap_threshold else 'middle left' for _, r in seam_envelopes.iterrows()]

            fig.add_trace(go.Scatter(
                x=seam_envelopes['X_POS'] - LABEL_OFFSET, y=seam_envelopes['FROM_RL'], mode='markers+text',
                marker=dict(size=1, color='rgba(0,0,0,0)'),
                text=seam_envelopes['FMT_FROM'], textposition=pos_from,
                textfont=dict(size=10, color='black'), hoverinfo='skip', showlegend=False
            ))

            fig.add_trace(go.Scatter(
                x=seam_envelopes['X_POS'] - LABEL_OFFSET, y=seam_envelopes['TO_RL'], mode='markers+text',
                marker=dict(size=1, color='rgba(0,0,0,0)'),
                text=seam_envelopes['FMT_TO'], textposition=pos_to,
                textfont=dict(size=10, color='black'), hoverinfo='skip', showlegend=False
            ))

    fig.update_layout(
        title="Interactive Borehole Graphic Log",
        xaxis=dict(tickmode='array', tickvals=list(x_map.values()), ticktext=list(x_map.keys()), showgrid=False, zeroline=False),
        yaxis=dict(title="Elevation (RL) above MSL (m)", showgrid=True, zeroline=True),
        barmode='overlay', plot_bgcolor='white', hovermode='closest', height=800,
        margin=dict(l=100, r=80, t=100) 
    )
    return fig

def render_graphic_logs():
    st.markdown("## 📜 Graphic Logs & Lithology")
    
    # --- NEW LOGIC: Filter dropdown to only show CMPDI boreholes ---
    collar = st.session_state['df_collar']
    cmpdi_bhids = collar[collar['TYPE'] == 'CMPDI']['BHID'].unique()
    bhid_list = sorted([b for b in st.session_state['df_litho']['BHID'].unique() if b in cmpdi_bhids])
    
    col_sel, col_mode, col_pdf = st.columns([2, 1.5, 1])
    with col_sel:
        export_bhids = st.multiselect(
            "Select Boreholes to Visualize:", 
            bhid_list, 
            default=bhid_list[:3] if len(bhid_list) >= 3 else bhid_list
        )
    with col_mode:
        st.write("") 
        display_mode = st.radio(
            "Display Mode:", 
            ["All Lithology", "Only Coal Seams"], 
            horizontal=True
        )
        
    current_df_litho = st.session_state['df_litho']
    if display_mode == "Only Coal Seams":
        current_df_litho = current_df_litho[current_df_litho['IS_COAL']].copy()

    with col_pdf:
        st.write("") 
        if st.button("📄 Generate PDF Logs (1:500)", type="secondary", use_container_width=True):
            if export_bhids:
                with st.spinner("Rendering Lithology and Generating PDF..."):
                    pdf_data = generate_graphic_log_pdf(current_df_litho, st.session_state['df_collar'], export_bhids)
                    st.download_button(
                        label="⬇️ Download Graphic_Logs.pdf",
                        data=pdf_data, file_name="Graphic_Logs.pdf", mime="application/pdf",
                        use_container_width=True
                    )
            else:
                st.error("Please select at least one borehole.")
                
    if export_bhids:
        fig_log = plot_plotly_graphic_log(current_df_litho, st.session_state['df_collar'], export_bhids)
        st.plotly_chart(fig_log, use_container_width=True)

def render_seam_statistics():
    st.markdown("## 📈 Comprehensive Seam Statistics")
    df_litho = st.session_state['df_litho']
    df_collar = st.session_state['df_collar']
    
    # --- NEW LOGIC: Filter to only calculate stats for CMPDI boreholes ---
    df_collar = df_collar[df_collar['TYPE'] == 'CMPDI'].copy()
    valid_bhids = df_collar['BHID'].unique()
    
    # Only keep coal data if the borehole ID belongs to a CMPDI borehole
    raw_coal_df = df_litho[df_litho['IS_COAL'] & df_litho['BHID'].isin(valid_bhids)].copy()
    if raw_coal_df.empty:
        st.info("No coal seam data available to generate statistics.")
        return

    # BULLETPROOF MAPPING: Drop duplicates in collar data first to guarantee a unique index
    if 'DISPLAY_ID' not in raw_coal_df.columns:
        unique_collars = df_collar.drop_duplicates(subset=['BHID'])
        collar_mapping = unique_collars.set_index('BHID')['DISPLAY_ID']
        raw_coal_df['DISPLAY_ID'] = raw_coal_df['BHID'].map(collar_mapping)
        
    raw_coal_df['DISPLAY_ID'] = raw_coal_df['DISPLAY_ID'].fillna(raw_coal_df['BHID'])
    
    # --- CONTINUITY ENGINE FOR SEAM PARTS ---
    # ... (Keep the rest of your continuity engine and table logic exactly the same from here down)

    # --- CONTINUITY ENGINE FOR SEAM PARTS ---
    raw_coal_df = raw_coal_df.sort_values(['DISPLAY_ID', 'SEAM', 'FROM']).reset_index(drop=True)
    raw_coal_df['PREV_TO'] = raw_coal_df.groupby(['DISPLAY_ID', 'SEAM'])['TO'].shift(1)
    raw_coal_df['NEW_PART_FLAG'] = ((raw_coal_df['FROM'] - raw_coal_df['PREV_TO']) > 0.001).astype(int).fillna(0)
    raw_coal_df['PART_NO'] = raw_coal_df.groupby(['DISPLAY_ID', 'SEAM'])['NEW_PART_FLAG'].cumsum() + 1
    
    agg_parts_df = raw_coal_df.groupby(['DISPLAY_ID', 'SEAM', 'PART_NO']).agg({
        'FROM': 'min',
        'TO': 'max',
        'WIDTH': 'sum' 
    }).reset_index()
    
    agg_parts_df['PART_LABEL'] = agg_parts_df['SEAM'] + " (Part " + agg_parts_df['PART_NO'].astype(str) + ")"
    # ----------------------------------------

    view_mode = st.radio("Analysis Scope", ["Single Borehole", "Site Wide (All Boreholes)"], horizontal=True)
    st.markdown("---")

    if view_mode == "Single Borehole":
        st.markdown("### 🎯 Borehole-Centric Analysis")

        bh_list = sorted(agg_parts_df['DISPLAY_ID'].unique())
        col_sel1, col_sel2 = st.columns(2)
        
        with col_sel1:
            sel_bh = st.selectbox("Select Borehole / Point:", bh_list, key="stat_bh_sel")
        
        bh_data = agg_parts_df[agg_parts_df['DISPLAY_ID'] == sel_bh].copy()
        
        if not bh_data.empty:
            bh_data = bh_data.sort_values('FROM')
            seam_list = bh_data['SEAM'].unique().tolist()
            with col_sel2:
                sel_seam = st.selectbox("Select Coal Seam:", sorted(seam_list), key="stat_seam_sel")
            
            collar_info = df_collar[df_collar['DISPLAY_ID'] == sel_bh]
            if not collar_info.empty:
                bh_td = collar_info.iloc[0]['DEPTH']
                bh_rl = collar_info.iloc[0]['RL']
            else:
                bh_td, bh_rl = 0.0, 0.0

            total_coal = bh_data['WIDTH'].sum()
            
            st.markdown("#### Overall Borehole Summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Coal in Borehole", f"{total_coal:.2f} m")
            c2.metric("Borehole Depth (TD)", f"{bh_td:.2f} m")
            c3.metric("RL (Elevation)", f"{bh_rl:.2f} m")
            
            bh_seam_data = bh_data[bh_data['SEAM'] == sel_seam].copy()
            st.markdown(f"#### Selected Seam Highlights: {sel_seam}")
            
            if not bh_seam_data.empty:
                num_parts = len(bh_seam_data)
                seam_total_coal = bh_seam_data['WIDTH'].sum()
                roof_depth = bh_seam_data['FROM'].min()
                floor_depth = bh_seam_data['TO'].max()
                
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Seam parts Present", f"{num_parts}")
                sc2.metric(f"Total Coal thickness - seam - {sel_seam}", f"{seam_total_coal:.2f} m")
                sc3.metric("Roof and floor depth of the seam", f"{roof_depth:.2f} to {floor_depth:.2f} m")
            
            col_chart, col_table = st.columns([3, 2])
            
            with col_chart:
                fig_bh = go.Figure()
                chart_colors = ['#ff7f0e' if seam == sel_seam else '#1f77b4' for seam in bh_data['SEAM']]
                
                fig_bh.add_trace(go.Bar(
                    x=bh_data['PART_LABEL'], y=bh_data['WIDTH'], 
                    name=sel_bh, marker_color=chart_colors,
                    text=bh_data['WIDTH'].apply(lambda x: f"{x:.2f}m"), textposition='auto'
                ))
                
                ordered_parts = bh_data['PART_LABEL'].tolist()
                fig_bh.update_xaxes(categoryorder='array', categoryarray=ordered_parts)
                fig_bh.update_layout(
                    title=f"Continuous Seam Parts Profile (Sorted Shallow to Deep)", 
                    xaxis_title="Continuous Seam Parts", yaxis_title="Thickness (m)", 
                    plot_bgcolor='white', barmode='group'
                )
                fig_bh.update_yaxes(showgrid=True, gridcolor='LightGray')
                st.plotly_chart(fig_bh, use_container_width=True)

            with col_table:
                st.markdown("#### Detailed Intersections")
                disp_bh = bh_data[['SEAM', 'PART_NO', 'FROM', 'TO', 'WIDTH']].rename(columns={
                    'SEAM': 'Seam', 'PART_NO': 'Part', 'FROM': 'Roof (m)', 'TO': 'Floor (m)', 'WIDTH': 'Thickness (m)'
                })
                
                def highlight_seam_row(row):
                    if row['Seam'] == sel_seam:
                        return ['background-color: rgba(255, 127, 14, 0.2)'] * len(row)
                    return [''] * len(row)
                    
                st.dataframe(
                    disp_bh.style.apply(highlight_seam_row, axis=1).format("{:.2f}", subset=['Roof (m)', 'Floor (m)', 'Thickness (m)']), 
                    use_container_width=True, 
                    hide_index=True
                )
    else:
        st.markdown("### 🌍 Block Management Overview")
        
        # Calculate Site-Wide Top Level Metrics
        drilled_bhs = df_collar[df_collar['STATUS'].isin(['Closed', 'Running'])]
        avg_bh_depth = drilled_bhs['DEPTH'].mean() if not drilled_bhs.empty else 0
        max_bh_depth = drilled_bhs['DEPTH'].max() if not drilled_bhs.empty else 0
        total_bhs = len(df_collar)
        completed_bhs = len(df_collar[df_collar['STATUS'] == 'Closed'])
        total_coal_volume = agg_parts_df['WIDTH'].sum()
        
        # Calculate average coal thickness per closed borehole
        avg_coal_per_bh = total_coal_volume / completed_bhs if completed_bhs > 0 else 0
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Drilled Borehole Depth", f"{avg_bh_depth:.2f} m", help=f"Max Depth achieved: {max_bh_depth:.2f} m")
        c2.metric("Completion Ratio", f"{completed_bhs} / {total_bhs} BHs")
        c3.metric("Avg Coal Thickness / Closed Borehole", f"{avg_coal_per_bh:.2f} m")
        
        st.markdown("---")
        st.markdown("#### 📖 Comprehensive Seam Stratigraphy Summary")
        st.write("Chronological seam distribution, depths, and part thicknesses block-wide.")
        
        # 1. Aggregate Intersections (Count of unique boreholes that encountered the seam)
        intersections = agg_parts_df.groupby('SEAM')['DISPLAY_ID'].nunique().reset_index(name='INTERSECTIONS')
        
        # 2. Calculate Max Total Coal Thickness per seam (Sum of all parts in a BH, then Max across site)
        seam_bh_totals = agg_parts_df.groupby(['DISPLAY_ID', 'SEAM'])['WIDTH'].sum().reset_index()
        max_total_thickness = seam_bh_totals.groupby('SEAM')['WIDTH'].max().reset_index(name='MAX_TOTAL_THICKNESS')
        
        # 3. Aggregate Depth parameters (Roof and Floor)
        depth_stats = agg_parts_df.groupby('SEAM').agg({
            'FROM': ['min', 'max', 'mean'],
            'TO': ['min', 'max', 'mean']
        }).reset_index()
        
        depth_stats.columns = [
            'SEAM', 'MIN_ROOF', 'MAX_ROOF', 'AVG_ROOF', 
            'MIN_FLOOR', 'MAX_FLOOR', 'AVG_FLOOR'
        ]
        
        # 4. Identify the Max Thickness of a single part and the specific BHID
        idx_max = agg_parts_df.groupby('SEAM')['WIDTH'].idxmax()
        max_info = agg_parts_df.loc[idx_max, ['SEAM', 'WIDTH', 'DISPLAY_ID']]
        max_info.columns = ['SEAM', 'MAX_THICKNESS', 'MAX_THICKNESS_BHID']
        
        # 5. Merge all calculations into a single master summary
        site_stats = pd.merge(depth_stats, intersections, on='SEAM', how='left')
        site_stats = pd.merge(site_stats, max_total_thickness, on='SEAM', how='left')
        site_stats = pd.merge(site_stats, max_info, on='SEAM', how='left')
        site_stats = site_stats.sort_values('AVG_ROOF').reset_index(drop=True)
        
        # 6. Reorder and Rename columns for a professional geological table layout
        display_stats = site_stats[[
            'SEAM', 'INTERSECTIONS', 
            'MIN_ROOF', 'AVG_ROOF', 
            'AVG_FLOOR', 'MAX_FLOOR',
            'MAX_TOTAL_THICKNESS', 'MAX_THICKNESS', 'MAX_THICKNESS_BHID'
        ]].copy()
        
        display_stats.rename(columns={
            'SEAM': 'Coal Seam',
            'INTERSECTIONS': 'BH Intersections',
            'MIN_ROOF': 'Min Roof (m)',
            'AVG_ROOF': 'Avg Roof (m)',
            'AVG_FLOOR': 'Avg Floor (m)',
            'MAX_FLOOR': 'Max Floor (m)',
            'MAX_TOTAL_THICKNESS': 'Max Total Coal Thick. (m)',
            'MAX_THICKNESS': 'Max Thick. (m)',
            'MAX_THICKNESS_BHID': 'BHID (Max Thick.)'
        }, inplace=True)
        
        # Define strict CSS styles to force center alignment
        center_styles = [
            {'selector': 'th', 'props': [('text-align', 'center !important'), ('font-weight', 'bold !important')]},
            {'selector': 'td', 'props': [('text-align', 'center !important')]}
        ]

        float_cols = [
            'Min Roof (m)', 'Avg Roof (m)', 'Avg Floor (m)', 'Max Floor (m)', 
            'Max Total Coal Thick. (m)', 'Max Thick. (m)'
        ]
        
        styled_stats = (
            display_stats.style
            .format("{:.2f}", subset=float_cols)
            .set_properties(**{'text-align': 'center !important'})
            .set_table_styles(center_styles)
        )
        
        st.dataframe(
            styled_stats,
            use_container_width=True,
            hide_index=True
        )
# -----------------------------------------------------------------------------
# 5. MAIN TAB ARCHITECTURE
# -----------------------------------------------------------------------------

def main():
    st.title("⛰️ KUDANALI - LUBRI EXPLORATION")
    
    # Restructured tabs: Map is reinstated as an individual tab as requested
    tab_data, tab_dash, tab_map, tab_logs, tab_db, tab_stats = st.tabs([
        "1. Data Management",
        "2. Dashboard",
        "3. Map",  
        "4. Graphic Logs", 
        "5. Database",
        "6. Seam Statistics"
    ])
    
    with tab_data:
        st.markdown("Upload the required CSV files. **Collar, Boundary, and Lithology are mandatory.**")
        
        col_c, col_b, col_l = st.columns(3)
        with col_c: f_collar = st.file_uploader("1. Collar (Must contain ED, DOC, RIG NO, etc.)", type="csv")
        with col_b: f_boundary = st.file_uploader("2. Boundary", type="csv")
        with col_l: f_litho = st.file_uploader("3. Lithology", type="csv")
        
        st.write("")
        if st.button("🚀 Process Data", type="primary", use_container_width=True):
            if not f_collar or not f_boundary or not f_litho:
                st.error("All three files are required to process the data.")
            else:
                with st.spinner("Processing & Compositing datasets..."):
                    df_c, df_b, df_l = load_csv_safe(f_collar), load_csv_safe(f_boundary), load_csv_safe(f_litho)
                    
                    st.session_state['df_collar'] = process_collar(df_c)
                    st.session_state['df_boundary'] = process_boundary(df_b)
                    st.session_state['df_litho'] = process_litho(df_l)
                    st.success("Data successfully processed and composited!")
        
        st.markdown("---")
        stat_c = st.columns(3)
        stat_map = [
            ('Collar Data', 'df_collar' in st.session_state),
            ('Boundary Data', 'df_boundary' in st.session_state),
            ('Lithology Data', 'df_litho' in st.session_state)
        ]
        for i, (name, exists) in enumerate(stat_map):
            stat_c[i].metric(name, "✅ Ready" if exists else "❌ Missing")

    if 'df_collar' not in st.session_state or 'df_litho' not in st.session_state:
        return

    with tab_dash: render_dashboard()
    
    # Add unique key suffix for the main tab map to fix the ID collision error
    with tab_map: render_map(key_suffix="main_tab")
    
    with tab_logs: render_graphic_logs()
    with tab_stats: render_seam_statistics() 
        
    with tab_db:
        st.markdown("## 📋 Raw Composited Databases")
        t1, t2 = st.tabs(["Collar Database", "Composited Lithology Database"])
        
        with t1: 
            st.dataframe(st.session_state['df_collar'], use_container_width=True, height=500)
            
        with t2: 
            col_db1, col_db2 = st.columns([2, 1])
            with col_db1:
                db_bhid = st.multiselect("Filter by Borehole(s):", sorted(st.session_state['df_litho']['BHID'].unique()))
            with col_db2:
                st.write("") 
                show_only_coal_db = st.checkbox("Show Only Coal Seams in Database")
            
            df_display = st.session_state['df_litho'].copy()
            if db_bhid:
                df_display = df_display[df_display['BHID'].isin(db_bhid)]
            if show_only_coal_db:
                df_display = df_display[df_display['IS_COAL']]
                
            seams = st.session_state['df_litho'][st.session_state['df_litho']['IS_COAL']]['SEAM'].unique()
            seam_colors = generate_seam_colors(seams)
            
            def highlight_seams(row):
                if row.get('IS_COAL', False):
                    bg_color = seam_colors.get(row['SEAM'], '#D3D3D3')
                    return [f'background-color: {bg_color}; color: white; font-weight: bold; text-shadow: 1px 1px 2px rgba(0,0,0,0.8);'] * len(row)
                return [''] * len(row)
                
            st.dataframe(
                df_display.style.apply(highlight_seams, axis=1), 
                use_container_width=True, 
                height=500
            )

if __name__ == "__main__":
    main()
