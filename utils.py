"""
utils.py
Utility functions for the Coal Explorer application.
"""
import plotly.express as px
from typing import List, Dict

def generate_seam_colors(seams: List[str]) -> Dict[str, str]:
    """
    Dynamically generates a distinct color palette for detected coal seams.
    Returns a dictionary mapping seam names to hex colors.
    """
    cleaned_seams = sorted([str(s) for s in seams if s and str(s).strip() != 'nan'])
    colors = px.colors.qualitative.Dark24 + px.colors.qualitative.Alphabet
    
    seam_color_map = {}
    for i, seam in enumerate(cleaned_seams):
        seam_color_map[seam] = colors[i % len(colors)]
        
    return seam_color_map