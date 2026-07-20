"""
config.py
Centralized configuration for the Coal Explorer application.
"""

# Default mapping for status colors
STATUS_COLORS = {
    'Closed': '#00CC96',  # Green
    'Running': '#FFA15A', # Orange
    'Pending': '#AB63FA', # Purple/Grey
    'Unknown': '#B6E880'
}

# Standard geological colors for non-coal lithology (fallback)
LITHO_COLORS = {
    'Soil': '#8B4513',
    'Sandstone': '#F4A460',
    'Shale': '#708090',
    'Siltstone': '#A0522D',
    'Mudstone': '#696969',
    'Carbonaceous Shale': '#2F4F4F'
}

# Coal properties
COAL_COLOR = '#111111'
COAL_LINE_COLOR = '#000000'

# Application settings
APP_TITLE = "Coal Exploration Dashboard"
LAYOUT = "wide"