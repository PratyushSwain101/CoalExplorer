"""
loader.py
Master data model for the Coal Explorer application.
"""
import pandas as pd
import numpy as np
from typing import Tuple

class ProjectData:
    """
    Centralized data model that ingests CSVs, derives calculated attributes,
    and exposes standardized dataframes for the application.
    """
    def __init__(self, collar_file, boundary_file, litho_file):
        self.collar = pd.read_csv(collar_file)
        self.boundary = pd.read_csv(boundary_file)
        self.litho = pd.read_csv(litho_file)
        
        self.seams = []
        self._process_data()

    def _process_data(self) -> None:
        """Executes all data cleaning and derivations upon instantiation."""
        # 1. Standardize and clean Status
        if 'STATUS' in self.collar.columns:
            self.collar['STATUS'] = self.collar['STATUS'].fillna('Pending').astype(str).str.title()
        else:
            self.collar['STATUS'] = 'Unknown'

        # 2. Generate DISPLAY_ID
        self.collar['DISPLAY_ID'] = self.collar.apply(
            lambda row: f"{row['POINT_ID']} / {row['BHID']}" 
            if pd.notna(row.get('BHID')) and str(row.get('BHID')).strip() else str(row['POINT_ID']), 
            axis=1
        )

        # 3. Process Lithology & Coal Intersections
        if 'SEAM' in self.litho.columns:
            # Clean seam names
            self.litho['SEAM'] = self.litho['SEAM'].fillna('')
            self.litho['IS_COAL'] = self.litho['SEAM'].astype(str).str.strip() != ''
            
            # Extract unique seams dynamically
            self.seams = self.litho[self.litho['IS_COAL']]['SEAM'].unique().tolist()
            
            # Calculate total coal per BHID
            coal_summary = self.litho[self.litho['IS_COAL']].groupby('BHID')['WIDTH'].sum().reset_index(name='TOTAL_COAL')
            
            # Count number of seams per BHID
            seam_count = self.litho[self.litho['IS_COAL']].groupby('BHID')['SEAM'].nunique().reset_index(name='NO_OF_SEAMS')
            
            # Merge calculations back to collar
            self.collar = pd.merge(self.collar, coal_summary, on='BHID', how='left')
            self.collar = pd.merge(self.collar, seam_count, on='BHID', how='left')
            self.collar['TOTAL_COAL'] = self.collar['TOTAL_COAL'].fillna(0.0)
            self.collar['NO_OF_SEAMS'] = self.collar['NO_OF_SEAMS'].fillna(0).astype(int)
            
            # Calculate Depth Intervals for Coal
            self.litho['MID_DEPTH'] = (self.litho['FROM'] + self.litho['TO']) / 2
            
            def assign_depth_interval(depth):
                if depth <= 100: return '0-100 m'
                if depth <= 200: return '100-200 m'
                if depth <= 300: return '200-300 m'
                return '> 300 m'
                
            self.litho['DEPTH_INTERVAL'] = self.litho['MID_DEPTH'].apply(assign_depth_interval)