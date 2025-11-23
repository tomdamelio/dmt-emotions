"""
TET Physiological Data Loader

This module provides functionality to load and merge preprocessed physiological
data (HR, SMNA AUC, RVT) with TET data for physiological-affective integration analysis.

The loader handles:
- Loading composite physiological data (already merged and z-scored with PC1)
- Loading and aggregating TET data to 30-second bins
- Merging datasets on (subject, state, dose, window)
- Temporal alignment and data validation

Author: TET Analysis Pipeline
Date: 2025
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TETPhysioDataLoader:
    """
    Load and merge physiological and TET data for integration analysis.
    
    This class handles loading preprocessed composite physiological data
    (HR, SMNA AUC, RVT with PC1) and TET data, then merging them on
    (subject, state, dose, window) for correlation and regression analysis.
    
    The composite physiological data is already:
    - Merged across all three measures (HR, SMNA_AUC, RVT)
    - Z-scored within subject
    - Aggregated to 30-second windows
    - PC1 (ArousalIndex) computed from PCA on (HR_z, SMNA_AUC_z, RVT_z)
    
    TET data is:
    - Originally at 4-second resolution (0.25 Hz)
    - Aggregated to 30-second bins to match physio resolution
    - Z-scored within subject (already done in preprocessing)
    
    Attributes:
        composite_physio_path: Path to composite physiological data CSV
        tet_path: Path to preprocessed TET data CSV
        target_bin_duration_sec: Target bin duration in seconds (30s)
        physio_data: Loaded physiological data
        tet_data: Loaded and aggregated TET data
        merged_data: Merged physio-TET dataset
    
    Example:
        >>> loader = TETPhysioDataLoader()
        >>> physio_df = loader.load_physiological_data()
        >>> tet_df = loader.load_tet_data()
        >>> merged_df = loader.merge_datasets()
        >>> print(f"Merged {len(merged_df)} timepoints")
    """
    
    def __init__(
        self,
        composite_physio_path: str = 'results/composite/arousal_index_long.csv',
        tet_path: str = 'results/tet/preprocessed/tet_preprocessed.csv',
        target_bin_duration_sec: int = 30
    ):
        """
        Initialize physiological-TET data loader.
        
        Args:
            composite_physio_path: Path to composite physiological data
                Contains: subject, window, State, Dose, SMNA_AUC, HR, RVT,
                         window_c, SMNA_AUC_z, HR_z, RVT_z, ArousalIndex (PC1)
            tet_path: Path to preprocessed TET data
            target_bin_duration_sec: Target bin duration in seconds (default: 30)
        """
        self.composite_physio_path = Path(composite_physio_path)
        self.tet_path = Path(tet_path)
        self.target_bin_duration_sec = target_bin_duration_sec
        
        # Data storage
        self.physio_data: Optional[pd.DataFrame] = None
        self.tet_data: Optional[pd.DataFrame] = None
        self.merged_data: Optional[pd.DataFrame] = None
        
        logger.info(f"Initialized TETPhysioDataLoader")
        logger.info(f"  Composite physio path: {self.composite_physio_path}")
        logger.info(f"  TET path: {self.tet_path}")
        logger.info(f"  Target bin duration: {self.target_bin_duration_sec}s")
    
    def load_physiological_data(self) -> pd.DataFrame:
        """
        Load composite physiological data.
        
        The composite file contains all three physiological measures already:
        - Merged across HR, SMNA_AUC, RVT
        - Z-scored within subject
        - Aggregated to 30-second windows
        - PC1 (ArousalIndex) computed from PCA
        
        Expected columns:
        - subject: Subject ID (S01-S20)
        - window: Window index (1, 2, 3, ..., 18 for first 9 minutes)
        - State: Experimental state (DMT or RS)
        - Dose: Dose level (High or Low)
        - SMNA_AUC: Raw skin conductance AUC (μS·s)
        - HR: Raw heart rate (bpm)
        - RVT: Raw respiratory volume per time (L/s)
        - window_c: Centered window index
        - SMNA_AUC_z: Z-scored SMNA AUC
        - HR_z: Z-scored HR
        - RVT_z: Z-scored RVT
        - ArousalIndex: PC1 from PCA on (HR_z, SMNA_AUC_z, RVT_z)
        
        Returns:
            DataFrame with composite physiological data
            
        Raises:
            FileNotFoundError: If composite physio file not found
            ValueError: If required columns are missing
        """
        if not self.composite_physio_path.exists():
            raise FileNotFoundError(
                f"Composite physiological data not found: {self.composite_physio_path}"
            )
        
        logger.info(f"Loading composite physiological data from {self.composite_physio_path}")
        
        # Load data
        physio_df = pd.read_csv(self.composite_physio_path)
        
        # Validate required columns
        required_cols = [
            'subject', 'window', 'State', 'Dose',
            'SMNA_AUC', 'HR', 'RVT',
            'SMNA_AUC_z', 'HR_z', 'RVT_z',
            'ArousalIndex'
        ]
        missing_cols = set(required_cols) - set(physio_df.columns)
        if missing_cols:
            raise ValueError(
                f"Missing required columns in physiological data: {missing_cols}"
            )
        
        # Log data summary
        n_subjects = physio_df['subject'].nunique()
        n_windows = physio_df['window'].nunique()
        n_total = len(physio_df)
        
        logger.info(f"Loaded physiological data:")
        logger.info(f"  N subjects: {n_subjects}")
        logger.info(f"  N windows: {n_windows}")
        logger.info(f"  N total observations: {n_total}")
        logger.info(f"  States: {physio_df['State'].unique()}")
        logger.info(f"  Doses: {physio_df['Dose'].unique()}")
        
        # Store data
        self.physio_data = physio_df
        
        return physio_df
    
    def load_tet_data(self) -> pd.DataFrame:
        """
        Load preprocessed TET data and aggregate to 30-second bins.
        
        Process:
        1. Load preprocessed TET data (4-second resolution, 0.25 Hz)
        2. Aggregate to 30-second bins to match physio resolution
           - Group by (subject, session_id, state, dose) and floor(t_sec / 30)
           - Compute mean of each dimension within each 30-second window
           - Each 30-second bin contains ~7.5 original timepoints (30s / 4s)
        3. Extract z-scored affective dimensions
        4. Compute valence_index_z = pleasantness_z - unpleasantness_z
        5. Create window column to match physio data (window = 1, 2, 3, ..., 18)
        6. Filter to first 9 minutes (18 windows) to match physio data
        
        Expected bin counts after aggregation:
        - RS: 150 timepoints @ 4s → 20 bins @ 30s (but only first 18 used)
        - DMT: 300 timepoints @ 4s → 40 bins @ 30s (but only first 18 used)
        
        Returns:
            DataFrame with aggregated TET data
            Columns: subject, session_id, state, dose, window, t_sec,
                    pleasantness_z, unpleasantness_z, emotional_intensity_z,
                    interoception_z, bliss_z, anxiety_z, valence_index_z
                    
        Raises:
            FileNotFoundError: If TET data file not found
            ValueError: If required columns are missing
        """
        if not self.tet_path.exists():
            raise FileNotFoundError(
                f"TET data not found: {self.tet_path}"
            )
        
        logger.info(f"Loading TET data from {self.tet_path}")
        
        # Load data
        tet_df = pd.read_csv(self.tet_path)
        
        # Validate required columns
        required_cols = [
            'subject', 'session_id', 'state', 'dose', 't_sec',
            'pleasantness_z', 'unpleasantness_z', 'emotional_intensity_z',
            'interoception_z', 'bliss_z', 'anxiety_z'
        ]
        missing_cols = set(required_cols) - set(tet_df.columns)
        if missing_cols:
            raise ValueError(
                f"Missing required columns in TET data: {missing_cols}"
            )
        
        logger.info(f"Original TET data: {len(tet_df)} timepoints @ 4s resolution")
        
        # Aggregate to 30-second bins
        logger.info(f"Aggregating TET data to {self.target_bin_duration_sec}s bins")
        
        # Create window index (1-based to match physio data)
        tet_df['window'] = (tet_df['t_sec'] // self.target_bin_duration_sec) + 1
        
        # Define affective dimensions to aggregate
        affective_dims = [
            'pleasantness_z', 'unpleasantness_z', 'emotional_intensity_z',
            'interoception_z', 'bliss_z', 'anxiety_z'
        ]
        
        # Group by session and window, compute mean
        groupby_cols = ['subject', 'session_id', 'state', 'dose', 'window']
        agg_dict = {dim: 'mean' for dim in affective_dims}
        agg_dict['t_sec'] = 'min'  # Use start time of window
        
        tet_agg = tet_df.groupby(groupby_cols, as_index=False).agg(agg_dict)
        
        # Compute valence index
        tet_agg['valence_index_z'] = (
            tet_agg['pleasantness_z'] - tet_agg['unpleasantness_z']
        )
        
        # Filter to first 9 minutes (18 windows @ 30s = 540s)
        max_window = 18
        tet_agg = tet_agg[tet_agg['window'] <= max_window].copy()
        
        # Log aggregation summary
        n_subjects = tet_agg['subject'].nunique()
        n_sessions = tet_agg.groupby('subject')['session_id'].nunique().mean()
        n_windows = tet_agg['window'].nunique()
        n_total = len(tet_agg)
        
        logger.info(f"Aggregated TET data:")
        logger.info(f"  N subjects: {n_subjects}")
        logger.info(f"  Avg sessions per subject: {n_sessions:.1f}")
        logger.info(f"  N windows: {n_windows}")
        logger.info(f"  N total observations: {n_total}")
        logger.info(f"  Time range: 0-{tet_agg['t_sec'].max()}s")
        
        # Store data
        self.tet_data = tet_agg
        
        return tet_agg
    
    def merge_datasets(self) -> pd.DataFrame:
        """
        Merge physiological and TET data on (subject, state, dose, window).
        
        Process:
        1. Standardize naming conventions:
           - Physio: State (DMT/RS), Dose (High/Low), window (1-18)
           - TET: state (DMT/RS), dose (Alta/Baja), window (1-18)
           - Map Dose: High→Alta, Low→Baja
           - Map State: keep as is (DMT/RS)
        2. Merge on (subject, State, Dose, window) using inner join
        3. Drop rows with any missing values in key variables
        4. Log merge statistics
        
        Expected final N per subject:
        - ~72 windows @ 30s bins (18 RS + 18 DMT per session × 2 sessions)
        - Note: Only first 9 minutes analyzed (18 windows × 30s = 540s)
        
        Returns:
            DataFrame with merged physio-TET data
            Columns:
            - subject, state, dose, window, window_c (centered window)
            - HR_z, SMNA_AUC_z, RVT_z (physiological measures, z-scored)
            - ArousalIndex (PC1 from physio PCA)
            - pleasantness_z, unpleasantness_z, emotional_intensity_z,
              interoception_z, bliss_z, anxiety_z (TET affective, z-scored)
            - valence_index_z (computed)
            
        Raises:
            ValueError: If physio or TET data not loaded
        """
        if self.physio_data is None:
            raise ValueError("Physiological data not loaded. Call load_physiological_data() first.")
        if self.tet_data is None:
            raise ValueError("TET data not loaded. Call load_tet_data() first.")
        
        logger.info("Merging physiological and TET datasets")
        
        # Create copies for merging
        physio_df = self.physio_data.copy()
        tet_df = self.tet_data.copy()
        
        # Standardize dose naming: TET uses Alta/Baja, physio uses High/Low
        dose_map = {'High': 'Alta', 'Low': 'Baja'}
        physio_df['Dose'] = physio_df['Dose'].map(dose_map)
        
        # Standardize column names for merging
        physio_df = physio_df.rename(columns={'State': 'state', 'Dose': 'dose'})
        
        # Log pre-merge counts
        logger.info(f"Pre-merge counts:")
        logger.info(f"  Physio: {len(physio_df)} observations")
        logger.info(f"  TET: {len(tet_df)} observations")
        
        # Merge on (subject, state, dose, window)
        merge_cols = ['subject', 'state', 'dose', 'window']
        merged_df = pd.merge(
            physio_df,
            tet_df,
            on=merge_cols,
            how='inner',
            suffixes=('_physio', '_tet')
        )
        
        # Drop rows with missing values in key variables
        key_vars = [
            'HR_z', 'SMNA_AUC_z', 'RVT_z', 'ArousalIndex',
            'pleasantness_z', 'unpleasantness_z', 'emotional_intensity_z',
            'interoception_z', 'bliss_z', 'anxiety_z', 'valence_index_z'
        ]
        n_before = len(merged_df)
        merged_df = merged_df.dropna(subset=key_vars)
        n_after = len(merged_df)
        n_dropped = n_before - n_after
        
        # Log merge statistics
        n_subjects = merged_df['subject'].nunique()
        n_sessions = merged_df.groupby('subject')['session_id'].nunique().sum()
        n_windows = len(merged_df)
        pct_retained_physio = (n_windows / len(physio_df)) * 100
        pct_retained_tet = (n_windows / len(tet_df)) * 100
        
        logger.info(f"Merge complete:")
        logger.info(f"  N subjects: {n_subjects}")
        logger.info(f"  N sessions: {n_sessions}")
        logger.info(f"  N windows (30s bins): {n_windows}")
        logger.info(f"  Rows dropped (missing data): {n_dropped}")
        logger.info(f"  % retained from physio: {pct_retained_physio:.1f}%")
        logger.info(f"  % retained from TET: {pct_retained_tet:.1f}%")
        
        # Log distribution by state and dose
        logger.info(f"Distribution by condition:")
        for state in merged_df['state'].unique():
            for dose in merged_df['dose'].unique():
                n = len(merged_df[(merged_df['state'] == state) & (merged_df['dose'] == dose)])
                logger.info(f"  {state} × {dose}: {n} windows")
        
        # Store merged data
        self.merged_data = merged_df
        
        return merged_df
    
    def get_affective_columns(self) -> list:
        """
        Get list of TET affective dimension column names.
        
        Returns:
            List of affective dimension column names
        """
        return [
            'pleasantness_z',
            'unpleasantness_z',
            'emotional_intensity_z',
            'interoception_z',
            'bliss_z',
            'anxiety_z'
        ]
    
    def get_physio_columns(self) -> list:
        """
        Get list of physiological measure column names.
        
        Returns:
            List of physiological measure column names
        """
        return ['HR_z', 'SMNA_AUC_z', 'RVT_z']
    
    def load_and_merge_high_resolution(self) -> pd.DataFrame:
        """
        Load TET at original 4s resolution and interpolate physio data to match.
        
        This method provides an alternative to the standard 30s binning approach:
        1. Load TET data at original 4-second resolution (no aggregation)
        2. Load physiological data at 30-second resolution
        3. Interpolate physiological data to 4-second resolution
        4. Merge on (subject, state, dose, time_sec)
        
        This preserves the temporal dynamics of TET while maintaining
        physiological signal continuity through interpolation.
        
        Returns:
            DataFrame with merged data at 4-second resolution
            
        Note:
            - Interpolation uses linear method for physiological signals
            - This assumes physiological signals change smoothly over time
            - Results in ~7.5x more observations than 30s binning
        """
        logger.info("\n" + "=" * 80)
        logger.info("Loading data at HIGH RESOLUTION (4-second bins)")
        logger.info("=" * 80)
        
        # Load TET data without aggregation
        logger.info("\nLoading TET data at original 4s resolution...")
        tet_df = pd.read_csv(self.tet_path)
        
        # Filter to first 9 minutes (540 seconds)
        tet_df = tet_df[tet_df['t_sec'] <= 540].copy()
        
        logger.info(f"  Loaded {len(tet_df)} TET timepoints @ 4s resolution")
        logger.info(f"  Subjects: {tet_df['subject'].nunique()}")
        logger.info(f"  Time range: 0-{tet_df['t_sec'].max()}s")
        
        # Load physiological data (30s resolution)
        logger.info("\nLoading physiological data @ 30s resolution...")
        physio_df = self.load_physiological_data()
        
        # Convert window to time_sec for physio (window 1 = 0-30s, window 2 = 30-60s, etc.)
        physio_df['t_sec'] = (physio_df['window'] - 1) * 30
        
        # Prepare for interpolation
        logger.info("\nInterpolating physiological data to 4s resolution...")
        
        # Create list to store interpolated data
        interpolated_dfs = []
        
        # Interpolate for each subject-session combination
        for (subject, state, dose), group in physio_df.groupby(['subject', 'State', 'Dose']):
            # Sort by time
            group = group.sort_values('t_sec')
            
            # Create target time points (every 4 seconds from 0 to 540)
            target_times = np.arange(0, 544, 4)  # 0, 4, 8, ..., 540
            
            # Interpolate each physiological variable
            interp_data = {
                'subject': subject,
                'state': state,
                'dose': dose,
                't_sec': target_times
            }
            
            # Interpolate z-scored physiological measures
            for col in ['HR_z', 'SMNA_AUC_z', 'RVT_z', 'ArousalIndex']:
                interp_data[col] = np.interp(
                    target_times,
                    group['t_sec'].values,
                    group[col].values
                )
            
            interpolated_dfs.append(pd.DataFrame(interp_data))
        
        # Combine all interpolated data
        physio_interp = pd.concat(interpolated_dfs, ignore_index=True)
        
        logger.info(f"  Interpolated to {len(physio_interp)} timepoints @ 4s resolution")
        
        # Standardize column names for merging
        tet_df = tet_df.rename(columns={'state': 'state_tet'})
        tet_df['state'] = tet_df['state_tet'].str.upper()  # Ensure uppercase
        
        # Map dose names
        dose_map = {'Alta': 'High', 'Baja': 'Low'}
        tet_df['dose'] = tet_df['dose'].map(dose_map)
        
        # Round t_sec to avoid floating point issues
        tet_df['t_sec'] = tet_df['t_sec'].round(0).astype(int)
        physio_interp['t_sec'] = physio_interp['t_sec'].round(0).astype(int)
        
        # Merge on (subject, state, dose, t_sec)
        logger.info("\nMerging TET and interpolated physiological data...")
        merged_df = pd.merge(
            tet_df,
            physio_interp,
            on=['subject', 'state', 'dose', 't_sec'],
            how='inner'
        )
        
        # Drop rows with missing values
        n_before = len(merged_df)
        merged_df = merged_df.dropna(subset=[
            'HR_z', 'SMNA_AUC_z', 'RVT_z', 'ArousalIndex',
            'pleasantness_z', 'unpleasantness_z', 'emotional_intensity_z',
            'interoception_z', 'bliss_z', 'anxiety_z'
        ])
        n_after = len(merged_df)
        n_dropped = n_before - n_after
        
        # Log merge statistics
        logger.info(f"\nMerge complete:")
        logger.info(f"  Total observations: {len(merged_df)}")
        logger.info(f"  Subjects: {merged_df['subject'].nunique()}")
        logger.info(f"  Rows dropped (missing data): {n_dropped}")
        logger.info(f"  Time resolution: 4 seconds")
        logger.info(f"  Time range: 0-{merged_df['t_sec'].max()}s")
        
        # Log distribution by state and dose
        logger.info(f"\nDistribution by condition:")
        for state in merged_df['state'].unique():
            for dose in merged_df['dose'].unique():
                n = len(merged_df[(merged_df['state'] == state) & (merged_df['dose'] == dose)])
                logger.info(f"  {state} × {dose}: {n} timepoints")
        
        # Store merged data
        self.merged_data = merged_df
        
        return merged_df
    
    def export_merged_data(self, output_path: str) -> str:
        """
        Export merged dataset to CSV.
        
        Args:
            output_path: Path to save merged data CSV
            
        Returns:
            Path to exported file
            
        Raises:
            ValueError: If merged data not available
        """
        if self.merged_data is None:
            raise ValueError("Merged data not available. Call merge_datasets() first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.merged_data.to_csv(output_path, index=False)
        logger.info(f"Exported merged data to {output_path}")
        
        return str(output_path)


if __name__ == '__main__':
    # Example usage
    loader = TETPhysioDataLoader()
    
    # Load data
    physio_df = loader.load_physiological_data()
    tet_df = loader.load_tet_data()
    
    # Merge datasets
    merged_df = loader.merge_datasets()
    
    # Export merged data
    output_path = 'results/tet/physio_correlation/merged_physio_tet_data.csv'
    loader.export_merged_data(output_path)
    
    print(f"\nMerged dataset shape: {merged_df.shape}")
    print(f"Columns: {list(merged_df.columns)}")
