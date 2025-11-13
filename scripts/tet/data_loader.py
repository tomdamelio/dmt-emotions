# -*- coding: utf-8 -*-
"""
TET Data Loader Module

This module provides functionality for loading Temporal Experience Tracking (TET)
data from CSV files or .mat files with validation of required columns.
"""

import logging
import pandas as pd
import numpy as np
import scipy.io
from typing import List, Optional
from pathlib import Path
import sys
import os
import re

# Add project root to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TETDataLoader:
    """
    Loads TET (Temporal Experience Tracking) data from CSV or .mat files.
    
    This class handles loading TET data from either:
    - A single CSV file with all data consolidated
    - A directory of .mat files (one per subject/session)
    
    Attributes:
        file_path (str): Path to CSV file or directory with .mat files
        required_columns (List[str]): List of required column names
        mat_dir (Optional[str]): Directory containing .mat files
    
    Example:
        >>> # Load from CSV
        >>> loader = TETDataLoader('data/tet/tet_data.csv')
        >>> data = loader.load_data()
        
        >>> # Load from .mat directory
        >>> loader = TETDataLoader(mat_dir='../data/original/reports/resampled')
        >>> data = loader.load_data()
    """
    
    def __init__(self, file_path: Optional[str] = None, mat_dir: Optional[str] = None):
        """
        Initialize the TET data loader.
        
        Args:
            file_path (str, optional): Path to the TET data CSV file
            mat_dir (str, optional): Path to directory containing .mat files
        """
        self.file_path = file_path
        self.mat_dir = mat_dir
        self.required_columns = config.TET_REQUIRED_COLUMNS
        
        if not file_path and not mat_dir:
            raise ValueError("Must provide either file_path or mat_dir")
    
    def _parse_filename(self, filename: str) -> dict:
        """
        Parse .mat filename to extract metadata.
        
        Expected format: s01_DMT_Session1_DMT.mat or s01_RS_Session1_EC.mat
        
        Args:
            filename (str): Name of .mat file
            
        Returns:
            dict: Metadata with keys: subject, state, session_id, dose
        """
        # Remove extension
        name = filename.replace('.mat', '')
        
        # Parse pattern: s01_DMT_Session1_DMT or s01_RS_Session1_EC
        parts = name.split('_')
        
        subject = parts[0].upper()  # s01 -> S01
        state = 'DMT' if 'DMT' in name else 'RS'
        
        # Extract session number
        session_match = re.search(r'Session(\d+)', name)
        session_id = int(session_match.group(1)) if session_match else None
        
        # Get dose from config
        dose = None
        if subject in config.SUJETOS_INDICES and session_id:
            dose = config.get_dosis_sujeto(subject, session_id)
        
        return {
            'subject': subject,
            'state': state,
            'session_id': session_id,
            'dose': dose
        }
    
    def _load_mat_file(self, mat_path: str) -> pd.DataFrame:
        """
        Load a single .mat file and convert to DataFrame.
        
        The .mat files contain a 'dimensions' matrix of shape (n_bins, 15)
        where the 15 columns represent phenomenological dimensions in a
        specific order. Column names are assigned from config.TET_DIMENSION_COLUMNS
        which follows the order documented in docs/PIPELINE.md.
        
        Args:
            mat_path (str): Path to .mat file
            
        Returns:
            pd.DataFrame: Data from .mat file with metadata columns
        """
        # Load .mat file
        mat_data = scipy.io.loadmat(mat_path)
        
        # Extract dimensions array (shape: n_bins x 15)
        dimensions = mat_data['dimensions']
        
        # Parse filename for metadata
        filename = os.path.basename(mat_path)
        metadata = self._parse_filename(filename)
        
        # Create DataFrame with dimension names from config
        # NOTE: Column order is critical and must match the order in the .mat files
        # See config.TET_DIMENSION_COLUMNS and docs/PIPELINE.md for documentation
        df = pd.DataFrame(dimensions, columns=config.TET_DIMENSION_COLUMNS)
        
        # Add metadata columns
        df['subject'] = metadata['subject']
        df['session_id'] = metadata['session_id']
        df['state'] = metadata['state']
        df['dose'] = metadata['dose']
        
        # Add time information
        # t_bin: índice temporal (0, 1, 2, ...)
        # t_sec: tiempo exacto en segundos (0, 4, 8, 12, ...)
        # Resolución: 0.25 Hz = 1 punto cada 4 segundos
        df['t_bin'] = np.arange(len(df))
        df['t_sec'] = df['t_bin'] * config.TET_SAMPLING_INTERVAL_SEC
        
        # Reorder columns to match expected format
        column_order = ['subject', 'session_id', 'state', 'dose', 't_bin', 't_sec'] + config.TET_DIMENSION_COLUMNS
        df = df[column_order]
        
        return df
    
    def _load_from_mat_directory(self) -> pd.DataFrame:
        """
        Load all .mat files from directory and consolidate.
        
        Returns:
            pd.DataFrame: Consolidated data from all .mat files
        """
        mat_path = Path(self.mat_dir)
        
        if not mat_path.exists():
            raise FileNotFoundError(f"Directory not found: {self.mat_dir}")
        
        # Find all .mat files
        mat_files = list(mat_path.glob('*.mat'))
        
        if not mat_files:
            raise FileNotFoundError(f"No .mat files found in: {self.mat_dir}")
        
        logger.info(f"Found {len(mat_files)} .mat files in {self.mat_dir}")
        
        # Load each file
        dfs = []
        for mat_file in sorted(mat_files):
            try:
                df = self._load_mat_file(str(mat_file))
                dfs.append(df)
                logger.debug(f"Loaded {mat_file.name}: {len(df)} rows")
            except Exception as e:
                logger.warning(f"Failed to load {mat_file.name}: {e}")
                continue
        
        if not dfs:
            raise ValueError("No valid .mat files could be loaded")
        
        # Concatenate all DataFrames
        data = pd.concat(dfs, ignore_index=True)
        
        logger.info(f"Consolidated {len(dfs)} files into {len(data)} rows")
        
        return data
    
    def load_data(self) -> pd.DataFrame:
        """
        Load TET data from CSV file or .mat directory with validation.
        
        This method reads data from either a CSV file or a directory of .mat files,
        validates that all required columns are present, and returns the data as 
        a pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame containing TET data with columns for
                subject, session_id, state, dose, t_bin, t_sec, and 15
                dimension rating columns
        
        Raises:
            FileNotFoundError: If the specified file/directory does not exist
            ValueError: If required columns are missing from the data
            pandas.errors.ParserError: If the CSV file is malformed
        
        Example:
            >>> # Load from CSV
            >>> loader = TETDataLoader('data/tet/tet_data.csv')
            >>> data = loader.load_data()
            
            >>> # Load from .mat directory
            >>> loader = TETDataLoader(mat_dir='../data/original/reports/resampled')
            >>> data = loader.load_data()
        """
        try:
            # Determine source type and load data
            if self.mat_dir:
                logger.info(f"Loading TET data from .mat directory: {self.mat_dir}")
                data = self._load_from_mat_directory()
            else:
                logger.info(f"Loading TET data from CSV: {self.file_path}")
                data = pd.read_csv(self.file_path)
            
            # Validate required columns are present
            missing_columns = set(self.required_columns) - set(data.columns)
            
            if missing_columns:
                missing_cols_str = ', '.join(sorted(missing_columns))
                error_msg = (
                    f"Missing required columns in TET data: {missing_cols_str}. "
                    f"Expected columns: {', '.join(self.required_columns)}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Log successful load
            n_rows = len(data)
            n_subjects = data['subject'].nunique() if 'subject' in data.columns else 0
            n_sessions = len(data.groupby(['subject', 'session_id'])) if all(
                col in data.columns for col in ['subject', 'session_id']
            ) else 0
            
            logger.info(
                f"Successfully loaded TET data: {n_rows} rows, "
                f"{n_subjects} subjects, {n_sessions} sessions"
            )
            
            return data
            
        except FileNotFoundError:
            error_msg = (
                f"TET data source not found: {self.file_path or self.mat_dir}. "
                f"Please ensure the file/directory exists at the specified path."
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        except pd.errors.ParserError as e:
            error_msg = (
                f"Failed to parse TET data file: {self.file_path}. "
                f"The CSV file may be malformed or corrupted. "
                f"Parser error: {str(e)}"
            )
            logger.error(error_msg)
            raise pd.errors.ParserError(error_msg)
        
        except Exception as e:
            # Catch any other unexpected errors
            error_msg = (
                f"Unexpected error loading TET data: "
                f"{type(e).__name__}: {str(e)}"
            )
            logger.error(error_msg)
            raise
