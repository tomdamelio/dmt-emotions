# -*- coding: utf-8 -*-
"""
TET Linear Mixed Effects (LME) Analysis Module

This module provides functionality for fitting LME models to test dose and state
effects on TET dimensions.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import logging
import sys
import os
import warnings

# Add project root to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config

# Statistical packages
try:
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests
except ImportError:
    raise ImportError("statsmodels is required. Install with: pip install statsmodels")

# Configure logging
logger = logging.getLogger(__name__)


class TETLMEAnalyzer:
    """
    Fits Linear Mixed Effects models for TET dimensions.
    
    This class fits LME models to test dose and state effects on each dimension,
    controlling for repeated measures within subjects.
    
    Model formula:
        Y ~ State + Dose + Time_c + State:Dose + State:Time_c + Dose:Time_c + (1|Subject)
    
    Attributes:
        data (pd.DataFrame): Preprocessed TET data
        dimensions (List[str]): List of z-scored dimension names to analyze
        models (Dict): Fitted models for each dimension
        results (pd.DataFrame): Combined results from all models
    
    Example:
        >>> from tet.lme_analyzer import TETLMEAnalyzer
        >>> import pandas as pd
        >>> 
        >>> # Load preprocessed data
        >>> data = pd.read_csv('results/tet/preprocessed/tet_preprocessed.csv')
        >>> 
        >>> # Fit LME models
        >>> analyzer = TETLMEAnalyzer(data)
        >>> results = analyzer.fit_all_dimensions()
        >>> 
        >>> # Export results
        >>> analyzer.export_results('results/tet/lme')
    """
    
    def __init__(self, data: pd.DataFrame, dimensions: Optional[List[str]] = None):
        """
        Initialize LME analyzer.
        
        Args:
            data (pd.DataFrame): Preprocessed TET data
            dimensions (Optional[List[str]]): List of dimensions to analyze.
                If None, uses affective dimensions + valence_index_z.
        """
        self.data_raw = data
        
        # Default: z-scored affective dimensions + composite index (not all dimensions)
        if dimensions is None:
            # Use only affective dimensions + valence index
            affective_dims = [f"{dim}_z" for dim in config.TET_AFFECTIVE_COLUMNS]
            self.dimensions = affective_dims + ['valence_index_z']
        else:
            self.dimensions = dimensions
        
        # Prepare data for LME
        self.data_lme = self._prepare_data()
        
        # Storage for fitted models and results
        self.models = {}
        self.results = None
        
        logger.info(f"Initialized TETLMEAnalyzer with {len(self.dimensions)} dimensions")
        logger.info(f"Analysis window: t_bin 0-18 (0-540 seconds, 0-9 minutes)")
    
    def _prepare_data(self) -> pd.DataFrame:
        """
        Prepare data for LME analysis.
        
        Returns:
            pd.DataFrame: Prepared data with:
                - Filtered to t_bin 0-18 (0-9 minutes)
                - Centered time variable (time_c)
                - Categorical variables with reference levels set
        """
        logger.info("Preparing data for LME analysis...")
        
        # Filter to 0-9 minutes (t_bin 0-18)
        data_lme = self.data_raw[self.data_raw['t_bin'] <= 18].copy()
        logger.info(f"  Filtered to {len(data_lme)} rows (t_bin 0-18)")
        
        # Center time variable
        data_lme['time_c'] = data_lme['t_bin'] - data_lme['t_bin'].mean()
        logger.info(f"  Centered time: mean = {data_lme['time_c'].mean():.6f}")
        
        # Set categorical variables with reference levels
        # Reference: State='RS', Dose='Baja'
        data_lme['state'] = pd.Categorical(
            data_lme['state'],
            categories=['RS', 'DMT'],
            ordered=False
        )
        data_lme['dose'] = pd.Categorical(
            data_lme['dose'],
            categories=['Baja', 'Alta'],
            ordered=False
        )
        
        logger.info("  Reference levels: State='RS', Dose='Baja'")
        logger.info("  Data preparation complete")
        
        return data_lme
    
    def fit_lme(self, dimension: str) -> Optional[Any]:
        """
        Fit LME model for a single dimension.
        
        Args:
            dimension (str): Dimension column name (z-scored)
            
        Returns:
            Fitted model result or None if fitting failed
        """
        if dimension not in self.data_lme.columns:
            raise ValueError(f"Dimension '{dimension}' not found in data")
        
        # Model formula
        formula = f"{dimension} ~ state + dose + time_c + state:dose + state:time_c + dose:time_c"
        
        try:
            # Fit mixed linear model
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                
                model = smf.mixedlm(
                    formula=formula,
                    data=self.data_lme,
                    groups=self.data_lme['subject'],
                    re_formula='1'  # Random intercept only
                )
                
                # Fit with ML (REML=False for maximum likelihood)
                result = model.fit(reml=False, maxiter=100)
            
            # Check convergence
            if not result.converged:
                logger.warning(f"  Model for {dimension} did not converge")
                return None
            
            return result
            
        except Exception as e:
            logger.error(f"  Error fitting model for {dimension}: {e}")
            return None
    
    def extract_results(self, dimension: str, result: Any) -> pd.DataFrame:
        """
        Extract results from fitted model.
        
        Args:
            dimension (str): Dimension name
            result: Fitted model result
            
        Returns:
            pd.DataFrame: Results with columns:
                - dimension, effect, beta, ci_lower, ci_upper, p_value
        """
        # Extract parameters
        params = result.params
        conf_int = result.conf_int()
        pvalues = result.pvalues
        
        # Create results DataFrame
        results = pd.DataFrame({
            'dimension': dimension,
            'effect': params.index,
            'beta': params.values,
            'ci_lower': conf_int[0].values,
            'ci_upper': conf_int[1].values,
            'p_value': pvalues.values
        })
        
        # Add model diagnostics
        results['aic'] = result.aic
        results['bic'] = result.bic
        results['llf'] = result.llf
        
        return results
    
    def fit_all_dimensions(self) -> pd.DataFrame:
        """
        Fit LME models for all dimensions.
        
        Returns:
            pd.DataFrame: Combined results from all models with FDR correction
        """
        logger.info(f"Fitting LME models for {len(self.dimensions)} dimensions...")
        
        all_results = []
        
        for i, dimension in enumerate(self.dimensions, 1):
            logger.info(f"  [{i}/{len(self.dimensions)}] Fitting {dimension}...")
            
            # Fit model
            result = self.fit_lme(dimension)
            
            if result is not None:
                # Store fitted model
                self.models[dimension] = result
                
                # Extract results
                dim_results = self.extract_results(dimension, result)
                all_results.append(dim_results)
                
                logger.info(f"    ✓ Model fitted successfully")
            else:
                logger.warning(f"    ✗ Model fitting failed")
        
        # Combine all results
        if len(all_results) > 0:
            self.results = pd.concat(all_results, ignore_index=True)
            logger.info(f"Combined results: {len(self.results)} rows")
            
            # Apply FDR correction
            self._apply_fdr_correction()
            
            return self.results
        else:
            logger.error("No models were successfully fitted")
            return pd.DataFrame()
    
    def _apply_fdr_correction(self):
        """
        Apply Benjamini-Hochberg FDR correction.
        
        Correction is applied separately for each fixed effect across all dimensions.
        """
        logger.info("Applying FDR correction...")
        
        # Get unique effects (exclude Intercept and random effects)
        effects = self.results['effect'].unique()
        effects = [e for e in effects if e != 'Intercept' and 'Group' not in e]
        
        # Apply FDR correction for each effect
        for effect in effects:
            effect_mask = self.results['effect'] == effect
            effect_pvalues = self.results.loc[effect_mask, 'p_value'].values
            
            # Apply BH-FDR correction
            _, p_fdr, _, _ = multipletests(
                effect_pvalues,
                alpha=0.05,
                method='fdr_bh'
            )
            
            # Store corrected p-values
            self.results.loc[effect_mask, 'p_fdr'] = p_fdr
            
            # Mark significant results
            self.results.loc[effect_mask, 'significant'] = p_fdr < 0.05
            
            n_sig = (p_fdr < 0.05).sum()
            logger.info(f"  {effect}: {n_sig}/{len(p_fdr)} significant after FDR correction")
        
        # For Intercept, just copy p_value to p_fdr
        intercept_mask = self.results['effect'] == 'Intercept'
        self.results.loc[intercept_mask, 'p_fdr'] = self.results.loc[intercept_mask, 'p_value']
        self.results.loc[intercept_mask, 'significant'] = False
    
    def export_results(
        self,
        output_dir: str,
        filename: str = 'lme_results.csv'
    ) -> str:
        """
        Export LME results to CSV.
        
        Args:
            output_dir (str): Output directory path
            filename (str): Output filename (default: 'lme_results.csv')
            
        Returns:
            str: Path to exported file
        """
        if self.results is None:
            raise ValueError("No results to export. Run fit_all_dimensions() first.")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Export to CSV
        output_path = os.path.join(output_dir, filename)
        self.results.to_csv(output_path, index=False)
        
        logger.info(f"Exported LME results to: {output_path}")
        
        return output_path
    
    def get_summary(self) -> pd.DataFrame:
        """
        Get summary of significant effects.
        
        Returns:
            pd.DataFrame: Summary with counts of significant effects per dimension
        """
        if self.results is None:
            raise ValueError("No results available. Run fit_all_dimensions() first.")
        
        # Count significant effects per dimension
        summary = self.results[self.results['significant'] == True].groupby('dimension').size().reset_index()
        summary.columns = ['dimension', 'n_significant_effects']
        
        return summary
