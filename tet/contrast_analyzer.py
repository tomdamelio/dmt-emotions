# -*- coding: utf-8 -*-
"""
TET Contrast Analysis Module

This module provides functionality for computing dose contrasts within states
from fitted LME models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging
import sys
import os
from scipy import stats

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Statistical packages
try:
    from statsmodels.stats.multitest import multipletests
except ImportError:
    raise ImportError("statsmodels is required. Install with: pip install statsmodels")

# Configure logging
logger = logging.getLogger(__name__)


class TETContrastAnalyzer:
    """
    Computes dose contrasts within states from fitted LME models.
    
    This class computes:
    1. DMT High vs Low: Effect of high dose in DMT state
    2. RS High vs Low: Effect of high dose in RS state
    
    Attributes:
        models (Dict): Fitted LME models for each dimension
        contrasts (pd.DataFrame): Computed contrasts
    
    Example:
        >>> from tet.contrast_analyzer import TETContrastAnalyzer
        >>> 
        >>> # Assuming you have fitted models from TETLMEAnalyzer
        >>> contrast_analyzer = TETContrastAnalyzer(lme_analyzer.models)
        >>> contrasts = contrast_analyzer.compute_all_contrasts()
        >>> contrast_analyzer.export_contrasts('results/tet/lme')
    """
    
    def __init__(self, models: Dict[str, Any]):
        """
        Initialize contrast analyzer.
        
        Args:
            models (Dict[str, Any]): Dictionary of fitted LME models
                {dimension_name: fitted_model}
        """
        self.models = models
        self.contrasts = None
        
        logger.info(f"Initialized TETContrastAnalyzer with {len(models)} models")
    
    def compute_contrast(
        self,
        dimension: str,
        result: Any,
        contrast_type: str
    ) -> Dict[str, Any]:
        """
        Compute a single contrast.
        
        Args:
            dimension (str): Dimension name
            result: Fitted model result
            contrast_type (str): 'dmt_high_vs_low' or 'rs_high_vs_low'
            
        Returns:
            Dict with contrast results
        """
        try:
            if contrast_type == 'dmt_high_vs_low':
                # DMT High vs Low = beta_dose + beta_state:dose
                # This gives the dose effect specifically in DMT state
                
                beta_dose = result.params.get('dose[T.Alta]', 0)
                beta_interaction = result.params.get('state[T.DMT]:dose[T.Alta]', 0)
                
                estimate = beta_dose + beta_interaction
                
                # Compute SE using variance-covariance matrix
                vcov = result.cov_params()
                
                var_dose = vcov.loc['dose[T.Alta]', 'dose[T.Alta]']
                var_interaction = vcov.loc['state[T.DMT]:dose[T.Alta]', 'state[T.DMT]:dose[T.Alta]']
                cov_dose_interaction = vcov.loc['dose[T.Alta]', 'state[T.DMT]:dose[T.Alta]']
                
                var_contrast = var_dose + var_interaction + 2 * cov_dose_interaction
                se = np.sqrt(var_contrast)
                
            elif contrast_type == 'rs_high_vs_low':
                # RS High vs Low = beta_dose
                # This gives the dose effect specifically in RS state (reference)
                
                estimate = result.params.get('dose[T.Alta]', 0)
                se = result.bse.get('dose[T.Alta]', np.nan)
                
            else:
                raise ValueError(f"Unknown contrast type: {contrast_type}")
            
            # Compute confidence interval and p-value
            z_stat = estimate / se if se > 0 else np.nan
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat))) if not np.isnan(z_stat) else np.nan
            
            ci_lower = estimate - 1.96 * se
            ci_upper = estimate + 1.96 * se
            
            return {
                'dimension': dimension,
                'contrast': contrast_type,
                'estimate': estimate,
                'se': se,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'p_value': p_value
            }
            
        except Exception as e:
            logger.error(f"Error computing {contrast_type} for {dimension}: {e}")
            return {
                'dimension': dimension,
                'contrast': contrast_type,
                'estimate': np.nan,
                'se': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'p_value': np.nan
            }
    
    def compute_all_contrasts(self) -> pd.DataFrame:
        """
        Compute all contrasts for all dimensions.
        
        Returns:
            pd.DataFrame: Contrasts with columns:
                - dimension, contrast, estimate, se, ci_lower, ci_upper, p_value, p_fdr
        """
        logger.info(f"Computing contrasts for {len(self.models)} dimensions...")
        
        all_contrasts = []
        
        for dimension, result in self.models.items():
            # DMT High vs Low
            dmt_contrast = self.compute_contrast(dimension, result, 'dmt_high_vs_low')
            all_contrasts.append(dmt_contrast)
            
            # RS High vs Low
            rs_contrast = self.compute_contrast(dimension, result, 'rs_high_vs_low')
            all_contrasts.append(rs_contrast)
        
        # Convert to DataFrame
        self.contrasts = pd.DataFrame(all_contrasts)
        
        # Apply FDR correction separately for each contrast type
        self._apply_fdr_correction()
        
        logger.info(f"Computed {len(self.contrasts)} contrasts")
        
        return self.contrasts
    
    def _apply_fdr_correction(self):
        """
        Apply Benjamini-Hochberg FDR correction to contrasts.
        
        Correction is applied separately for each contrast type.
        """
        logger.info("Applying FDR correction to contrasts...")
        
        for contrast_type in self.contrasts['contrast'].unique():
            mask = self.contrasts['contrast'] == contrast_type
            pvalues = self.contrasts.loc[mask, 'p_value'].values
            
            # Apply BH-FDR correction
            _, p_fdr, _, _ = multipletests(
                pvalues,
                alpha=0.05,
                method='fdr_bh'
            )
            
            # Store corrected p-values
            self.contrasts.loc[mask, 'p_fdr'] = p_fdr
            
            # Mark significant results
            self.contrasts.loc[mask, 'significant'] = p_fdr < 0.05
            
            n_sig = (p_fdr < 0.05).sum()
            logger.info(f"  {contrast_type}: {n_sig}/{len(p_fdr)} significant after FDR correction")
    
    def export_contrasts(
        self,
        output_dir: str,
        filename: str = 'lme_contrasts.csv'
    ) -> str:
        """
        Export contrasts to CSV.
        
        Args:
            output_dir (str): Output directory path
            filename (str): Output filename (default: 'lme_contrasts.csv')
            
        Returns:
            str: Path to exported file
        """
        if self.contrasts is None:
            raise ValueError("No contrasts to export. Run compute_all_contrasts() first.")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Export to CSV
        output_path = os.path.join(output_dir, filename)
        self.contrasts.to_csv(output_path, index=False)
        
        logger.info(f"Exported contrasts to: {output_path}")
        
        return output_path
    
    def get_summary(self) -> pd.DataFrame:
        """
        Get summary of significant contrasts.
        
        Returns:
            pd.DataFrame: Summary with counts per contrast type
        """
        if self.contrasts is None:
            raise ValueError("No contrasts available. Run compute_all_contrasts() first.")
        
        summary = self.contrasts[self.contrasts['significant'] == True].groupby('contrast').size().reset_index()
        summary.columns = ['contrast', 'n_significant']
        
        return summary
