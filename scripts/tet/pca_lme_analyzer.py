# -*- coding: utf-8 -*-
"""
TET PCA Linear Mixed Effects (LME) Analysis Module

This module provides functionality for fitting LME models to principal component
scores from PCA analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
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
except ImportError:
    raise ImportError("statsmodels is required. Install with: pip install statsmodels")

# Configure logging
logger = logging.getLogger(__name__)


class TETPCALMEAnalyzer:
    """
    Fits Linear Mixed Effects models for principal component scores.
    
    This class fits LME models to test dose and state effects on PC scores,
    using the same model structure as the original dimension analysis.
    
    Model formula:
        PC ~ State + Dose + Time_c + State:Dose + State:Time_c + Dose:Time_c + (1|Subject)
    
    Attributes:
        pc_scores (pd.DataFrame): PC scores from TETPCAAnalyzer
        components (List[str]): List of component names to analyze (default: ['PC1', 'PC2'])
        models (Dict): Fitted models for each component
        results_df (pd.DataFrame): Combined results from all models
    
    Example:
        >>> from tet.pca_lme_analyzer import TETPCALMEAnalyzer
        >>> import pandas as pd
        >>> 
        >>> # Load PC scores
        >>> pc_scores = pd.read_csv('results/tet/pca/pca_scores.csv')
        >>> 
        >>> # Fit LME models for PC1 and PC2
        >>> analyzer = TETPCALMEAnalyzer(pc_scores)
        >>> results = analyzer.fit_pc_models()
        >>> 
        >>> # Export results
        >>> analyzer.export_results('results/tet/pca')
    """
    
    def __init__(self, pc_scores: pd.DataFrame, components: List[str] = None):
        """
        Initialize LME analyzer for PC scores.
        
        Args:
            pc_scores (pd.DataFrame): DataFrame with PC scores from TETPCAAnalyzer.
                Must contain columns: subject, session_id, state, dose, t_bin, t_sec,
                and component columns (PC1, PC2, etc.)
            components (List[str], optional): List of components to analyze.
                Default: ['PC1', 'PC2']
        """
        self.pc_scores = pc_scores.copy()
        
        # Default to PC1 and PC2
        if components is None:
            self.components = ['PC1', 'PC2']
        else:
            self.components = components
        
        # Validate that components exist in data
        missing_components = [c for c in self.components if c not in self.pc_scores.columns]
        if missing_components:
            raise ValueError(f"Components not found in data: {missing_components}")
        
        # Storage for fitted models and results
        self.models = {}  # Dict[component, MixedLMResults]
        self.results_df = None
        
        logger.info(f"Initialized TETPCALMEAnalyzer with {len(self.components)} components")
        logger.info(f"Components: {', '.join(self.components)}")

    def prepare_pc_data(self) -> pd.DataFrame:
        """
        Prepare PC scores for LME analysis.
        
        This method:
        1. Filters to t_bin 0-18 (0-9 minutes) for consistency with original LME analysis
        2. Centers time within each state-session: Time_c = t_sec - mean(t_sec)
        3. Ensures categorical variables are properly encoded:
           - state: categorical with RS as reference
           - dose: categorical with Baja as reference
        
        Returns:
            pd.DataFrame: Prepared data ready for LME fitting
        
        Example:
            >>> analyzer = TETPCALMEAnalyzer(pc_scores)
            >>> prepared_data = analyzer.prepare_pc_data()
            >>> # prepared_data now has time_c column and proper categorical encoding
        """
        logger.info("Preparing PC data for LME analysis...")
        
        # Filter to 0-9 minutes (t_bin 0-18) for consistency with original analysis
        data_lme = self.pc_scores[self.pc_scores['t_bin'] <= 18].copy()
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

    def fit_pc_models(self) -> Dict[str, Any]:
        """
        Fit LME models for each principal component.
        
        Model specification (same as Requirement 4):
            PC ~ state * dose * time_c + (1 | subject)
        
        Fixed effects:
        - state: DMT vs RS
        - dose: Alta vs Baja
        - time_c: Centered time
        - state:dose: State-dose interaction
        - state:time_c: State-time interaction
        - dose:time_c: Dose-time interaction
        - state:dose:time_c: Three-way interaction (if included in formula)
        
        Random effects:
        - (1 | subject): Random intercept per subject
        
        Returns:
            Dict[str, MixedLMResults]: Dictionary mapping component name to fitted model
        
        Example:
            >>> analyzer = TETPCALMEAnalyzer(pc_scores)
            >>> models = analyzer.fit_pc_models()
            >>> # models['PC1'] contains the fitted model for PC1
        """
        logger.info(f"Fitting LME models for {len(self.components)} components...")
        
        # Prepare data
        data_lme = self.prepare_pc_data()
        
        # Fit model for each component
        for i, component in enumerate(self.components, 1):
            logger.info(f"  [{i}/{len(self.components)}] Fitting {component}...")
            
            # Model formula - using Treatment coding for reference levels
            formula = (
                f"{component} ~ C(state, Treatment('RS')) * C(dose, Treatment('Baja')) * time_c"
            )
            
            try:
                # Fit mixed linear model
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    
                    model = smf.mixedlm(
                        formula=formula,
                        data=data_lme,
                        groups=data_lme['subject'],
                        re_formula='1'  # Random intercept only
                    )
                    
                    # Fit with REML method (default)
                    result = model.fit(method='lbfgs', maxiter=100)
                
                # Check convergence
                if not result.converged:
                    logger.warning(f"    Model for {component} did not converge")
                    continue
                
                # Store fitted model
                self.models[component] = result
                logger.info(f"    ✓ Model fitted successfully")
                
            except Exception as e:
                logger.error(f"    ✗ Error fitting model for {component}: {e}")
                continue
        
        logger.info(f"Successfully fitted {len(self.models)}/{len(self.components)} models")
        
        return self.models

    def extract_results(self) -> pd.DataFrame:
        """
        Extract coefficients, standard errors, and p-values from fitted models.
        
        For each fitted model:
        - Extract coefficients (beta)
        - Extract standard errors (se)
        - Calculate z-values
        - Extract p-values
        - Calculate 95% confidence intervals
        - Create row for each fixed effect
        
        Returns:
            pd.DataFrame: Results with columns:
                - component: Component name (PC1, PC2)
                - effect: Fixed effect name
                - beta: Coefficient estimate
                - se: Standard error
                - z_value: Z-statistic
                - p_value: P-value
                - ci_lower: 95% CI lower bound
                - ci_upper: 95% CI upper bound
        
        Example:
            >>> analyzer = TETPCALMEAnalyzer(pc_scores)
            >>> analyzer.fit_pc_models()
            >>> results = analyzer.extract_results()
            >>> # results contains coefficients for all components
        """
        if not self.models:
            raise ValueError("No models fitted. Run fit_pc_models() first.")
        
        logger.info("Extracting results from fitted models...")
        
        all_results = []
        
        for component, result in self.models.items():
            logger.info(f"  Extracting results for {component}...")
            
            # Extract parameters
            params = result.params
            conf_int = result.conf_int()
            pvalues = result.pvalues
            
            # Calculate standard errors and z-values
            bse = result.bse
            z_values = params / bse
            
            # Create results DataFrame for this component
            component_results = pd.DataFrame({
                'component': component,
                'effect': params.index,
                'beta': params.values,
                'se': bse.values,
                'z_value': z_values.values,
                'p_value': pvalues.values,
                'ci_lower': conf_int[0].values,
                'ci_upper': conf_int[1].values
            })
            
            all_results.append(component_results)
        
        # Combine all results
        self.results_df = pd.concat(all_results, ignore_index=True)
        
        logger.info(f"Extracted results: {len(self.results_df)} rows")
        
        return self.results_df

    def export_results(self, output_dir: str) -> Dict[str, str]:
        """
        Export LME results for PC scores to CSV.
        
        Creates:
        - pca_lme_results.csv: Coefficients and p-values for all components
        
        Args:
            output_dir (str): Directory to save output files
        
        Returns:
            Dict[str, str]: Dictionary mapping file types to file paths
        
        Example:
            >>> analyzer = TETPCALMEAnalyzer(pc_scores)
            >>> analyzer.fit_pc_models()
            >>> analyzer.extract_results()
            >>> paths = analyzer.export_results('results/tet/pca')
            >>> # paths['lme_results'] contains the path to pca_lme_results.csv
        """
        if self.results_df is None:
            raise ValueError("No results to export. Run extract_results() first.")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Export results
        results_path = os.path.join(output_dir, 'pca_lme_results.csv')
        self.results_df.to_csv(results_path, index=False)
        
        logger.info(f"Exported PCA LME results to: {results_path}")
        
        return {'lme_results': results_path}
