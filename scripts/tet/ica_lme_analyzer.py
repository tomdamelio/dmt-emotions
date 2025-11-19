"""
Linear Mixed Effects (LME) modeling for ICA component scores.

This module fits LME models to IC scores to test for State, Dose, and Time effects,
following the same approach as PCA LME analysis.
"""

from typing import Dict, List
import pandas as pd
import numpy as np
from statsmodels.regression.mixed_linear_model import MixedLM
from pathlib import Path


class TETICALMEAnalyzer:
    """
    Fit Linear Mixed Effects models to ICA component scores.
    
    Model specification:
        IC ~ C(state, Treatment('RS')) * C(dose, Treatment('Baja')) * time_c
        Random effects: (1 | subject)
    
    This allows testing whether independent components show significant
    State, Dose, or interaction effects, similar to PCA analysis.
    
    Example:
        >>> analyzer = TETICALMEAnalyzer(
        ...     ic_scores=ic_scores_df,
        ...     components=['IC1', 'IC2']
        ... )
        >>> prepared_data = analyzer.prepare_ic_data()
        >>> models = analyzer.fit_ic_models()
        >>> results = analyzer.extract_results()
        >>> analyzer.export_results('results/tet/ica')
    """
    
    def __init__(
        self,
        ic_scores: pd.DataFrame,
        components: List[str] = None
    ):
        """
        Initialize ICA LME analyzer.
        
        Args:
            ic_scores: DataFrame with IC scores and metadata
            components: List of IC components to analyze (default: ['IC1', 'IC2'])
        """
        self.ic_scores = ic_scores.copy()
        self.components = components if components is not None else ['IC1', 'IC2']
        
        # Storage for results
        self.models = {}  # Dict mapping component to fitted model
        self.results_df = None
    
    def prepare_ic_data(self) -> pd.DataFrame:
        """
        Prepare IC scores for LME modeling.
        
        Process:
        1. Center time within each state-session: Time_c = t_sec - mean(t_sec)
        2. Ensure categorical variables properly encoded:
           - state: RS as reference level
           - dose: Baja as reference level
        
        Returns:
            Prepared DataFrame ready for LME modeling
        """
        data = self.ic_scores.copy()
        
        # Center time within each state-session combination
        data['time_c'] = data.groupby(['state', 'session_id'])['t_sec'].transform(
            lambda x: x - x.mean()
        )
        
        # Ensure categorical encoding with correct reference levels
        # state: RS as reference
        data['state'] = pd.Categorical(
            data['state'],
            categories=['RS', 'DMT'],
            ordered=False
        )
        
        # dose: Baja as reference
        data['dose'] = pd.Categorical(
            data['dose'],
            categories=['Baja', 'Alta'],
            ordered=False
        )
        
        return data
    
    def fit_ic_models(self) -> Dict[str, MixedLM]:
        """
        Fit LME models for specified IC components.
        
        Model specification:
            IC ~ C(state, Treatment('RS')) * C(dose, Treatment('Baja')) * time_c
            Random effects: (1 | subject)
        
        Fixed effects:
        - State (DMT vs RS)
        - Dose (Alta vs Baja)
        - Time_c (centered time)
        - State:Dose interaction
        - State:Time_c interaction
        - Dose:Time_c interaction
        - State:Dose:Time_c three-way interaction
        
        Returns:
            Dict mapping component name to fitted MixedLM model
        """
        # Prepare data
        data = self.prepare_ic_data()
        
        # Fit model for each component
        for component in self.components:
            if component not in data.columns:
                print(f"Warning: Component {component} not found in data. Skipping.")
                continue
            
            # Define formula
            formula = f"{component} ~ C(state, Treatment('RS')) * C(dose, Treatment('Baja')) * time_c"
            
            try:
                # Fit LME model with random intercept for subject
                model = MixedLM.from_formula(
                    formula=formula,
                    data=data,
                    groups=data['subject'],
                    re_formula='1'
                )
                
                # Fit using REML
                fitted_model = model.fit(reml=True)
                
                self.models[component] = fitted_model
                
            except Exception as e:
                print(f"Error fitting model for {component}: {e}")
                continue
        
        return self.models
    
    def extract_results(self) -> pd.DataFrame:
        """
        Extract LME results for all fitted models.
        
        For each model:
        1. Extract coefficients, standard errors, z-values, p-values
        2. Calculate 95% confidence intervals
        3. Create row for each fixed effect
        
        Returns:
            DataFrame with columns:
            - component: IC component name (IC1, IC2, ...)
            - effect: Fixed effect name
            - beta: Coefficient estimate
            - se: Standard error
            - z_value: Z-statistic
            - p_value: P-value
            - ci_lower: Lower 95% CI
            - ci_upper: Upper 95% CI
        """
        if not self.models:
            raise ValueError("No models fitted. Call fit_ic_models() first.")
        
        results = []
        
        for component, model in self.models.items():
            # Extract parameter estimates
            params = model.params
            std_errors = model.bse
            z_values = model.tvalues
            p_values = model.pvalues
            conf_int = model.conf_int()
            
            # Create row for each fixed effect
            for effect_name in params.index:
                results.append({
                    'component': component,
                    'effect': effect_name,
                    'beta': params[effect_name],
                    'se': std_errors[effect_name],
                    'z_value': z_values[effect_name],
                    'p_value': p_values[effect_name],
                    'ci_lower': conf_int.loc[effect_name, 0],
                    'ci_upper': conf_int.loc[effect_name, 1]
                })
        
        self.results_df = pd.DataFrame(results)
        return self.results_df
    
    def export_results(self, output_dir: str) -> Dict[str, str]:
        """
        Export LME results to CSV file.
        
        Creates:
        - ica_lme_results.csv: LME coefficients for all IC components
        
        Args:
            output_dir: Directory to save output files
        
        Returns:
            Dict mapping file types to file paths
        """
        if self.results_df is None:
            raise ValueError("No results to export. Call extract_results() first.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export LME results
        lme_path = output_path / 'ica_lme_results.csv'
        self.results_df.to_csv(lme_path, index=False)
        
        return {'lme_results': str(lme_path)}
