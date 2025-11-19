"""
Quick test for TETPhysioCCAAnalyzer implementation.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.tet.physio_cca_analyzer import TETPhysioCCAAnalyzer


def create_synthetic_data(n_obs=200):
    """Create synthetic merged physio-TET data for testing."""
    np.random.seed(42)
    
    # Create two states with different patterns
    data_list = []
    
    for state in ['RS', 'DMT']:
        # Physiological measures (correlated)
        hr = np.random.randn(n_obs) * 10 + 70
        smna_auc = np.random.randn(n_obs) * 2 + 5
        rvt = np.random.randn(n_obs) * 0.5 + 0.3
        
        # TET affective dimensions (correlated with physio)
        # Stronger correlation in DMT state
        correlation_strength = 0.6 if state == 'DMT' else 0.3
        
        emotional_intensity_z = (
            correlation_strength * (hr - 70) / 10 + 
            (1 - correlation_strength) * np.random.randn(n_obs)
        )
        anxiety_z = (
            correlation_strength * (smna_auc - 5) / 2 + 
            (1 - correlation_strength) * np.random.randn(n_obs)
        )
        pleasantness_z = -0.3 * emotional_intensity_z + np.random.randn(n_obs) * 0.8
        unpleasantness_z = 0.4 * anxiety_z + np.random.randn(n_obs) * 0.8
        interoception_z = 0.5 * emotional_intensity_z + np.random.randn(n_obs) * 0.7
        bliss_z = -0.2 * anxiety_z + np.random.randn(n_obs) * 0.9
        
        state_data = pd.DataFrame({
            'subject': [f'S{i:02d}' for i in range(1, n_obs + 1)],
            'session_id': [f'{state}_{i % 2 + 1}' for i in range(n_obs)],
            'state': state,
            'dose': ['High' if i % 2 == 0 else 'Low' for i in range(n_obs)],
            't_bin': [i % 20 for i in range(n_obs)],
            't_sec': [(i % 20) * 30 for i in range(n_obs)],
            'HR': hr,
            'SMNA_AUC': smna_auc,
            'RVT': rvt,
            'pleasantness_z': pleasantness_z,
            'unpleasantness_z': unpleasantness_z,
            'emotional_intensity_z': emotional_intensity_z,
            'interoception_z': interoception_z,
            'bliss_z': bliss_z,
            'anxiety_z': anxiety_z
        })
        
        data_list.append(state_data)
    
    return pd.concat(data_list, ignore_index=True)


def test_cca_analyzer():
    """Test TETPhysioCCAAnalyzer with synthetic data."""
    print("=" * 80)
    print("Testing TETPhysioCCAAnalyzer")
    print("=" * 80)
    
    # Create synthetic data
    print("\n1. Creating synthetic data...")
    merged_data = create_synthetic_data(n_obs=200)
    print(f"   Created data: {merged_data.shape}")
    print(f"   States: {merged_data['state'].unique()}")
    print(f"   Columns: {list(merged_data.columns)}")
    
    # Initialize analyzer
    print("\n2. Initializing TETPhysioCCAAnalyzer...")
    analyzer = TETPhysioCCAAnalyzer(merged_data)
    print(f"   Physio measures: {analyzer.physio_measures}")
    print(f"   TET affective: {analyzer.tet_affective}")
    
    # Test matrix preparation
    print("\n3. Testing matrix preparation...")
    try:
        X_rs, Y_rs = analyzer.prepare_matrices('RS')
        print(f"   RS matrices: X shape {X_rs.shape}, Y shape {Y_rs.shape}")
        print(f"   X mean: {X_rs.mean(axis=0)}")
        print(f"   X std: {X_rs.std(axis=0)}")
        
        X_dmt, Y_dmt = analyzer.prepare_matrices('DMT')
        print(f"   DMT matrices: X shape {X_dmt.shape}, Y shape {Y_dmt.shape}")
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    # Test CCA fitting
    print("\n4. Testing CCA fitting...")
    try:
        cca_models = analyzer.fit_cca(n_components=2)
        print(f"   Fitted CCA for {len(cca_models)} states")
        for state, model in cca_models.items():
            corrs = analyzer.canonical_correlations[state]
            print(f"   {state}: canonical correlations = {corrs}")
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    # Test canonical variate extraction
    print("\n5. Testing canonical variate extraction...")
    try:
        variates_df = analyzer.extract_canonical_variates()
        print(f"   Extracted {len(variates_df)} canonical variates")
        print("\n   Results:")
        print(variates_df.to_string(index=False))
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    # Test canonical loading computation
    print("\n6. Testing canonical loading computation...")
    try:
        loadings_df = analyzer.compute_canonical_loadings()
        print(f"   Computed {len(loadings_df)} loadings")
        print("\n   Sample loadings (CV1, RS):")
        sample = loadings_df[
            (loadings_df['state'] == 'RS') & 
            (loadings_df['canonical_variate'] == 1)
        ].sort_values('loading', ascending=False)
        print(sample.to_string(index=False))
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    # Test results export
    print("\n7. Testing results export...")
    try:
        output_dir = 'test/tet/test_output_cca'
        file_paths = analyzer.export_results(output_dir)
        print(f"   Exported {len(file_paths)} files:")
        for file_type, path in file_paths.items():
            print(f"   - {file_type}: {path}")
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
    return True


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    
    success = test_cca_analyzer()
    sys.exit(0 if success else 1)
