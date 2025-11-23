"""
Quick test for redundancy index computation.

This script tests the redundancy index computation methods added to
TETPhysioCCAAnalyzer.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from scripts.tet.physio_cca_analyzer import TETPhysioCCAAnalyzer


def create_synthetic_data(n_subjects=10, n_timepoints=18):
    """Create synthetic physiological-TET data for testing."""
    np.random.seed(42)
    
    records = []
    
    for subject_id in range(1, n_subjects + 1):
        for state in ['RS', 'DMT']:
            for t_bin in range(n_timepoints):
                # Create correlated physio and TET data
                base_arousal = np.random.randn()
                
                record = {
                    'subject': f'S{subject_id:02d}',
                    'state': state,
                    't_bin': t_bin,
                    # Physiological measures (correlated with arousal)
                    'HR': base_arousal + np.random.randn() * 0.5,
                    'SMNA_AUC': base_arousal * 0.8 + np.random.randn() * 0.5,
                    'RVT': base_arousal * 0.6 + np.random.randn() * 0.5,
                    # TET affective dimensions (correlated with arousal)
                    'pleasantness_z': -base_arousal * 0.5 + np.random.randn() * 0.5,
                    'unpleasantness_z': base_arousal * 0.7 + np.random.randn() * 0.5,
                    'emotional_intensity_z': abs(base_arousal) + np.random.randn() * 0.3,
                    'interoception_z': base_arousal * 0.4 + np.random.randn() * 0.5,
                    'bliss_z': -base_arousal * 0.6 + np.random.randn() * 0.5,
                    'anxiety_z': base_arousal * 0.8 + np.random.randn() * 0.5,
                }
                
                records.append(record)
    
    return pd.DataFrame(records)


def test_redundancy_computation():
    """Test redundancy index computation."""
    print("=" * 80)
    print("Testing Redundancy Index Computation")
    print("=" * 80)
    
    # Create synthetic data
    print("\n1. Creating synthetic data...")
    data = create_synthetic_data(n_subjects=10, n_timepoints=18)
    print(f"   Created data: {len(data)} observations")
    print(f"   Subjects: {data['subject'].nunique()}")
    print(f"   States: {data['state'].unique()}")
    
    # Initialize analyzer
    print("\n2. Initializing CCA analyzer...")
    analyzer = TETPhysioCCAAnalyzer(data)
    
    # Fit CCA
    print("\n3. Fitting CCA models...")
    analyzer.fit_cca(n_components=2)
    print("   CCA models fitted successfully")
    
    # Compute redundancy indices
    print("\n4. Computing redundancy indices...")
    for state in ['RS', 'DMT']:
        print(f"\n   {state} State:")
        redundancy_df = analyzer.compute_redundancy_index(state)
        print(redundancy_df.to_string(index=False))
        
        # Check that redundancy values are reasonable
        for _, row in redundancy_df.iterrows():
            if row['canonical_variate'] != 'Total':
                assert 0 <= row['redundancy_Y_given_X'] <= 1, \
                    f"Invalid redundancy Y|X: {row['redundancy_Y_given_X']}"
                assert 0 <= row['redundancy_X_given_Y'] <= 1, \
                    f"Invalid redundancy X|Y: {row['redundancy_X_given_Y']}"
    
    print("\n5. Testing redundancy visualization...")
    output_dir = Path('test/tet/temp_redundancy_test')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig_path = analyzer.plot_redundancy_indices(str(output_dir))
    print(f"   Saved figure to: {fig_path}")
    
    # Check that figure was created
    assert Path(fig_path).exists(), "Figure file not created"
    
    print("\n6. Testing export with redundancy indices...")
    file_paths = analyzer.export_results(str(output_dir))
    
    # Check that redundancy files were created
    assert 'cca_redundancy_indices' in file_paths, \
        "Redundancy indices not exported"
    assert Path(file_paths['cca_redundancy_indices']).exists(), \
        "Redundancy indices file not found"
    
    print(f"   Exported files:")
    for key, path in file_paths.items():
        if 'redundancy' in key:
            print(f"     - {key}: {path}")
    
    # Load and verify redundancy file
    redundancy_df = pd.read_csv(file_paths['cca_redundancy_indices'])
    print(f"\n   Redundancy indices file shape: {redundancy_df.shape}")
    print(f"   Columns: {list(redundancy_df.columns)}")
    
    # Check for interpretation column in interpreted file
    if 'cca_redundancy_indices_interpreted' in file_paths:
        interpreted_df = pd.read_csv(file_paths['cca_redundancy_indices_interpreted'])
        assert 'interpretation' in interpreted_df.columns, \
            "Interpretation column missing"
        print(f"\n   Interpretations:")
        for _, row in interpreted_df.iterrows():
            if row['canonical_variate'] != 'Total':
                print(f"     {row['state']} CV{int(row['canonical_variate'])}: "
                      f"{row['interpretation']}")
    
    print("\n" + "=" * 80)
    print("✓ All redundancy tests passed!")
    print("=" * 80)
    
    # Cleanup
    import shutil
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"\nCleaned up temporary directory: {output_dir}")


if __name__ == '__main__':
    test_redundancy_computation()
