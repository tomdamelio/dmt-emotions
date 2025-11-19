"""
Basic test for TETPhysioVisualizer class.

This test verifies that the visualizer can be instantiated and that
the basic structure is correct.
"""

import sys
from pathlib import Path

# Add scripts/tet to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts' / 'tet'))

from physio_visualizer import TETPhysioVisualizer


def test_visualizer_instantiation():
    """Test that TETPhysioVisualizer can be instantiated."""
    visualizer = TETPhysioVisualizer()
    
    assert visualizer is not None
    assert hasattr(visualizer, 'figure_paths')
    assert hasattr(visualizer, 'logger')
    assert len(visualizer.figure_paths) == 0
    
    print("✓ TETPhysioVisualizer instantiation test passed")


def test_visualizer_methods():
    """Test that all required methods exist."""
    visualizer = TETPhysioVisualizer()
    
    assert hasattr(visualizer, 'plot_correlation_heatmaps')
    assert hasattr(visualizer, 'plot_regression_scatter')
    assert hasattr(visualizer, 'plot_cca_loadings')
    assert hasattr(visualizer, 'export_figures')
    
    assert callable(visualizer.plot_correlation_heatmaps)
    assert callable(visualizer.plot_regression_scatter)
    assert callable(visualizer.plot_cca_loadings)
    assert callable(visualizer.export_figures)
    
    print("✓ TETPhysioVisualizer methods test passed")


if __name__ == '__main__':
    print("Running basic TETPhysioVisualizer tests...")
    test_visualizer_instantiation()
    test_visualizer_methods()
    print("\nAll basic tests passed! ✓")
