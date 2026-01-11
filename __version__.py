"""
Version information for DMT-Emotions Analysis Pipeline.

This file contains version information for the entire project and its components.
"""

# Main project version
__version__ = "1.1.0"

# Component versions
ANALYSIS_MODULES_VERSION = "1.0.0"  # New analysis utility modules
PREPROCESSING_VERSION = "1.0.0"     # Physiological preprocessing
TET_ANALYSIS_VERSION = "1.0.0"      # TET analysis pipeline
COUPLING_ANALYSIS_VERSION = "1.0.0" # Physiology-experience coupling

# Version history
VERSION_HISTORY = {
    "1.1.0": {
        "date": "2026-01-11",
        "description": "Supervisor-requested analysis revisions",
        "changes": [
            "Alternative statistical approaches (one-tailed, uncorrected tests)",
            "Temporal phase-based analysis",
            "Feature extraction from physiological time series",
            "Baseline comparisons (DMT vs RS)",
            "Enhanced visualizations with significance markers",
            "APA-style statistical reporting",
            "Homogeneous figure aesthetics",
            "Version tracking with timestamps"
        ]
    },
    "1.0.0": {
        "date": "2025-XX-XX",
        "description": "Initial release",
        "changes": [
            "Complete DMT physiological analysis pipeline",
            "ECG, EDA, Respiration analyses",
            "Composite Arousal Index (PC1)",
            "TET analysis",
            "Physiology-Experience coupling (CCA)",
            "Publication figure generation"
        ]
    }
}


def get_version_info() -> str:
    """
    Get formatted version information string.
    
    Returns:
        Formatted string with version and component information
    """
    info = [
        f"DMT-Emotions Analysis Pipeline v{__version__}",
        "",
        "Component Versions:",
        f"  - Analysis Modules: v{ANALYSIS_MODULES_VERSION}",
        f"  - Preprocessing: v{PREPROCESSING_VERSION}",
        f"  - TET Analysis: v{TET_ANALYSIS_VERSION}",
        f"  - Coupling Analysis: v{COUPLING_ANALYSIS_VERSION}",
    ]
    return "\n".join(info)


if __name__ == "__main__":
    print(get_version_info())
