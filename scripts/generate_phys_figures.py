# -*- coding: utf-8 -*-
"""
Generate All Physiological Figures.

This script runs the complete pipeline to generate all physiological analysis
figures, including unimodal (HR, SMNA, RVT) and composite arousal index analyses.

Run:
  python scripts/generate_phys_figures.py
"""

import os
import sys
import subprocess


def run_script(script_path: str, description: str) -> bool:
    """Run a Python script and return success status."""
    print("\n" + "=" * 80)
    print(f"Running: {description}")
    print("=" * 80 + "\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ {description} failed: {str(e)}")
        return False


def main():
    """Main execution."""
    print("\n" + "=" * 80)
    print("GENERATING ALL PHYSIOLOGICAL FIGURES")
    print("=" * 80)
    
    scripts = [
        ('pipelines/run_resp_rvt_analysis.py', 'Respiration (RVT) Analysis'),
        ('pipelines/run_ecg_hr_analysis.py', 'ECG (HR) Analysis'),
        ('pipelines/run_eda_smna_analysis.py', 'EDA (SMNA) Analysis'),
        ('scripts/save_extended_dmt_data.py', 'Extract Extended DMT Data'),
        ('pipelines/run_composite_arousal_index.py', 'Composite Arousal Index Analysis'),
    ]
    
    results = []
    for script_path, description in scripts:
        success = run_script(script_path, description)
        results.append((description, success))
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80 + "\n")
    
    for description, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {description}")
    
    all_success = all(success for _, success in results)
    
    if all_success:
        print("\n✓ All analyses completed successfully!")
        print("\nGenerated figures:")
        print("  - results/resp/rvt/plots/")
        print("  - results/ecg/hr/plots/")
        print("  - results/eda/smna/plots/")
        print("  - results/composite/plots/")
    else:
        print("\n⚠ Some analyses failed. Check logs above for details.")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
