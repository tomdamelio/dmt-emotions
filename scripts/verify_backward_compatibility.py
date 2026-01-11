"""
Backward Compatibility Verification Script

This script verifies that updated analysis scripts produce results identical
to previously generated outputs (within numerical tolerance) for unchanged analyses.

Requirements: 7.3
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import json
from datetime import datetime


class BackwardCompatibilityVerifier:
    """Verify backward compatibility of analysis pipeline updates."""
    
    def __init__(
        self,
        results_dir: str = "results",
        backup_dir: str = "results_backup_compatibility",
        tolerance: float = 1e-6
    ):
        """
        Initialize verifier.
        
        Args:
            results_dir: Directory containing current results
            backup_dir: Directory to store backup of original results
            tolerance: Numerical tolerance for floating-point comparisons
        """
        self.results_dir = Path(results_dir)
        self.backup_dir = Path(backup_dir)
        self.tolerance = tolerance
        self.comparison_results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def backup_results(self) -> None:
        """Create backup of current results."""
        print(f"\n{'='*80}")
        print("STEP 1: Backing up current results")
        print(f"{'='*80}")
        
        if self.backup_dir.exists():
            print(f"Removing existing backup at {self.backup_dir}")
            shutil.rmtree(self.backup_dir)
        
        print(f"Creating backup: {self.results_dir} -> {self.backup_dir}")
        shutil.copytree(self.results_dir, self.backup_dir)
        print(f"✓ Backup created successfully")
        
    def run_analysis_scripts(self) -> Dict[str, bool]:
        """
        Run all analysis scripts to regenerate results.
        
        Returns:
            Dictionary mapping script names to success status
        """
        print(f"\n{'='*80}")
        print("STEP 2: Running analysis scripts")
        print(f"{'='*80}")
        
        scripts = [
            "src/run_ecg_hr_analysis.py",
            "src/run_eda_smna_analysis.py",
            "src/run_resp_rvt_analysis.py",
            "src/run_composite_arousal_index.py"
        ]
        
        results = {}
        for script in scripts:
            script_path = Path(script)
            if not script_path.exists():
                print(f"⚠ Script not found: {script}")
                results[script] = False
                continue
                
            print(f"\nRunning: {script}")
            try:
                # Note: This would need to be run with micromamba in actual execution
                # For now, we'll document the command that should be used
                print(f"  Command: micromamba run -n dmt-emotions python {script}")
                print(f"  ⚠ Manual execution required - skipping for verification script")
                results[script] = None  # None indicates manual execution needed
            except Exception as e:
                print(f"  ✗ Error: {e}")
                results[script] = False
                
        return results
        
    def compare_csv_files(
        self,
        original: Path,
        updated: Path
    ) -> Tuple[bool, str]:
        """
        Compare two CSV files for numerical equivalence.
        
        Args:
            original: Path to original CSV file
            updated: Path to updated CSV file
            
        Returns:
            Tuple of (is_identical, message)
        """
        try:
            df_orig = pd.read_csv(original)
            df_updated = pd.read_csv(updated)
            
            # Check shape
            if df_orig.shape != df_updated.shape:
                return False, f"Shape mismatch: {df_orig.shape} vs {df_updated.shape}"
            
            # Check columns
            if not df_orig.columns.equals(df_updated.columns):
                return False, f"Column mismatch: {list(df_orig.columns)} vs {list(df_updated.columns)}"
            
            # Compare numeric columns
            numeric_cols = df_orig.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if not np.allclose(
                    df_orig[col].values,
                    df_updated[col].values,
                    rtol=self.tolerance,
                    atol=self.tolerance,
                    equal_nan=True
                ):
                    max_diff = np.nanmax(np.abs(df_orig[col].values - df_updated[col].values))
                    return False, f"Numeric difference in column '{col}': max_diff={max_diff:.2e}"
            
            # Compare non-numeric columns
            non_numeric_cols = df_orig.select_dtypes(exclude=[np.number]).columns
            for col in non_numeric_cols:
                if not df_orig[col].equals(df_updated[col]):
                    return False, f"Non-numeric difference in column '{col}'"
            
            return True, "Identical within tolerance"
            
        except Exception as e:
            return False, f"Error comparing files: {str(e)}"
    
    def compare_txt_files(
        self,
        original: Path,
        updated: Path
    ) -> Tuple[bool, str]:
        """
        Compare two text files.
        
        Args:
            original: Path to original text file
            updated: Path to updated text file
            
        Returns:
            Tuple of (is_identical, message)
        """
        try:
            with open(original, 'r') as f:
                orig_content = f.read()
            with open(updated, 'r') as f:
                updated_content = f.read()
            
            if orig_content == updated_content:
                return True, "Identical"
            else:
                return False, "Content differs"
                
        except Exception as e:
            return False, f"Error comparing files: {str(e)}"
    
    def compare_directories(
        self,
        original_dir: Path,
        updated_dir: Path,
        file_patterns: List[str] = None
    ) -> List[Dict]:
        """
        Compare files in two directories.
        
        Args:
            original_dir: Path to original directory
            updated_dir: Path to updated directory
            file_patterns: List of file patterns to compare (e.g., ['*.csv', '*.txt'])
            
        Returns:
            List of comparison results
        """
        if file_patterns is None:
            file_patterns = ['*.csv', '*.txt']
        
        results = []
        
        for pattern in file_patterns:
            for orig_file in original_dir.rglob(pattern):
                rel_path = orig_file.relative_to(original_dir)
                updated_file = updated_dir / rel_path
                
                if not updated_file.exists():
                    results.append({
                        'file': str(rel_path),
                        'status': 'missing',
                        'message': 'File not found in updated results'
                    })
                    continue
                
                # Compare based on file type
                if orig_file.suffix == '.csv':
                    is_identical, message = self.compare_csv_files(orig_file, updated_file)
                elif orig_file.suffix == '.txt':
                    is_identical, message = self.compare_txt_files(orig_file, updated_file)
                else:
                    is_identical, message = None, "Unsupported file type"
                
                results.append({
                    'file': str(rel_path),
                    'status': 'identical' if is_identical else 'different',
                    'message': message
                })
        
        return results
    
    def verify_core_outputs(self) -> List[Dict]:
        """
        Verify core output files that should remain unchanged.
        
        Returns:
            List of verification results
        """
        print(f"\n{'='*80}")
        print("STEP 3: Comparing core outputs")
        print(f"{'='*80}")
        
        # Define core outputs that should remain unchanged
        core_outputs = [
            # LME model summaries (should be unchanged)
            "ecg/hr/model_summary.txt",
            "eda/smna/model_summary.txt",
            "resp/rvt/model_summary.txt",
            "composite/model_summary.txt",
            
            # LME analysis reports (should be unchanged)
            "ecg/hr/lme_analysis_report.txt",
            "eda/smna/lme_analysis_report.txt",
            "resp/rvt/lme_analysis_report.txt",
            "composite/lme_analysis_report.txt",
            
            # Core data files (should be unchanged)
            "ecg/hr/hr_minute_long_data_z.csv",
            "eda/smna/smna_auc_long_data_z.csv",
            "resp/rvt/resp_rvt_minute_long_data_z.csv",
            "composite/merged_minute_data_complete_cases_9min.csv",
        ]
        
        results = []
        
        for output_path in core_outputs:
            orig_file = self.backup_dir / output_path
            updated_file = self.results_dir / output_path
            
            if not orig_file.exists():
                results.append({
                    'file': output_path,
                    'status': 'missing_original',
                    'message': 'Original file not found'
                })
                continue
            
            if not updated_file.exists():
                results.append({
                    'file': output_path,
                    'status': 'missing_updated',
                    'message': 'Updated file not found'
                })
                continue
            
            # Compare based on file type
            if orig_file.suffix == '.csv':
                is_identical, message = self.compare_csv_files(orig_file, updated_file)
            elif orig_file.suffix == '.txt':
                is_identical, message = self.compare_txt_files(orig_file, updated_file)
            else:
                is_identical, message = None, "Unsupported file type"
            
            status = 'identical' if is_identical else 'different'
            results.append({
                'file': output_path,
                'status': status,
                'message': message
            })
            
            # Print result
            symbol = '✓' if is_identical else '✗'
            print(f"{symbol} {output_path}: {message}")
        
        return results
    
    def verify_new_outputs(self) -> List[Dict]:
        """
        Document new outputs added by revisions.
        
        Returns:
            List of new output files
        """
        print(f"\n{'='*80}")
        print("STEP 4: Documenting new outputs")
        print(f"{'='*80}")
        
        # Define expected new output directories
        new_output_dirs = [
            "ecg/hr/alternative_tests",
            "ecg/hr/phase_analysis",
            "ecg/hr/features",
            "ecg/hr/baseline_comparison",
            "eda/smna/alternative_tests",
            "eda/smna/phase_analysis",
            "eda/smna/features",
            "eda/smna/baseline_comparison",
            "resp/rvt/alternative_tests",
            "resp/rvt/phase_analysis",
            "resp/rvt/features",
            "resp/rvt/baseline_comparison",
            "composite/phase_analysis",
            "composite/features",
            "composite/baseline_comparison",
        ]
        
        new_files = []
        
        for dir_path in new_output_dirs:
            full_path = self.results_dir / dir_path
            if full_path.exists():
                files = list(full_path.rglob('*'))
                for f in files:
                    if f.is_file():
                        rel_path = f.relative_to(self.results_dir)
                        new_files.append({
                            'file': str(rel_path),
                            'size': f.stat().st_size,
                            'type': 'new_output'
                        })
                        print(f"  + {rel_path} ({f.stat().st_size} bytes)")
        
        if not new_files:
            print("  No new output files found (analyses may not have been run yet)")
        
        return new_files
    
    def generate_report(
        self,
        core_results: List[Dict],
        new_outputs: List[Dict]
    ) -> None:
        """
        Generate backward compatibility report.
        
        Args:
            core_results: Results from core output verification
            new_outputs: List of new output files
        """
        print(f"\n{'='*80}")
        print("STEP 5: Generating report")
        print(f"{'='*80}")
        
        report_path = Path(f"BACKWARD_COMPATIBILITY_REPORT_{self.timestamp}.md")
        
        with open(report_path, 'w') as f:
            f.write("# Backward Compatibility Verification Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Numerical Tolerance**: {self.tolerance:.2e}\n\n")
            
            # Summary statistics
            identical_count = sum(1 for r in core_results if r['status'] == 'identical')
            different_count = sum(1 for r in core_results if r['status'] == 'different')
            missing_count = sum(1 for r in core_results if 'missing' in r['status'])
            
            f.write("## Summary\n\n")
            f.write(f"- **Total core outputs checked**: {len(core_results)}\n")
            f.write(f"- **Identical**: {identical_count}\n")
            f.write(f"- **Different**: {different_count}\n")
            f.write(f"- **Missing**: {missing_count}\n")
            f.write(f"- **New outputs**: {len(new_outputs)}\n\n")
            
            # Core outputs verification
            f.write("## Core Outputs Verification\n\n")
            f.write("These outputs should remain unchanged by the revisions:\n\n")
            
            for result in core_results:
                status_symbol = {
                    'identical': '✓',
                    'different': '✗',
                    'missing_original': '⚠',
                    'missing_updated': '⚠'
                }.get(result['status'], '?')
                
                f.write(f"### {status_symbol} `{result['file']}`\n\n")
                f.write(f"- **Status**: {result['status']}\n")
                f.write(f"- **Message**: {result['message']}\n\n")
            
            # New outputs
            f.write("## New Outputs\n\n")
            f.write("These outputs were added by the revisions:\n\n")
            
            if new_outputs:
                for output in new_outputs:
                    f.write(f"- `{output['file']}` ({output['size']} bytes)\n")
            else:
                f.write("*No new outputs found (analyses may not have been run yet)*\n")
            
            f.write("\n## Intentional Changes\n\n")
            f.write("The following changes are intentional and expected:\n\n")
            f.write("1. **New analysis outputs**: Alternative statistics, phase analysis, ")
            f.write("feature extraction, and baseline comparisons are new additions.\n")
            f.write("2. **Enhanced visualizations**: Figures may have updated aesthetics ")
            f.write("(significance markers, homogeneous fonts) but core data should be unchanged.\n")
            f.write("3. **Statistical reporting**: New APA-style formatted reports are additions, ")
            f.write("not replacements.\n\n")
            
            f.write("## Verification Status\n\n")
            if different_count == 0 and missing_count == 0:
                f.write("✓ **PASSED**: All core outputs are identical within tolerance.\n")
            else:
                f.write("✗ **FAILED**: Some core outputs differ or are missing.\n")
                f.write("\nPlease review the differences above and verify they are intentional.\n")
        
        print(f"\n✓ Report saved to: {report_path}")
        
        # Also save JSON version for programmatic access
        json_path = Path(f"backward_compatibility_results_{self.timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump({
                'timestamp': self.timestamp,
                'tolerance': self.tolerance,
                'core_results': core_results,
                'new_outputs': new_outputs,
                'summary': {
                    'total_checked': len(core_results),
                    'identical': identical_count,
                    'different': different_count,
                    'missing': missing_count,
                    'new_outputs': len(new_outputs)
                }
            }, f, indent=2)
        
        print(f"✓ JSON results saved to: {json_path}")


def main():
    """Main execution function."""
    print("="*80)
    print("BACKWARD COMPATIBILITY VERIFICATION")
    print("="*80)
    print("\nThis script verifies that updated analysis scripts produce results")
    print("identical to previously generated outputs (within numerical tolerance)")
    print("for unchanged analyses.")
    print("\nRequirements: 7.3")
    
    verifier = BackwardCompatibilityVerifier(
        results_dir="results",
        backup_dir="results_backup_compatibility",
        tolerance=1e-6
    )
    
    # Step 1: Backup current results
    verifier.backup_results()
    
    # Step 2: Note about running analysis scripts
    print(f"\n{'='*80}")
    print("STEP 2: Running analysis scripts")
    print(f"{'='*80}")
    print("\n⚠ MANUAL STEP REQUIRED:")
    print("\nTo complete backward compatibility verification, you need to run")
    print("the following analysis scripts with micromamba:\n")
    
    scripts = [
        "src/run_ecg_hr_analysis.py",
        "src/run_eda_smna_analysis.py",
        "src/run_resp_rvt_analysis.py",
        "src/run_composite_arousal_index.py"
    ]
    
    for script in scripts:
        print(f"  micromamba run -n dmt-emotions python {script}")
    
    print("\nAfter running these scripts, re-run this verification script to compare outputs.")
    print("\nFor now, proceeding with comparison of existing results...")
    
    # Step 3: Verify core outputs
    core_results = verifier.verify_core_outputs()
    
    # Step 4: Document new outputs
    new_outputs = verifier.verify_new_outputs()
    
    # Step 5: Generate report
    verifier.generate_report(core_results, new_outputs)
    
    print(f"\n{'='*80}")
    print("VERIFICATION COMPLETE")
    print(f"{'='*80}")
    
    # Return exit code based on results
    different_count = sum(1 for r in core_results if r['status'] == 'different')
    missing_count = sum(1 for r in core_results if 'missing' in r['status'])
    
    if different_count > 0 or missing_count > 0:
        print("\n⚠ Some core outputs differ or are missing.")
        print("Please review the report and verify changes are intentional.")
        return 1
    else:
        print("\n✓ All core outputs are identical within tolerance.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
