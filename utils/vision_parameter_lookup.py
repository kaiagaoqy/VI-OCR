"""
Vision Parameter Lookup Tool
Retrieves vision parameters (a, b) for subjects based on their measured VA and CS.
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional, Dict


class VisionParameterLookup:
    """
    Class to lookup vision parameters for subjects.
    """
    
    def __init__(self, 
                 param_matrix_path: str = 'data/param.matrix.csv',
                 human_vision_path: str = 'data/human/human_measured_vision_cleaned.csv'):
        """
        Initialize the lookup tool.
        
        Args:
            param_matrix_path: Path to param.matrix.csv
            human_vision_path: Path to human_measured_vision_cleaned.csv
        """
        self.param_matrix_path = param_matrix_path
        self.human_vision_path = human_vision_path
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load the CSV files."""
        if not os.path.exists(self.param_matrix_path):
            raise FileNotFoundError(f"Parameter matrix file not found: {self.param_matrix_path}")
        
        if not os.path.exists(self.human_vision_path):
            raise FileNotFoundError(f"Human vision file not found: {self.human_vision_path}")
        
        # Load parameter matrix
        self.param_df = pd.read_csv(self.param_matrix_path)
        
        # Convert columns to numeric, coercing errors to NaN
        for col in ['VA', 'CS', 'a', 'b']:
            if col in self.param_df.columns:
                self.param_df[col] = pd.to_numeric(self.param_df[col], errors='coerce')
        
        # Remove rows with NaN values in critical columns
        self.param_df = self.param_df.dropna(subset=['VA', 'CS', 'a', 'b'])
        
        print(f"Loaded parameter matrix: {len(self.param_df)} rows")
        
        # Load human vision data
        self.human_df = pd.read_csv(self.human_vision_path)
        print(f"Loaded human vision data: {len(self.human_df)} subjects")
    
    def get_subject_va_cs(self, subject_id: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Get VA and CS for a subject.
        
        Priority:
        - VA: VA_OU_logMAR if available, otherwise max(VA_OD_logMAR, VA_OS_logMAR)
        - CS: CS_OU if available, otherwise max(CS_OD, CS_OS)
        
        Args:
            subject_id: Subject ID (e.g., 'Sub123')
            
        Returns:
            Tuple of (VA, CS), where each can be None if not available
        """
        # Find subject
        subject_row = self.human_df[self.human_df['SubID'] == subject_id]
        
        if subject_row.empty:
            raise ValueError(f"Subject {subject_id} not found in human vision data")
        
        subject_row = subject_row.iloc[0]
        
        # Get VA
        va = None
        if pd.notna(subject_row.get('VA_OU_logMAR')):
            va = float(subject_row['VA_OU_logMAR'])
        else:
            # Use max of left and right eye
            va_od = subject_row.get('VA_OD_logMAR')
            va_os = subject_row.get('VA_OS_logMAR')
            
            valid_vas = []
            if pd.notna(va_od):
                valid_vas.append(float(va_od))
            if pd.notna(va_os):
                valid_vas.append(float(va_os))
            
            if valid_vas:
                va = max(valid_vas)  # Use worse eye (higher logMAR = worse)
        
        # Get CS
        cs = None
        if pd.notna(subject_row.get('CS_OU')):
            cs = float(subject_row['CS_OU'])
        else:
            # Use max of left and right eye
            cs_od = subject_row.get('CS_OD')
            cs_os = subject_row.get('CS_OS')
            
            valid_css = []
            if pd.notna(cs_od):
                valid_css.append(float(cs_od))
            if pd.notna(cs_os):
                valid_css.append(float(cs_os))
            
            if valid_css:
                cs = max(valid_css)  # Use better eye (higher CS = better)
        
        return va, cs
    
    def find_closest_params(self, va: float, cs: float) -> Tuple[float, float, float, float]:
        """
        Find the closest (a, b) parameters for given VA and CS.
        
        Finds the row in param matrix where VA and CS are closest to the given values.
        
        Args:
            va: Visual acuity (logMAR)
            cs: Contrast sensitivity (logCS)
            
        Returns:
            Tuple of (va_matched, cs_matched, a, b)
        """
        if self.param_df.empty:
            raise ValueError("Parameter matrix is empty")
        
        # Work with a copy to avoid modifying the original dataframe
        df_copy = self.param_df.copy()
        
        # Find closest VA
        df_copy['va_diff'] = np.abs(df_copy['VA'] - va)
        
        # Find closest CS
        df_copy['cs_diff'] = np.abs(df_copy['CS'] - cs)
        
        # Find row with minimum combined distance
        # We want to find the closest match for both VA and CS
        # Use Euclidean distance in the normalized space
        va_range = df_copy['VA'].max() - df_copy['VA'].min()
        cs_range = df_copy['CS'].max() - df_copy['CS'].min()
        
        # Avoid division by zero
        if va_range == 0:
            va_range = 1
        if cs_range == 0:
            cs_range = 1
        
        df_copy['distance'] = np.sqrt(
            (df_copy['va_diff'] / va_range) ** 2 + 
            (df_copy['cs_diff'] / cs_range) ** 2
        )
        
        # Find closest match
        closest_idx = df_copy['distance'].idxmin()
        closest_row = df_copy.loc[closest_idx]
        
        va_matched = float(closest_row['VA'])
        cs_matched = float(closest_row['CS'])
        a = float(closest_row['a'])
        b = float(closest_row['b'])
        
        return va_matched, cs_matched, a, b
    
    def get_filter_params_for_subject(self, subject_id: str, 
                                      verbose: bool = True) -> Dict[str, float]:
        """
        Get complete filter parameters for a subject.
        
        Args:
            subject_id: Subject ID (e.g., 'Sub123')
            verbose: Print detailed information
            
        Returns:
            Dictionary with keys:
                - 'subject_id': Subject ID
                - 'va_measured': Measured VA
                - 'cs_measured': Measured CS
                - 'va_matched': Matched VA from parameter matrix
                - 'cs_matched': Matched CS from parameter matrix
                - 'a': Parameter a (raw, before 1/a transform)
                - 'b': Parameter b (vshift)
                - 'hshift': Computed hshift = 1/a
                - 'vshift': Computed vshift = b
        """
        # Get subject's VA and CS
        va, cs = self.get_subject_va_cs(subject_id)
        
        if va is None or cs is None:
            raise ValueError(
                f"Cannot compute parameters for {subject_id}: "
                f"VA={va}, CS={cs} (at least one is None)"
            )
        
        # Find closest parameters
        va_matched, cs_matched, a, b = self.find_closest_params(va, cs)
        
        # Compute hshift and vshift
        hshift = np.round(1 / a, 4) if a != 0 else 0
        vshift = b
        
        result = {
            'subject_id': subject_id,
            'va_measured': va,
            'cs_measured': cs,
            'va_matched': va_matched,
            'cs_matched': cs_matched,
            'a': a,
            'b': b,
            'hshift': hshift,
            'vshift': vshift
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Subject: {subject_id}")
            print(f"{'='*60}")
            print(f"Measured  - VA: {va:.4f} logMAR, CS: {cs:.4f} logCS")
            print(f"Matched   - VA: {va_matched:.4f} logMAR, CS: {cs_matched:.4f} logCS")
            print(f"Parameters - a: {a:.4f}, b: {b:.4f}")
            print(f"Filter    - hshift: {hshift:.4f}, vshift: {vshift:.4f}")
            print(f"{'='*60}\n")
        
        return result
    
    def get_all_subjects(self) -> list:
        """
        Get list of all subject IDs.
        
        Returns:
            List of subject IDs
        """
        return self.human_df['SubID'].tolist()
    
    def export_subject_parameters(self, output_path: str):
        """
        Export parameters for all subjects to a CSV file.
        
        Args:
            output_path: Path to output CSV file
        """
        all_results = []
        
        for subject_id in self.get_all_subjects():
            try:
                result = self.get_filter_params_for_subject(subject_id, verbose=False)
                all_results.append(result)
            except Exception as e:
                print(f"Warning: Could not process {subject_id}: {e}")
        
        # Convert to DataFrame and save
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_path, index=False)
        print(f"Exported parameters for {len(all_results)} subjects to {output_path}")


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Lookup vision parameters for subjects'
    )
    parser.add_argument(
        '--subject', 
        type=str,
        help='Subject ID to lookup (e.g., Sub123)'
    )
    parser.add_argument(
        '--export',
        type=str,
        help='Export all subjects to CSV file'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available subjects'
    )
    
    args = parser.parse_args()
    
    # Initialize lookup tool
    lookup = VisionParameterLookup()
    
    if args.list:
        print("\nAvailable subjects:")
        print("=" * 60)
        subjects = lookup.get_all_subjects()
        for i, subject_id in enumerate(subjects, 1):
            print(f"{i:3d}. {subject_id}")
        print(f"\nTotal: {len(subjects)} subjects")
        
    elif args.subject:
        # Lookup specific subject
        result = lookup.get_filter_params_for_subject(args.subject, verbose=True)
        
        print("Python usage:")
        print(f"  hshift = {result['hshift']}")
        print(f"  vshift = {result['vshift']}")
        
    elif args.export:
        # Export all subjects
        lookup.export_subject_parameters(args.export)
        print(f"\nParameters exported to: {args.export}")
        
    else:
        # Interactive mode
        print("\nVision Parameter Lookup Tool")
        print("=" * 60)
        print("Available commands:")
        print("  1. Lookup specific subject")
        print("  2. List all subjects")
        print("  3. Export all to CSV")
        print("  4. Exit")
        print("=" * 60)
        
        while True:
            choice = input("\nEnter choice (1-4): ").strip()
            
            if choice == '1':
                subject_id = input("Enter subject ID (e.g., Sub123): ").strip()
                try:
                    result = lookup.get_filter_params_for_subject(subject_id, verbose=True)
                except Exception as e:
                    print(f"Error: {e}")
                    
            elif choice == '2':
                subjects = lookup.get_all_subjects()
                print(f"\nFound {len(subjects)} subjects:")
                for i, subj in enumerate(subjects[:20], 1):  # Show first 20
                    print(f"  {i}. {subj}")
                if len(subjects) > 20:
                    print(f"  ... and {len(subjects) - 20} more")
                    
            elif choice == '3':
                output_path = input("Enter output CSV path: ").strip()
                lookup.export_subject_parameters(output_path)
                
            elif choice == '4':
                print("Goodbye!")
                break
                
            else:
                print("Invalid choice. Please enter 1-4.")


if __name__ == '__main__':
    main()

