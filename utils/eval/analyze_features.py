"""
Analyze text region features from extracted CSV.

This script loads the feature CSV and provides statistical analysis:
- Distribution statistics for each feature
- Feature comparison by luminance groups
- Correlation analysis
- Per-image aggregated statistics
"""

import pandas as pd
import numpy as np
import os


def analyze_features(csv_path='results/totaltext_16_text_region_features.csv'):
    """Analyze text region features and generate statistics."""
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        print("Please run extract_text_region_features.py first.")
        return
    
    # Load data
    print(f"Loading features from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} text regions from {df['image_name'].nunique()} images")
    
    # Check which features are available
    has_spatial_freq = 'spatial_frequency' in df.columns
    has_perimetric_comp = 'perimetric_complexity' in df.columns
    
    if not has_spatial_freq or not has_perimetric_comp:
        print("\nWarning: New features not found in CSV.")
        print("Please re-run extract_text_region_features.py to generate updated features.")
        if not has_spatial_freq:
            print("  Missing: spatial_frequency")
        if not has_perimetric_comp:
            print("  Missing: perimetric_complexity")
        print()
    
    # === Feature Distribution Statistics ===
    print("\n" + "="*60)
    print("FEATURE DISTRIBUTION STATISTICS")
    print("="*60)
    
    print("\nPolygon Size (pixels):")
    print(f"  Mean: {df['polygon_size'].mean():.2f}")
    print(f"  Median: {df['polygon_size'].median():.2f}")
    print(f"  25th percentile: {df['polygon_size'].quantile(0.25):.2f}")
    print(f"  75th percentile: {df['polygon_size'].quantile(0.75):.2f}")
    
    print("\nEdge Density:")
    print(f"  Mean: {df['edge_density'].mean():.4f}")
    print(f"  Median: {df['edge_density'].median():.4f}")
    print(f"  25th percentile: {df['edge_density'].quantile(0.25):.4f}")
    print(f"  75th percentile: {df['edge_density'].quantile(0.75):.4f}")
    
    print("\nLuminance (Mean):")
    print(f"  Mean: {df['luminance_mean'].mean():.2f}")
    print(f"  Median: {df['luminance_mean'].median():.2f}")
    print(f"  25th percentile: {df['luminance_mean'].quantile(0.25):.2f}")
    print(f"  75th percentile: {df['luminance_mean'].quantile(0.75):.2f}")
    
    print("\nLuminance (Std Dev):")
    print(f"  Mean: {df['luminance_std'].mean():.2f}")
    print(f"  Median: {df['luminance_std'].median():.2f}")
    print(f"  25th percentile: {df['luminance_std'].quantile(0.25):.2f}")
    print(f"  75th percentile: {df['luminance_std'].quantile(0.75):.2f}")
    
    print("\nCurvature (degrees):")
    print(f"  Mean: {df['avg_curvature'].mean():.2f}")
    print(f"  Median: {df['avg_curvature'].median():.2f}")
    print(f"  25th percentile: {df['avg_curvature'].quantile(0.25):.2f}")
    print(f"  75th percentile: {df['avg_curvature'].quantile(0.75):.2f}")
    
    print("\nGaussian RMS Contrast:")
    print(f"  Mean: {df['gaussian_rms_contrast'].mean():.4f}")
    print(f"  Median: {df['gaussian_rms_contrast'].median():.4f}")
    print(f"  25th percentile: {df['gaussian_rms_contrast'].quantile(0.25):.4f}")
    print(f"  75th percentile: {df['gaussian_rms_contrast'].quantile(0.75):.4f}")
    
    if has_spatial_freq:
        print("\nSpatial Frequency:")
        print(f"  Mean: {df['spatial_frequency'].mean():.4f}")
        print(f"  Median: {df['spatial_frequency'].median():.4f}")
        print(f"  25th percentile: {df['spatial_frequency'].quantile(0.25):.4f}")
        print(f"  75th percentile: {df['spatial_frequency'].quantile(0.75):.4f}")
    
    if has_perimetric_comp:
        print("\nPerimetric Complexity:")
        print(f"  Mean: {df['perimetric_complexity'].mean():.2f}")
        print(f"  Median: {df['perimetric_complexity'].median():.2f}")
        print(f"  25th percentile: {df['perimetric_complexity'].quantile(0.25):.2f}")
        print(f"  75th percentile: {df['perimetric_complexity'].quantile(0.75):.2f}")
    
    # === Luminance-based grouping analysis ===
    print("\n" + "="*60)
    print("FEATURE COMPARISON BY LUMINANCE GROUPS")
    print("="*60)
    
    df['luminance_group'] = pd.cut(df['luminance_mean'], 
                                     bins=[0, 80, 180, 255], 
                                     labels=['Dark (<80)', 'Normal (80-180)', 'Bright (>180)'])
    
    for group in ['Dark (<80)', 'Normal (80-180)', 'Bright (>180)']:
        group_data = df[df['luminance_group'] == group]
        if len(group_data) == 0:
            continue
        print(f"\n{group}: {len(group_data)} regions")
        print(f"  Gaussian RMS Contrast: {group_data['gaussian_rms_contrast'].mean():.4f}")
        print(f"  Edge Density: {group_data['edge_density'].mean():.4f}")
        print(f"  Curvature: {group_data['avg_curvature'].mean():.2f}°")
        if has_spatial_freq:
            print(f"  Spatial Frequency: {group_data['spatial_frequency'].mean():.4f}")
        if has_perimetric_comp:
            print(f"  Perimetric Complexity: {group_data['perimetric_complexity'].mean():.2f}")
    
    # === Correlation Analysis ===
    print("\n" + "="*60)
    print("FEATURE CORRELATION ANALYSIS")
    print("="*60)
    
    # Select numeric columns for correlation
    numeric_cols = ['polygon_size', 'edge_density', 'luminance_std', 'luminance_mean', 
                    'avg_curvature', 'gaussian_rms_contrast']
    if has_spatial_freq:
        numeric_cols.append('spatial_frequency')
    if has_perimetric_comp:
        numeric_cols.append('perimetric_complexity')
    
    corr_matrix = df[numeric_cols].corr()
    
    print("\nTop correlations (|r| > 0.3):")
    # Get upper triangle of correlation matrix
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.3:
                print(f"  {numeric_cols[i]} <-> {numeric_cols[j]}: {corr_val:.3f}")
    
    # === Per-Image Statistics ===
    print("\n" + "="*60)
    print("PER-IMAGE STATISTICS")
    print("="*60)
    
    image_stats = df.groupby('image_name').agg({
        'text': 'count',
        'polygon_size': 'mean',
        'edge_density': 'mean',
        'luminance_std': 'mean',
        'luminance_mean': 'mean',
        'avg_curvature': 'mean',
        'gaussian_rms_contrast': 'mean'
    }).rename(columns={'text': 'num_text_regions'})
    
    if has_spatial_freq:
        image_stats['spatial_frequency'] = df.groupby('image_name')['spatial_frequency'].mean()
    if has_perimetric_comp:
        image_stats['perimetric_complexity'] = df.groupby('image_name')['perimetric_complexity'].mean()
    
    # Round values
    for col in image_stats.columns:
        if col != 'num_text_regions':
            if col in ['edge_density', 'gaussian_rms_contrast', 'spatial_frequency']:
                image_stats[col] = image_stats[col].round(4)
            else:
                image_stats[col] = image_stats[col].round(2)
    
    print(f"\nGenerated statistics for {len(image_stats)} images")
    print("\nSample (first 5 images):")
    print(image_stats.head())
    
    # Save per-image statistics
    output_path = 'results/totaltext_16_image_level_features.csv'
    image_stats.to_csv(output_path)
    print(f"\nSaved per-image statistics to: {output_path}")
    
    # === Summary ===
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total text regions analyzed: {len(df)}")
    print(f"Total images: {df['image_name'].nunique()}")
    print(f"Average text regions per image: {len(df) / df['image_name'].nunique():.1f}")
    print(f"\nFeatures analyzed: {len(numeric_cols)}")
    for col in numeric_cols:
        print(f"  - {col}")
    
    print("\nOutput files:")
    print(f"  - {csv_path} (input)")
    print(f"  - {output_path} (per-image stats)")
    print("\n" + "="*60)


if __name__ == "__main__":
    analyze_features()






