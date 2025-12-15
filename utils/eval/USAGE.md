# Feature Extraction and Analysis - Quick Guide

## Extract Features (Generate CSV with 10 features)

```bash
cd /cis/home/qgao14/my_documents/VIOCR_infer_models
python untils/eval/extract_text_region_features.py
```

**Output**: `results/totaltext_16_text_region_features.csv`

**Features in CSV** (10 columns):
1. image_name
2. text
3. polygon_size
4. edge_density
5. luminance_std
6. luminance_mean
7. avg_curvature
8. gaussian_rms_contrast
9. **spatial_frequency** (NEW)
10. **perimetric_complexity** (NEW)

## Analyze Features

```bash
python untils/eval/analyze_features.py
```

**Output**: 
- Terminal: Statistical analysis of all features
- `results/totaltext_16_image_level_features.csv`: Per-image aggregated statistics

**What it shows**:
- Distribution statistics (mean, median, percentiles) for each feature
- Feature comparison by luminance groups (Dark/Normal/Bright)
- Correlation analysis between features
- Per-image statistics

**Note**: No difficulty classification is performed. Only statistical analysis.

## Verify CSV Format

Check if CSV has new features:

```bash
head -1 results/totaltext_16_text_region_features.csv
```

Should see:
```
image_name,text,polygon_size,edge_density,luminance_std,luminance_mean,avg_curvature,gaussian_rms_contrast,spatial_frequency,perimetric_complexity
```

## Quick Test

```bash
# Extract features
python untils/eval/extract_text_region_features.py

# Verify output
wc -l results/totaltext_16_text_region_features.csv

# Analyze
python untils/eval/analyze_features.py
```

Done!








