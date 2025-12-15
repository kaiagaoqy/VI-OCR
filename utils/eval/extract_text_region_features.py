#!/usr/bin/env python3
"""
Extract visual features from text regions in TotalText dataset
Specifically for images in the 16/ directory

Features extracted:
- Polygon size (area in pixels)
- Edge density (ratio of edge pixels in the region)
- Average local contrast (standard deviation of pixel intensities)
- Average luminance (mean brightness)
- Average curvature (polygon shape complexity)
- Gaussian RMS contrast (local contrast energy)
"""
import json
import numpy as np
import cv2
from pathlib import Path
from shapely.geometry import Polygon
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from skimage import feature

def load_image(image_path):
    """Load image in grayscale and color"""
    img_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    img_color = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    return img_gray, img_color

def compute_polygon_curvature(polygon_coords):
    """
    Compute average curvature of a polygon.
    Curvature is estimated by the angle change between consecutive edges.
    
    Returns:
        Average absolute curvature (in degrees)
    """
    if len(polygon_coords) < 3:
        return 0.0
    
    points = np.array(polygon_coords)
    n = len(points)
    
    # Compute angles at each vertex
    curvatures = []
    for i in range(n):
        p1 = points[(i - 1) % n]
        p2 = points[i]
        p3 = points[(i + 1) % n]
        
        # Vectors
        v1 = p2 - p1
        v2 = p3 - p2
        
        # Compute angle between vectors
        len_v1 = np.linalg.norm(v1)
        len_v2 = np.linalg.norm(v2)
        
        if len_v1 > 0 and len_v2 > 0:
            # Dot product
            cos_angle = np.dot(v1, v2) / (len_v1 * len_v2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            
            # Angle in degrees
            angle = np.arccos(cos_angle) * 180.0 / np.pi
            
            # Curvature is deviation from straight line (180 degrees)
            curvature = abs(180.0 - angle)
            curvatures.append(curvature)
    
    if len(curvatures) == 0:
        return 0.0
    
    return np.mean(curvatures)

def compute_local_contrast_energy(image, mask, sigma=19.4):
    """
    Compute local contrast energy using Gaussian-weighted RMS contrast.
    Based on Zuiderbaan et al., 2017.
    
    Formula:
        C(x,y) = sqrt[ sum_i w(i) * (I(i) - μ(x,y))^2 / sum_i w(i) ]
        
    where w(i) is Gaussian weight and μ(x,y) is local mean.
    
    Args:
        image: Grayscale image (0-255)
        mask: Binary mask for the region
        sigma: Standard deviation for Gaussian window (in pixels)
        
    Returns:
        Mean contrast energy over the masked region
    """
    # Normalize to [0, 1]
    gray = image.astype(np.float64) / 255.0
    
    # Compute local mean with Gaussian filter
    local_mean = gaussian_filter(gray, sigma=sigma)
    
    # Compute squared differences
    squared_diff = (gray - local_mean) ** 2
    
    # Apply Gaussian filter to squared differences
    local_var = gaussian_filter(squared_diff, sigma=sigma)
    
    # RMS contrast (root of local variance)
    contrast_map = np.sqrt(local_var)
    
    # Apply mask and compute mean
    masked_contrast = contrast_map[mask > 0]
    
    if len(masked_contrast) == 0:
        return 0.0
    
    mean_contrast = masked_contrast.mean()
    return float(mean_contrast)

def compute_spatial_frequency(image, mask):
    """
    Compute spatial frequency of the masked region.
    Spatial frequency measures the overall activity level in an image.
    
    Formula:
        RF = sqrt(mean((I[i,j] - I[i,j-1])^2))  # Row frequency
        CF = sqrt(mean((I[i,j] - I[i-1,j])^2))  # Column frequency
        SF = sqrt(RF^2 + CF^2)                   # Spatial frequency
    
    Args:
        image: Grayscale image (0-255)
        mask: Binary mask for the region
        
    Returns:
        Spatial frequency value
    """
    # Normalize to [0, 1]
    gray = image.astype(np.float64) / 255.0
    
    # Extract region
    region = gray[mask > 0]
    
    if len(region) == 0:
        return 0.0
    
    # Get bounding box of the mask
    rows, cols = np.where(mask > 0)
    if len(rows) == 0:
        return 0.0
    
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()
    
    # Extract the region of interest
    roi = gray[min_row:max_row+1, min_col:max_col+1]
    roi_mask = mask[min_row:max_row+1, min_col:max_col+1]
    
    if roi.shape[0] < 2 or roi.shape[1] < 2:
        return 0.0
    
    # Row frequency (horizontal changes)
    row_diff = np.diff(roi, axis=1)  # I[i,j] - I[i,j-1]
    row_mask = roi_mask[:, 1:] & roi_mask[:, :-1]
    if np.sum(row_mask) > 0:
        rf = np.sqrt(np.mean((row_diff[row_mask > 0]) ** 2))
    else:
        rf = 0.0
    
    # Column frequency (vertical changes)
    col_diff = np.diff(roi, axis=0)  # I[i,j] - I[i-1,j]
    col_mask = roi_mask[1:, :] & roi_mask[:-1, :]
    if np.sum(col_mask) > 0:
        cf = np.sqrt(np.mean((col_diff[col_mask > 0]) ** 2))
    else:
        cf = 0.0
    
    # Spatial frequency
    sf = np.sqrt(rf ** 2 + cf ** 2)
    
    return float(sf)

def compute_perimetric_complexity(image, mask):
    """
    Compute perimetric complexity of the text region.
    This measures the complexity of the text boundary.
    
    Formula:
        Perimetric_Complexity = (EdgeLength^2) / InkArea
    
    Args:
        image: Grayscale image (0-255)
        mask: Binary mask for the region
        
    Returns:
        Perimetric complexity value
    """
    # Convert to binary (0 or 1)
    # For text regions, we consider the masked area as "ink"
    binary = mask.astype(np.uint8)
    
    # Calculate ink area (number of pixels in the mask)
    ink_area = np.sum(binary)
    
    if ink_area == 0:
        return 0.0
    
    # Apply Canny edge detection on the masked region
    # First, apply mask to image
    masked_image = image.copy()
    masked_image[mask == 0] = 0
    
    # Canny edge detection
    edges = feature.canny(masked_image, sigma=1.0)
    
    # Calculate edge length (number of edge pixels)
    edge_length = np.sum(edges)
    
    if edge_length == 0:
        return 0.0
    
    # Perimetric complexity
    perimetric_complexity = (edge_length ** 2) / ink_area
    
    return float(perimetric_complexity)

def get_polygon_mask(polygon_coords, image_shape):
    """Create a binary mask for the polygon region"""
    mask = np.zeros(image_shape, dtype=np.uint8)
    points = np.array(polygon_coords, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    return mask

def compute_edge_density(img_gray, mask):
    """
    Compute edge density within the masked region
    Returns the ratio of edge pixels to total pixels in the region
    """
    # Apply Canny edge detection
    edges = cv2.Canny(img_gray, 50, 150)
    
    # Get edges within the mask
    masked_edges = cv2.bitwise_and(edges, edges, mask=mask)
    
    # Count edge pixels
    num_edge_pixels = np.count_nonzero(masked_edges)
    num_total_pixels = np.count_nonzero(mask)
    
    if num_total_pixels == 0:
        return 0.0
    
    edge_density = num_edge_pixels / num_total_pixels
    return edge_density

def compute_local_contrast(img_gray, mask):
    """
    Compute local contrast within the masked region
    Uses standard deviation as a measure of contrast
    """
    # Extract pixels within the mask
    masked_pixels = img_gray[mask > 0]
    
    if len(masked_pixels) == 0:
        return 0.0
    
    # Standard deviation represents local contrast
    contrast = np.std(masked_pixels)
    return float(contrast)

def compute_average_luminance(img_gray, mask):
    """
    Compute average luminance (brightness) within the masked region
    """
    # Extract pixels within the mask
    masked_pixels = img_gray[mask > 0]
    
    if len(masked_pixels) == 0:
        return 0.0
    
    # Mean pixel value represents average luminance
    luminance = np.mean(masked_pixels)
    return float(luminance)

def compute_polygon_area(polygon_coords):
    """Compute the area of a polygon in pixels"""
    try:
        poly = Polygon(polygon_coords)
        return poly.area
    except:
        return 0.0

def extract_features_from_region(img_gray, polygon_coords):
    """
    Extract all visual features from a text region defined by polygon
    
    Returns:
        dict with keys: polygon_size, edge_density, luminance_std, luminance_mean,
                       avg_curvature, gaussian_rms_contrast, spatial_frequency,
                       perimetric_complexity
    """
    # Compute polygon size
    polygon_size = compute_polygon_area(polygon_coords)
    
    # Compute polygon curvature
    avg_curvature = compute_polygon_curvature(polygon_coords)
    
    # Create mask for the polygon
    mask = get_polygon_mask(polygon_coords, img_gray.shape)
    
    # Compute features
    edge_density = compute_edge_density(img_gray, mask)
    local_contrast = compute_local_contrast(img_gray, mask)
    avg_luminance = compute_average_luminance(img_gray, mask)
    
    # Compute Gaussian RMS contrast (local contrast energy)
    gaussian_contrast = compute_local_contrast_energy(img_gray, mask, sigma=19.4)
    
    # Compute spatial frequency
    spatial_freq = compute_spatial_frequency(img_gray, mask)
    
    # Compute perimetric complexity
    perimetric_comp = compute_perimetric_complexity(img_gray, mask)
    
    return {
        'polygon_size': polygon_size,
        'edge_density': edge_density,
        'luminance_std': local_contrast,
        'luminance_mean': avg_luminance,
        'avg_curvature': avg_curvature,
        'gaussian_rms_contrast': gaussian_contrast,
        'spatial_frequency': spatial_freq,
        'perimetric_complexity': perimetric_comp
    }

def main():
    # Configuration
    GT_PATH = 'data/totaltext/anno.json'
    IMAGE_BASE_PATH = 'data/totaltext'
    OUTPUT_CSV = 'results/totaltext_16_text_region_features.csv'
    TARGET_DIR = '16/'  # Only process images in 16/ directory
    
    print("="*80)
    print("TotalText Text Region Feature Extraction")
    print(f"Target Directory: {TARGET_DIR}")
    print("="*80)
    
    # Load GT annotations
    print("\nLoading annotations...")
    with open(GT_PATH, 'r') as f:
        gt_data = json.load(f)
    
    # Create image lookup
    gt_images = {img['id']: img for img in gt_data['images']}
    
    # Filter annotations for images in 16/ directory
    print(f"Filtering annotations for {TARGET_DIR} images...")
    annotations_16 = []
    for ann in gt_data['annotations']:
        image_id = ann['image_id']
        img_info = gt_images.get(image_id, {})
        file_name = img_info.get('file_name', '')
        
        if file_name.startswith(TARGET_DIR):
            annotations_16.append({
                'annotation': ann,
                'image_info': img_info
            })
    
    print(f"Found {len(annotations_16)} annotations in {TARGET_DIR}")
    
    # Process each annotation
    print("\nExtracting features...")
    results = []
    
    # Track loaded images to avoid reloading
    image_cache = {}
    
    for item in tqdm(annotations_16):
        ann = item['annotation']
        img_info = item['image_info']
        
        image_id = ann['image_id']
        file_name = img_info.get('file_name', '')
        caption = ann.get('caption', '')
        
        # Split caption into words
        words = caption.split()
        bbox_list = ann.get('bbox', [])
        
        # Load image (cache to avoid reloading)
        if image_id not in image_cache:
            image_path = Path(IMAGE_BASE_PATH) / file_name
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                continue
            
            img_gray, img_color = load_image(image_path)
            image_cache[image_id] = (img_gray, img_color)
        else:
            img_gray, img_color = image_cache[image_id]
        
        # Process each word and its corresponding polygon
        for i, word in enumerate(words):
            if i >= len(bbox_list):
                continue
            
            polygon = bbox_list[i]
            
            # Validate polygon
            if not polygon or not isinstance(polygon, list) or len(polygon) < 3:
                continue
            
            # Skip "don't care" regions
            if word == "###":
                continue
            
            try:
                # Extract features
                features = extract_features_from_region(img_gray, polygon)
                H, W = img_gray.shape
                
                # Store results
                results.append({
                    'image_name': file_name,
                    'H': H,
                    'W': W,
                    'text': word,
                    'polygon_size': round(features['polygon_size'], 2),
                    'edge_density': round(features['edge_density'], 4),
                    'luminance_std': round(features['luminance_std'], 2),
                    'luminance_mean': round(features['luminance_mean'], 2),
                    'avg_curvature': round(features['avg_curvature'], 2),
                    'gaussian_rms_contrast': round(features['gaussian_rms_contrast'], 4),
                    'spatial_frequency': round(features['spatial_frequency'], 4),
                    'perimetric_complexity': round(features['perimetric_complexity'], 2)
                })
            except Exception as e:
                print(f"Error processing {file_name} - {word}: {e}")
                continue
    
    # Create DataFrame
    print(f"\nCreating DataFrame with {len(results)} text regions...")
    df = pd.DataFrame(results)
    
    # Display statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"\nTotal text regions processed: {len(df)}")
    print(f"Unique images: {df['image_name'].nunique()}")
    print(f"Unique words: {df['text'].nunique()}")
    
    print("\nFeature Statistics:")
    print("-" * 80)
    print(f"Polygon Size:")
    print(f"  Mean: {df['polygon_size'].mean():.2f} px²")
    print(f"  Std:  {df['polygon_size'].std():.2f} px²")
    print(f"  Min:  {df['polygon_size'].min():.2f} px²")
    print(f"  Max:  {df['polygon_size'].max():.2f} px²")
    
    print(f"\nEdge Density:")
    print(f"  Mean: {df['edge_density'].mean():.4f}")
    print(f"  Std:  {df['edge_density'].std():.4f}")
    print(f"  Min:  {df['edge_density'].min():.4f}")
    print(f"  Max:  {df['edge_density'].max():.4f}")
    
    print(f"\nLuminance Std (Local Contrast):")
    print(f"  Mean: {df['luminance_std'].mean():.2f}")
    print(f"  Std:  {df['luminance_std'].std():.2f}")
    print(f"  Min:  {df['luminance_std'].min():.2f}")
    print(f"  Max:  {df['luminance_std'].max():.2f}")
    
    print(f"\nLuminance Mean (Brightness):")
    print(f"  Mean: {df['luminance_mean'].mean():.2f}")
    print(f"  Std:  {df['luminance_mean'].std():.2f}")
    print(f"  Min:  {df['luminance_mean'].min():.2f}")
    print(f"  Max:  {df['luminance_mean'].max():.2f}")
    
    print(f"\nAverage Curvature:")
    print(f"  Mean: {df['avg_curvature'].mean():.2f}°")
    print(f"  Std:  {df['avg_curvature'].std():.2f}°")
    print(f"  Min:  {df['avg_curvature'].min():.2f}°")
    print(f"  Max:  {df['avg_curvature'].max():.2f}°")
    
    print(f"\nGaussian RMS Contrast:")
    print(f"  Mean: {df['gaussian_rms_contrast'].mean():.4f}")
    print(f"  Std:  {df['gaussian_rms_contrast'].std():.4f}")
    print(f"  Min:  {df['gaussian_rms_contrast'].min():.4f}")
    print(f"  Max:  {df['gaussian_rms_contrast'].max():.4f}")
    
    print(f"\nSpatial Frequency:")
    print(f"  Mean: {df['spatial_frequency'].mean():.4f}")
    print(f"  Std:  {df['spatial_frequency'].std():.4f}")
    print(f"  Min:  {df['spatial_frequency'].min():.4f}")
    print(f"  Max:  {df['spatial_frequency'].max():.4f}")
    
    print(f"\nPerimetric Complexity:")
    print(f"  Mean: {df['perimetric_complexity'].mean():.2f}")
    print(f"  Std:  {df['perimetric_complexity'].std():.2f}")
    print(f"  Min:  {df['perimetric_complexity'].min():.2f}")
    print(f"  Max:  {df['perimetric_complexity'].max():.2f}")
    
    # Save to CSV
    output_path = Path(OUTPUT_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print("\n" + "="*80)
    print(f"✅ Results saved to: {output_path.absolute()}")
    print("="*80)
    
    # Display sample rows
    print("\nSample data (first 10 rows):")
    print(df.head(10).to_string())

if __name__ == '__main__':
    main()


