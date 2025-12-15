"""
TotalText Detection & Recognition Evaluation - FINAL VERSION

This script evaluates text detection and recognition results against TotalText ground truth.
It handles both detection (IoU-based matching) and recognition (character/word level metrics).

Features:
- Detection matching using IoU threshold
- Character-level F1 score
- Word-level exact matching
- Edit distance calculation
- Per-subject Excel output with detailed word pairs and summary
- Support for "don't care" regions (###)
- Y-axis flipping for human data
- Detection order tracking

Usage:
    python totaltext_eval_FINAL.py

Outputs:
    - results/excel_outputs/{subject}_evaluation.xlsx (per subject)
    - Console summary of overall metrics
"""

import json
import os
import re
from pathlib import Path
from shapely.geometry import Polygon
from shapely import validation
import Levenshtein
import pandas as pd
from collections import defaultdict
from PIL import Image

# ============================================================================
# CONFIGURATION
# ============================================================================

GT_ANNO_PATH = "data/totaltext/anno.json"
MODEL_OUTPUT_PATH = "data/human/lowviz/converted"
IMAGE_BASE_PATH = "data/totaltext"  # Base path for actual image files
IOU_THRESHOLD = 0.1
RESULTS_DIR = "results/excel_outputs"

# ============================================================================
# TEXT NORMALIZATION (Based on official TotalText evaluation)
# ============================================================================

def normalize_text(text):
    """
    Normalize text for recognition evaluation.
    Based on official TotalText/SPTSv2 evaluation protocol.
    """
    # Convert to uppercase
    text = text.upper()
    
    # Handle apostrophes - keep them but normalize quotes
    text = text.replace('"', '').replace("'", "'")
    
    # Remove specific punctuation while keeping letters, numbers, apostrophes
    # This matches the official evaluation behavior
    text = re.sub(r'[^A-Z0-9\s\']', '', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text


def include_in_dictionary(transcription):
    """
    Check if text should be included in evaluation dictionary.
    Based on official TotalText evaluation criteria.
    
    Returns True if the text should be evaluated, False otherwise.
    """
    # Don't care regions
    if transcription == '###':
        return False
    
    # Must contain at least one alphanumeric character
    if not re.search(r'[A-Za-z0-9]', transcription):
        return False
    
    return True


# ============================================================================
# GEOMETRY & IoU
# ============================================================================

def compute_polygon_iou(poly1_coords, poly2_coords):
    """
    Compute IoU between two polygons.
    
    Args:
        poly1_coords: List of [x, y] coordinates
        poly2_coords: List of [x, y] coordinates
    
    Returns:
        IoU value (0-1), or 0 if invalid polygons
    """
    try:
        if len(poly1_coords) < 3 or len(poly2_coords) < 3:
            return 0.0
        
        # Convert to tuples for Shapely
        poly1_points = [(float(x), float(y)) for x, y in poly1_coords]
        poly2_points = [(float(x), float(y)) for x, y in poly2_coords]
        
        poly1 = Polygon(poly1_points)
        poly2 = Polygon(poly2_points)
        
        # Fix invalid polygons
        if not poly1.is_valid:
            poly1 = poly1.buffer(0)
        if not poly2.is_valid:
            poly2 = poly2.buffer(0)
        
        if not poly1.is_valid or not poly2.is_valid:
            return 0.0
        
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    except Exception as e:
        print(f"Error computing IoU: {e}")
        return 0.0


# ============================================================================
# RECOGNITION METRICS
# ============================================================================

def compute_char_level_metrics(pred_text, gt_text):
    """
    Compute character-level precision, recall, and F1.
    Texts should already be normalized.
    """
    if len(pred_text) == 0 and len(gt_text) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_text) == 0 or len(gt_text) == 0:
        return 0.0, 0.0, 0.0
    
    # Use edit distance to compute character-level metrics
    distance = Levenshtein.distance(pred_text, gt_text)
    max_len = max(len(pred_text), len(gt_text))
    
    if max_len == 0:
        return 1.0, 1.0, 1.0
    
    # Character-level F1 approximation
    accuracy = 1.0 - (distance / max_len)
    
    # For precision/recall, we approximate based on string lengths
    # This is a simplified approach
    precision = max(0.0, 1.0 - distance / len(pred_text))
    recall = max(0.0, 1.0 - distance / len(gt_text))
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1


def compute_word_level_metrics(pred_text, gt_text):
    """
    Compute word-level matching (exact match).
    Texts should already be normalized.
    """
    is_match = (pred_text == gt_text)
    
    # For word level: 1 if exact match, 0 otherwise
    precision = 1.0 if is_match else 0.0
    recall = 1.0 if is_match else 0.0
    f1 = 1.0 if is_match else 0.0
    
    return precision, recall, f1


# ============================================================================
# DETECTION MATCHING
# ============================================================================

def match_detections_to_gt(detections, ground_truths, iou_threshold):
    """
    Match detections to ground truth using greedy IoU-based matching.
    
    Strategy:
    1. Compute IoU for all detection-GT pairs
    2. Filter pairs with IoU < threshold
    3. Greedily match: pick highest IoU, assign, remove both from pool
    4. Repeat until no more matches possible
    
    Returns:
        matches: List of (det_idx, gt_idx, iou) tuples
    """
    matches = []
    
    if len(detections) == 0 or len(ground_truths) == 0:
        return matches
    
    # Compute all IoU pairs
    iou_matrix = []
    for det_idx, det in enumerate(detections):
        for gt_idx, gt in enumerate(ground_truths):
            iou = compute_polygon_iou(det['polygon'], gt['polygon'])
            if iou >= iou_threshold:
                iou_matrix.append((iou, det_idx, gt_idx))
    
    # Sort by IoU descending
    iou_matrix.sort(reverse=True, key=lambda x: x[0])
    
    # Greedy matching
    used_dets = set()
    used_gts = set()
    
    for iou, det_idx, gt_idx in iou_matrix:
        if det_idx not in used_dets and gt_idx not in used_gts:
            matches.append((det_idx, gt_idx, iou))
            used_dets.add(det_idx)
            used_gts.add(gt_idx)
    
    return matches


# ============================================================================
# IMAGE-LEVEL EVALUATION
# ============================================================================

def evaluate_image(detections, ground_truths, iou_threshold):
    """
    Evaluate detections against ground truth for a single image.
    
    Returns:
        dict with detection metrics, recognition metrics, and matched word pairs
    """
    # Match detections to GT
    matches = match_detections_to_gt(detections, ground_truths, iou_threshold)
    

    # Detection metrics
    num_detections = len(detections)
    num_gt = len(ground_truths)
    num_matches = len(matches)
    
    det_precision = num_matches / num_detections if num_detections > 0 else 0.0
    det_recall = num_matches / num_gt if num_gt > 0 else 0.0
    det_f1 = 2 * det_precision * det_recall / (det_precision + det_recall) if (det_precision + det_recall) > 0 else 0.0
    
    # Recognition metrics (only for care regions)
    char_precisions = []
    char_recalls = []
    char_f1s = []
    word_precisions = []
    word_recalls = []
    word_f1s = []
    ious = []
    edit_distances = []
    
    matched_word_pairs = []
    
    num_care_matches = 0
    num_care_gt = sum(1 for gt in ground_truths if not gt.get('dont_care', False))
    
    for det_idx, gt_idx, iou in matches:
        det = detections[det_idx]
        gt = ground_truths[gt_idx]
        
        # Get texts
        det_text_raw = det.get('text', '')
        gt_text_raw = gt.get('text', '')
        
        # Check if GT is don't care
        is_dont_care = gt.get('dont_care', False)
        
        # Apply filtering and normalization
        det_included = include_in_dictionary(det_text_raw)
        gt_included = include_in_dictionary(gt_text_raw) if not is_dont_care else False
        
        # Normalize texts
        det_text = normalize_text(det_text_raw) if det_included else ""
        gt_text = normalize_text(gt_text_raw) if gt_included else ""
        
        # Store word pair info
        word_pair_info = {
            'det_idx': det_idx,
            'gt_idx': gt_idx,
            'det_text_raw': det_text_raw,
            'gt_text_raw': gt_text_raw,
            'det_text': det_text,
            'gt_text': gt_text,
            'iou': iou,
            'dont_care': is_dont_care,
            'detection_order': det.get('detection_order', -1),
            'det_score': det.get('det_score', 0),
            'rec_score': det.get('rec_score', 0)
        }
        
        # Only compute recognition metrics for care regions that pass filtering
        if not is_dont_care and det_included and gt_included:
            num_care_matches += 1
            
            # Character-level
            char_p, char_r, char_f1 = compute_char_level_metrics(det_text, gt_text)
            char_precisions.append(char_p)
            char_recalls.append(char_r)
            char_f1s.append(char_f1)
            
            # Word-level (exact match)
            word_p, word_r, word_f1 = compute_word_level_metrics(det_text, gt_text)
            word_precisions.append(word_p)
            word_recalls.append(word_r)
            word_f1s.append(word_f1)
            
            # Edit distance
            edit_dist = Levenshtein.distance(det_text, gt_text)
            edit_distances.append(edit_dist)
            
            ious.append(iou)
            
            word_pair_info['char_f1'] = char_f1
            word_pair_info['word_match'] = (word_f1 == 1.0)
            word_pair_info['edit_distance'] = edit_dist
        
        matched_word_pairs.append(word_pair_info)
    
    # Aggregate recognition metrics
    avg_char_precision = sum(char_precisions) / len(char_precisions) if char_precisions else 0.0
    avg_char_recall = sum(char_recalls) / len(char_recalls) if char_recalls else 0.0
    avg_char_f1 = sum(char_f1s) / len(char_f1s) if char_f1s else 0.0
    
    avg_word_precision = sum(word_precisions) / len(word_precisions) if word_precisions else 0.0
    avg_word_recall = sum(word_recalls) / len(word_recalls) if word_recalls else 0.0
    avg_word_f1 = sum(word_f1s) / len(word_f1s) if word_f1s else 0.0
    
    avg_iou = sum(ious) / len(ious) if ious else 0.0
    avg_edit_distance = sum(edit_distances) / len(edit_distances) if edit_distances else 0.0
    
    return {
        # Detection
        'num_detections': num_detections,
        'num_gt': num_gt,
        'num_matches': num_matches,
        'det_precision': det_precision,
        'det_recall': det_recall,
        'det_f1': det_f1,
        
        # Recognition
        'char_precision': avg_char_precision,
        'char_recall': avg_char_recall,
        'char_f1': avg_char_f1,
        'word_precision': avg_word_precision,
        'word_recall': avg_word_recall,
        'word_f1': avg_word_f1,
        
        # Other
        'avg_iou': avg_iou,
        'avg_edit_distance': avg_edit_distance,
        'num_care_matches': num_care_matches,
        'num_care_gt': num_care_gt,
        
        # Word pairs for detailed output
        'matched_word_pairs': matched_word_pairs
    }


# ============================================================================
# DATA LOADING
# ============================================================================

def load_ground_truth(gt_path):
    """
    Load TotalText ground truth annotations (COCO format).
    
    Returns:
        gt_by_image: dict {image_id: [list of GT objects]}
        image_info: dict {image_id: image_info_dict}
    """
    with open(gt_path, 'r') as f:
        data = json.load(f)
    
    # Build image info mapping
    gt_images = data.get('images', [])
    gt_annotations = data.get('annotations', [])
    
    image_info = {img['id']: img for img in gt_images}
    gt_by_image = defaultdict(list)
    
    # Process annotations
    for ann in gt_annotations:
        image_id = ann['image_id']
        image_name = image_info.get(image_id, {}).get('file_name', '')
        
        # Filter for 16/ images
        if not image_name.startswith('16/'):
            continue
        
        # Get caption and bbox
        caption = ann.get('caption', '')
        bbox_list = ann.get('bbox', [])
        
        # Split caption by spaces
        words = caption.split()
        
        # Each word gets a corresponding bbox
        for idx, word in enumerate(words):
            if idx < len(bbox_list):
                polygon = bbox_list[idx]
                
                # Mark ### as don't care
                is_dont_care = (word == '###')
                
                gt_by_image[image_id].append({
                    'text': word,
                    'polygon': polygon,
                    'dont_care': is_dont_care,
                    'image_name': image_name
                })
    
    return gt_by_image, image_info


def load_model_output(output_path, image_info, image_base_path):
    """
    Load model detection and recognition outputs.
    Handles both directory of JSON files and single JSON file.
    
    Args:
        output_path: Path to model output JSON file(s)
        image_info: Dict mapping image_id to image info (from GT)
        image_base_path: Base path for actual image files
    
    Returns:
        dict: {(subject, image_id): [list of detection objects]}
    """
    detections_by_image = defaultdict(list)
    image_heights = {}  # Cache for image heights
    
    if os.path.isdir(output_path):
        json_files = list(Path(output_path).glob("*.json"))
    else:
        json_files = [Path(output_path)]
    
    for json_file in json_files:
        # Extract subject from filename (e.g., Sub161.json -> Sub161)
        subject = json_file.stem
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        for item in data:
            image_id = item.get('image_id')
            if image_id is None:
                continue
            
            key = (subject, image_id)
            
            # Get actual image height from file for y-axis flipping
            if image_id not in image_heights:
                img_info = image_info.get(image_id, {})
                image_path = Path(image_base_path) / img_info.get('file_name', '')
                try:
                    with Image.open(image_path) as img:
                        image_heights[image_id] = img.height
                except:
                    # Fallback to default
                    image_heights[image_id] = 720
            
            image_height = image_heights[image_id]
            
            # Parse rec_texts
            rec_texts = item.get('rec_texts', '')
            det_scores = item.get('det_score', [])
            rec_scores = item.get('rec_score', [])
            if isinstance(rec_texts, str) and rec_texts.startswith('['):
                try:
                    rec_texts = json.loads(rec_texts)
                except:
                    rec_texts = [rec_texts]
            elif not isinstance(rec_texts, list):
                rec_texts = [rec_texts] if rec_texts else []
            
            # Parse polys
            polys = item.get('polys', [])
            
            # Handle nested structure if present
            if isinstance(polys, str):
                try:
                    polys = json.loads(polys)
                except:
                    polys = []
            
            # Create detection objects
            for idx, poly in enumerate(polys):
                if isinstance(poly, list) and len(poly) >= 3:
                    # Remove score if present (keep only x, y)
                    polygon_points = poly
                    if polygon_points and isinstance(polygon_points[0], list) and len(polygon_points[0]) > 2:
                        polygon_points = [[p[0], p[1]] for p in polygon_points]
                    
                    # FLIP Y-AXIS for human data (y is inverted)
                    polygon_points = [[x, image_height - y] for x, y in polygon_points]
                    
                    text = rec_texts[idx] if idx < len(rec_texts) else ''
                    det_score = det_scores[idx] if idx < len(det_scores) else 0
                    rec_score = rec_scores[idx] if idx < len(rec_scores) else 0
                    
                    detections_by_image[key].append({
                        'polygon': polygon_points,
                        'text': text,
                        'detection_order': idx + 1,  # 1-indexed
                        'det_score': det_score,
                        'rec_score': rec_score
                    })
    
    return detections_by_image


# ============================================================================
# EXCEL OUTPUT
# ============================================================================

def save_subject_excel(subject, results_per_image, output_dir):
    """
    Save per-subject evaluation results to Excel with two sheets.
    
    Sheet 1 (Word Pairs): Detailed word-level information
    Sheet 2 (Summary): Image-level and overall summary
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{subject}_evaluation.xlsx")
    
    # === Sheet 1: Word Pairs ===
    word_pairs_data = []
    
    for (subj, img_id), metrics in results_per_image.items():
        if subj != subject:
            continue
        
        for pair in metrics['matched_word_pairs']:
            word_pairs_data.append({
                'Image': img_id,
                'Detection Order': pair.get('detection_order', -1),
                'GT Text': pair['gt_text_raw'],
                'Predicted Text': pair['det_text_raw'],
                'Det Score': round(pair.get('det_score', 0), 2),
                'Rec Score': round(pair.get('rec_score', 0), 2),
                'IoU': round(pair['iou'], 4),
                'Edit Distance': pair.get('edit_distance', ''),
                'Word Match': 'Yes' if pair.get('word_match', False) else 'No' if not pair['dont_care'] else 'N/A',
                'Char F1': round(pair.get('char_f1', 0), 4) if not pair['dont_care'] else 'N/A',
                'Dont Care': 'Yes' if pair['dont_care'] else 'No'
            })
    
    df_word_pairs = pd.DataFrame(word_pairs_data)
    
    # === Sheet 2: Summary ===
    summary_data = []
    
    # Per-image summaries
    for (subj, img_id), metrics in results_per_image.items():
        if subj != subject:
            continue
        
        summary_data.append({
            'Image': img_id,
            'Type': 'Image',
            'Word Precision': round(metrics['word_precision'], 4),
            'Word Recall': round(metrics['word_recall'], 4),
            'Word F1': round(metrics['word_f1'], 4),
            'Char Precision': round(metrics['char_precision'], 4),
            'Char Recall': round(metrics['char_recall'], 4),
            'Char F1': round(metrics['char_f1'], 4),
            'Avg IoU': round(metrics['avg_iou'], 4),
            'Matched Words': metrics['num_care_matches'],
            'Total GT Words': metrics['num_care_gt'],
            'Total Detections': metrics['num_detections'],
            'Avg Edit Distance': round(metrics['avg_edit_distance'], 2) if metrics['avg_edit_distance'] > 0 else 0
        })
    
    # Overall summary
    if summary_data:
        overall = {
            'Image': 'OVERALL',
            'Type': 'Summary',
            'Word Precision': round(sum(d['Word Precision'] for d in summary_data) / len(summary_data), 4),
            'Word Recall': round(sum(d['Word Recall'] for d in summary_data) / len(summary_data), 4),
            'Word F1': round(sum(d['Word F1'] for d in summary_data) / len(summary_data), 4),
            'Char Precision': round(sum(d['Char Precision'] for d in summary_data) / len(summary_data), 4),
            'Char Recall': round(sum(d['Char Recall'] for d in summary_data) / len(summary_data), 4),
            'Char F1': round(sum(d['Char F1'] for d in summary_data) / len(summary_data), 4),
            'Avg IoU': round(sum(d['Avg IoU'] for d in summary_data) / len(summary_data), 4),
            'Matched Words': sum(d['Matched Words'] for d in summary_data),
            'Total GT Words': sum(d['Total GT Words'] for d in summary_data),
            'Total Detections': sum(d['Total Detections'] for d in summary_data),
            'Avg Edit Distance': round(sum(d['Avg Edit Distance'] for d in summary_data) / len(summary_data), 2)
        }
        summary_data.append(overall)
    
    df_summary = pd.DataFrame(summary_data)
    
    # Write to Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_word_pairs.to_excel(writer, sheet_name='Word Pairs', index=False)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"  Saved: {output_path}")


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def evaluate_dataset():
    """Main evaluation function."""
    
    print("="*80)
    print("TotalText Detection & Recognition Evaluation - FINAL")
    print("="*80)
    
    # Load data
    print("\n[Loading Data]")
    print(f"Ground Truth: {GT_ANNO_PATH}")
    print(f"Model Output: {MODEL_OUTPUT_PATH}")
    print(f"Image Base Path: {IMAGE_BASE_PATH}")
    
    gt_by_image, image_info = load_ground_truth(GT_ANNO_PATH)
    det_by_image = load_model_output(MODEL_OUTPUT_PATH, image_info, IMAGE_BASE_PATH)
    
    print(f"Loaded {len(gt_by_image)} images with ground truth")
    print(f"Loaded {len(det_by_image)} images with detections")
    
    # Evaluate each image
    print("\n[Evaluating Images]")
    results_per_image = {}
    
    for key in det_by_image.keys():
        subject, image_id = key
        detections = det_by_image[key]
        ground_truths = gt_by_image.get(image_id, [])
        
        if len(ground_truths) == 0:
            print(f"Warning: No GT for {subject}/{image_id}")
            continue
        
        metrics = evaluate_image(detections, ground_truths, IOU_THRESHOLD)
        results_per_image[key] = metrics
    
    # Organize by subject
    results_by_subject = defaultdict(list)
    for key, metrics in results_per_image.items():
        subject, image_id = key
        results_by_subject[subject].append(metrics)
    
    # Generate Excel outputs per subject
    print(f"\n[Generating Excel Outputs]")
    print(f"Output Directory: {RESULTS_DIR}")
    
    for subject in results_by_subject.keys():
        save_subject_excel(subject, results_per_image, RESULTS_DIR)
    
    # Overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    
    all_metrics = list(results_per_image.values())
    
    avg_det_precision = sum(m['det_precision'] for m in all_metrics) / len(all_metrics)
    avg_det_recall = sum(m['det_recall'] for m in all_metrics) / len(all_metrics)
    avg_det_f1 = sum(m['det_f1'] for m in all_metrics) / len(all_metrics)
    
    avg_char_precision = sum(m['char_precision'] for m in all_metrics) / len(all_metrics)
    avg_char_recall = sum(m['char_recall'] for m in all_metrics) / len(all_metrics)
    avg_char_f1 = sum(m['char_f1'] for m in all_metrics) / len(all_metrics)
    
    avg_word_precision = sum(m['word_precision'] for m in all_metrics) / len(all_metrics)
    avg_word_recall = sum(m['word_recall'] for m in all_metrics) / len(all_metrics)
    avg_word_f1 = sum(m['word_f1'] for m in all_metrics) / len(all_metrics)
    
    total_matched = sum(m['num_care_matches'] for m in all_metrics)
    total_gt = sum(m['num_care_gt'] for m in all_metrics)
    total_det = sum(m['num_detections'] for m in all_metrics)
    
    print(f"\nImages Evaluated: {len(all_metrics)}")
    print(f"Subjects: {len(results_by_subject)}")
    
    print(f"\n[Detection Metrics]")
    print(f"Precision: {avg_det_precision:.4f}")
    print(f"Recall: {avg_det_recall:.4f}")
    print(f"F1: {avg_det_f1:.4f}")
    
    print(f"\n[Recognition Metrics - Character Level]")
    print(f"Precision: {avg_char_precision:.4f}")
    print(f"Recall: {avg_char_recall:.4f}")
    print(f"F1: {avg_char_f1:.4f}")
    
    print(f"\n[Recognition Metrics - Word Level (Exact Match)]")
    print(f"Precision: {avg_word_precision:.4f}")
    print(f"Recall: {avg_word_recall:.4f}")
    print(f"F1: {avg_word_f1:.4f}")
    
    print(f"\n[Counts]")
    print(f"Total Matched Words (care): {total_matched}")
    print(f"Total GT Words (care): {total_gt}")
    print(f"Total Detections: {total_det}")
    
    print("\n" + "="*80)
    print(f"Excel outputs saved to: {RESULTS_DIR}/")
    print("="*80)


if __name__ == "__main__":
    evaluate_dataset()

