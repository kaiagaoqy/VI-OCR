"""
Batch Process Subjects Script
Process all subjects with their individual vision parameters.
Uses the pre-generated sub_vision_rec_to_hvshift.csv file.
"""

import os
import sys
import json
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference_pipeline import FilteredInferencePipeline
from dotenv import load_dotenv

load_dotenv(".env")


def find_subject_image(subject_id: str, base_dirs: list) -> str:
    """
    Find image file for a subject across multiple possible directories.
    
    Args:
        subject_id: Subject ID (e.g., 'Sub123')
        base_dirs: List of base directories to search
        
    Returns:
        Path to the image file, or None if not found
    """
    # Possible file extensions
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    # Possible file naming patterns
    patterns = [
        f'{subject_id}{{ext}}',                    # Sub123.jpg
        f'{subject_id.lower()}{{ext}}',            # sub123.jpg
        f'{subject_id}/image{{ext}}',              # Sub123/image.jpg
        f'{subject_id}/img{{ext}}',                # Sub123/img.jpg
        f'{subject_id}/{subject_id}{{ext}}',       # Sub123/Sub123.jpg
    ]
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            continue
            
        for pattern in patterns:
            for ext in extensions:
                file_pattern = pattern.format(ext=ext)
                full_path = os.path.join(base_dir, file_pattern)
                
                if os.path.exists(full_path):
                    return full_path
    
    return None


def batch_process_subjects(
    subject_csv_path: str,
    model_type: str,
    model_path: str,
    output_dir: str,
    anno_json_path: str = None,
    img_dir: str = None,
    filtered_image_dir: str = None,
    prompt: str = None,
    model_size: str = 'Tiny',
    resume: bool = True,
    save_filtered_images: bool = False,
    shared_filtered_dir: str = "outputs/filtered_images"
):
    """
    Batch process all subjects with their individual parameters.
    
    Process flow:
    - All subjects use the SAME set of original images (from anno.json)
    - Each subject has different hshift/vshift parameters
    - For each subject, process all images with that subject's filter parameters
    
    Two modes of operation:
    1. Use pre-saved filtered images (if filtered_image_dir is provided)
       - filtered_image_dir/SubjectID/image1.jpg
       - filtered_image_dir/SubjectID/image2.jpg
    2. Apply filter in real-time (if anno_json_path and img_dir are provided)
       - Read images from img_dir
       - Apply each subject's filter parameters
       - Optionally save to shared_filtered_dir/SubjectID/
    
    Args:
        subject_csv_path: Path to sub_vision_rec_to_hvshift.csv
        model_type: Type of model (qwen2, gemini, dsocr, claude, gpt, cogvlm)
        model_path: Path to the model
        output_dir: Output directory for results
        anno_json_path: Path to anno.json file (Mode 2)
        img_dir: Base directory for original images (Mode 2)
        filtered_image_dir: Base directory with pre-saved filtered images (Mode 1)
                          Expected structure: filtered_image_dir/SubjectID/images/
        prompt: Text prompt for inference (default: from .env)
        model_size: Model size (for DeepSeek OCR)
        resume: Skip already processed subjects
        save_filtered_images: Save filtered images to disk (only for Mode 2)
        shared_filtered_dir: Shared directory for filtered images (default: outputs/filtered_images)
                           This is shared across all models to avoid duplication
    """
    # Validate mode selection
    if filtered_image_dir is None and (anno_json_path is None or img_dir is None):
        raise ValueError(
            "Must provide either:\n"
            "  - filtered_image_dir (Mode 1), or\n"
            "  - anno_json_path + img_dir (Mode 2)"
        )
    
    # Determine mode
    use_prefiltered = filtered_image_dir is not None
    mode_name = "Mode 1: Using pre-filtered images" if use_prefiltered else "Mode 2: Real-time filtering"
    
    print(f"\n{'='*70}")
    print(f"Batch Processing Mode: {mode_name}")
    print(f"{'='*70}")
    # Load subject parameters
    print(f"Loading subject parameters from {subject_csv_path}...")
    subject_df = pd.read_csv(subject_csv_path)
    
    # Remove empty rows
    subject_df = subject_df.dropna(subset=['subject_id'])
    
    print(f"Found {len(subject_df)} subjects to process")
    
    # Load images list
    samples = []
    if not use_prefiltered and anno_json_path:
        print(f"Loading image list from {anno_json_path}...")
        with open(anno_json_path, 'r') as f:
            anno_data = json.load(f)
        
        # Get images list (Filter_no = 16 for original images)
        if 'images' in anno_data:
            all_images = anno_data['images']
            # Filter for original images (Filter_no = 16)
            samples = [img for img in all_images if img.get('Filter_no', 0) == 16]
            print(f"Found {len(samples)} original images in anno.json")
            print(f"All subjects will process the same {len(samples)} images with their individual filter parameters")
        else:
            print("Warning: No 'images' key in anno.json")
            samples = []
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Load prompt
    if prompt is None:
        prompt = os.getenv("Prompt", "Read all text in this image.")
    
    # Adjust prompt for DeepSeek OCR
    if model_type == 'dsocr':
        prompt = '<image>\nFree OCR.'
    
    # Initialize pipeline
    print(f"\nInitializing {model_type} model...")
    model_kwargs = {}
    if model_type == 'dsocr':
        model_kwargs['size'] = model_size
    
    pipeline = FilteredInferencePipeline(
        model_type=model_type,
        model_path=model_path,
        **model_kwargs
    )
    
    # Track statistics
    stats = {
        'total_subjects': len(subject_df),
        'total_images': len(samples) if not use_prefiltered else 0,
        'processed_subjects': 0,
        'skipped_subjects': 0,
        'processed_images': 0,
        'errors': 0
    }
    
    # Prepare output files
    summary_jsonl = os.path.join(output_dir, 'all_results.jsonl')
    
    # Load already processed (subject, image) pairs if resuming
    processed_pairs = set()
    if resume and os.path.exists(summary_jsonl):
        with open(summary_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    processed_pairs.add((obj['subject_id'], obj['image_id']))
                except:
                    pass
        print(f"Resuming: Found {len(processed_pairs)} already processed (subject, image) pairs")
    
    # Process each subject
    print("\nProcessing subjects...")
    print("=" * 70)
    
    with open(summary_jsonl, 'a', encoding='utf-8') as f_summary:
        for subj_idx, row in enumerate(subject_df.iterrows()):
            _, row = row
            subject_id = row['subject_id']
            
            hshift = float(row['hshift'])
            vshift = float(row['vshift'])
            va = float(row['va_measured'])
            cs = float(row['cs_measured'])
            
            print(f"\n{'='*70}")
            print(f"Subject [{subj_idx+1}/{len(subject_df)}]: {subject_id}")
            print(f"Parameters: VA={va:.2f}, CS={cs:.2f}, hshift={hshift:.4f}, vshift={vshift:.4f}")
            print(f"{'='*70}")
            
            # Create subject output directory
            subject_output_dir = os.path.join(output_dir, subject_id)
            os.makedirs(subject_output_dir, exist_ok=True)
            
            # Create filtered images directory if needed (in shared location)
            if save_filtered_images and not use_prefiltered:
                subject_filtered_dir = os.path.join(shared_filtered_dir, subject_id)
                os.makedirs(subject_filtered_dir, exist_ok=True)
                print(f"Filtered images will be saved to: {subject_filtered_dir}")
            
            # Process all images for this subject
            subject_results = []
            
            if use_prefiltered:
                # Mode 1: Use pre-saved filtered images for this subject
                subject_filtered_dir = os.path.join(filtered_image_dir, subject_id)
                
                if not os.path.exists(subject_filtered_dir):
                    print(f"⚠️  Filtered images directory not found: {subject_filtered_dir}")
                    stats['skipped_subjects'] += 1
                    continue
                
                # Find all images in subject's filtered directory
                import glob
                filtered_images = glob.glob(os.path.join(subject_filtered_dir, '*.jpg')) + \
                                glob.glob(os.path.join(subject_filtered_dir, '*.png'))
                
                print(f"Found {len(filtered_images)} filtered images for {subject_id}")
                
                for img_path in tqdm(filtered_images, desc=f"Processing {subject_id}", leave=False):
                    img_filename = os.path.basename(img_path)
                    image_id = img_filename.split('.')[0]
                    
                    # Skip if already processed
                    if (subject_id, image_id) in processed_pairs:
                        continue
                    
                    try:
                        # Direct inference (already filtered)
                        result = pipeline.model.infer(image=img_path, prompt=prompt)
                        
                        result_dict = {
                            'subject_id': subject_id,
                            'image_id': image_id,
                            'image_filename': img_filename,
                            'va_measured': va,
                            'cs_measured': cs,
                            'hshift': hshift,
                            'vshift': vshift,
                            'rec_texts': result,
                            'mode': 'pre-filtered'
                        }
                        
                        subject_results.append(result_dict)
                        f_summary.write(json.dumps(result_dict, ensure_ascii=False) + '\n')
                        f_summary.flush()
                        stats['processed_images'] += 1
                        
                    except Exception as e:
                        print(f"✗ Error processing {img_filename}: {e}")
                        stats['errors'] += 1
                
            else:
                # Mode 2: Process original images with this subject's filter parameters
                print(f"Processing {len(samples)} images with subject's filter parameters...")
                
                for sample in tqdm(samples, desc=f"Processing {subject_id}", leave=False):
                    image_id = str(sample.get('id', ''))
                    img_filename = sample['file_name'].split('/')[-1]
                    
                    # Skip if already processed
                    if (subject_id, image_id) in processed_pairs:
                        continue
                    
                    # Construct image path
                    image_path = os.path.join(img_dir, img_filename)
                    
                    if not os.path.exists(image_path):
                        print(f"⚠️  Image not found: {image_path}")
                        continue
                    
                    try:
                        # Apply filter and infer
                        result = pipeline.infer_single_image(
                            image_path=image_path,
                            prompt=prompt,
                            hshift=hshift,
                            vshift=vshift
                        )
                        
                        result_dict = {
                            'subject_id': subject_id,
                            'image_id': image_id,
                            'image_filename': img_filename,
                            'va_measured': va,
                            'cs_measured': cs,
                            'hshift': hshift,
                            'vshift': vshift,
                            'rec_texts': result,
                            'mode': 'real-time-filter'
                        }
                        
                        subject_results.append(result_dict)
                        f_summary.write(json.dumps(result_dict, ensure_ascii=False) + '\n')
                        f_summary.flush()
                        stats['processed_images'] += 1
                        
                        # Optionally save filtered image
                        if save_filtered_images:
                            from utils.image_processor import ImageFilterProcessor
                            from PIL import Image
                            
                            processor = ImageFilterProcessor()
                            img = Image.open(image_path)
                            filtered_img = processor.apply_filter_to_pil(img, hshift, vshift)
                            
                            filtered_path = os.path.join(subject_filtered_dir, img_filename)
                            filtered_img.save(filtered_path)
                        
                    except Exception as e:
                        print(f"✗ Error processing {img_filename}: {e}")
                        stats['errors'] += 1
            
            # Save subject's results
            if subject_results:
                subject_json = os.path.join(subject_output_dir, 'results.json')
                with open(subject_json, 'w', encoding='utf-8') as f:
                    json.dump(subject_results, f, indent=2, ensure_ascii=False)
                
                print(f"✓ Saved {len(subject_results)} results for {subject_id}")
                stats['processed_subjects'] += 1
            else:
                print(f"⚠️  No results for {subject_id}")
                stats['skipped_subjects'] += 1
    
    # Cleanup
    pipeline.cleanup()
    
    # Convert summary JSONL to JSON
    print("\nConverting summary to JSON format...")
    summary_json = os.path.join(output_dir, 'all_results.json')
    
    with open(summary_jsonl, 'r', encoding='utf-8') as fin:
        with open(summary_json, 'w', encoding='utf-8') as fout:
            results = [json.loads(line) for line in fin if line.strip()]
            json.dump(results, fout, indent=2, ensure_ascii=False)
    
    # Print statistics
    print("\n" + "=" * 70)
    print("Processing Complete!")
    print("=" * 70)
    print(f"Total subjects:         {stats['total_subjects']}")
    print(f"Processed subjects:     {stats['processed_subjects']}")
    print(f"Skipped subjects:       {stats['skipped_subjects']}")
    if not use_prefiltered:
        print(f"Total images per subject: {stats['total_images']}")
    print(f"Total processed images: {stats['processed_images']}")
    print(f"Errors:                 {stats['errors']}")
    print("=" * 70)
    print(f"\nResults saved to:")
    print(f"  - Summary (JSONL):     {summary_jsonl}")
    print(f"  - Summary (JSON):      {summary_json}")
    print(f"  - Per-subject results: {output_dir}/SubjectID/results.json")
    if save_filtered_images and not use_prefiltered:
        print(f"  - Filtered images:     {shared_filtered_dir}/SubjectID/")
        print(f"    (Shared across all models)")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Batch process all subjects with their vision parameters'
    )
    
    # Required arguments
    parser.add_argument(
        '--subject_csv',
        type=str,
        default='data/human/sub_vision_rec_to_hvshift.csv',
        help='Path to subject parameters CSV file'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        required=True,
        choices=['qwen2', 'gemini', 'dsocr','claude','gpt','cogvlm'],
        help='Type of model to use'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the model or model identifier'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for results'
    )
    
    # Image sources (Mode selection)
    parser.add_argument(
        '--anno_json',
        type=str,
        default=None,
        help='Path to anno.json file (Mode 2: Real-time filtering)'
    )
    parser.add_argument(
        '--img_dir',
        type=str,
        default=None,
        help='Base directory containing original images (Mode 2: Real-time filtering)'
    )
    parser.add_argument(
        '--filtered_image_dir',
        type=str,
        default=None,
        help='Directory containing pre-filtered images (Mode 1: Use pre-saved filtered images)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='Text prompt for inference (default: from .env)'
    )
    parser.add_argument(
        '--size',
        type=str,
        default='Tiny',
        choices=['Tiny', 'Small', 'Base', 'Large', 'Gundam'],
        help='Model size (for DeepSeek OCR only)'
    )
    parser.add_argument(
        '--no_resume',
        action='store_true',
        help='Do not resume from existing results'
    )
    parser.add_argument(
        '--save_filtered_images',
        action='store_true',
        help='Save filtered images to disk (only for Mode 2)'
    )
    parser.add_argument(
        '--shared_filtered_dir',
        type=str,
        default='outputs/filtered_images',
        help='Shared directory for filtered images (default: outputs/filtered_images). '
             'This directory is shared across all models to avoid duplication.'
    )
    
    args = parser.parse_args()
    
    # Validate mode selection
    if args.filtered_image_dir is None and (args.anno_json is None or args.img_dir is None):
        # Set defaults for Mode 2
        if args.anno_json is None:
            args.anno_json = 'data/totaltext/anno.json'
            print(f"No anno.json specified, using default: {args.anno_json}")
        if args.img_dir is None:
            args.img_dir = 'data/totaltext/16'
            print(f"No img_dir specified, using default: {args.img_dir}")
    
    if args.filtered_image_dir and (args.anno_json or args.img_dir):
        print("⚠️  Warning: Both filtered_image_dir and anno_json/img_dir specified.")
        print("    Will use Mode 1 (pre-filtered images) and ignore anno_json/img_dir.")
    
    # Run batch processing
    batch_process_subjects(
        subject_csv_path=args.subject_csv,
        model_type=args.model_type,
        model_path=args.model_path,
        output_dir=args.output_dir,
        anno_json_path=args.anno_json,
        img_dir=args.img_dir,
        filtered_image_dir=args.filtered_image_dir,
        prompt=args.prompt,
        model_size=args.size,
        resume=not args.no_resume,
        save_filtered_images=args.save_filtered_images,
        shared_filtered_dir=args.shared_filtered_dir
    )


if __name__ == '__main__':
    main()

