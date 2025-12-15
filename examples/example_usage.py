"""
Example Usage of the Filtered Inference Pipeline
Demonstrates how to use the new modular architecture for image filtering and inference.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference_pipeline import FilteredInferencePipeline
from utils.image_processor import ImageFilterProcessor
from dotenv import load_dotenv
load_dotenv(".env")
PROMPT = os.getenv("Prompt")

def example_1_basic_usage():
    """
    Example 1: Basic usage with a single image
    """
    print("=" * 60)
    print("Example 1: Basic Single Image Inference with Filter")
    print("=" * 60)
    
    # Initialize the pipeline
    pipeline = FilteredInferencePipeline(
        model_type='qwen2',
        model_path='Qwen/Qwen2.5-VL-7B-Instruct'
    )
    
    # Define filter parameters
    hshift = 0.288  # Horizontal shift (affects blur)
    vshift = 0.288  # Vertical shift (affects contrast)
    
    # Perform inference
    result = pipeline.infer_single_image(
        image_path='path/to/your/image.jpg',
        prompt=PROMPT,
        hshift=hshift,
        vshift=vshift
    )
    
    print(f"Result: {result}")
    
    # Cleanup
    pipeline.cleanup()


def example_2_using_filter_id():
    """
    Example 2: Using predefined filter IDs
    """
    print("\n" + "=" * 60)
    print("Example 2: Using Predefined Filter ID")
    print("=" * 60)
    
    # Get hshift and vshift from filter ID
    filter_id = 2  # Use filter #2 from the standard list
    hshift, vshift = ImageFilterProcessor.convert_filter_id_to_shifts(filter_id)
    
    print(f"Filter ID {filter_id} -> hshift={hshift}, vshift={vshift}")
    
    # Initialize the pipeline
    pipeline = FilteredInferencePipeline(
        model_type='gemini',
        model_path='gemini-1.5-flash'
    )
    
    # Perform inference
    result = pipeline.infer_single_image(
        image_path='path/to/your/image.jpg',
        prompt=PROMPT,
        hshift=hshift,
        vshift=vshift
    )
    
    print(f"Result: {result}")
    
    pipeline.cleanup()


def example_3_batch_processing():
    """
    Example 3: Batch processing with JSON input
    """
    print("\n" + "=" * 60)
    print("Example 3: Batch Processing")
    print("=" * 60)
    
    import json
    
    # Load samples from JSON file
    samples = json.load(open('data/totaltext/anno.json', 'r'))['images']
    samples = [sample for sample in samples if sample["Filter_no"] in [16] ] # filter 16 is the original image
    
    # Initialize the pipeline
    model_type = 'dsocr'
    pipeline = FilteredInferencePipeline(
        model_type= model_type,
        model_path='deepseek-ai/DeepSeek-OCR',
        size='Tiny'
    )
    
    # Define filter parameters
    filter_id = 5
    hshift, vshift = ImageFilterProcessor.convert_filter_id_to_shifts(filter_id)
    
      # Process batch
    results = pipeline.infer_batch(
          samples=samples[:10],  # Process first 10 samples
          img_dir='data/totaltext/16',
          prompt=PROMPT if model_type != 'dsocr' else '<image>\nFree OCR.',
          hshift=hshift,
          vshift=vshift,
          output_file='output/results.jsonl',
          filter_id=filter_id,
          resize=False,  # Do not resize it back to the original size
          resume=True
      )
    
    print(f"Processed {len(results)} images")
    
    pipeline.cleanup()


def example_4_custom_filter_settings():
    """
    Example 4: Using custom filter settings
    """
    print("\n" + "=" * 60)
    print("Example 4: Custom Filter Settings")
    print("=" * 60)
    
    # Create custom filter processor
    filter_processor = ImageFilterProcessor(
        screen_reso=(1920, 1200),  # Custom screen resolution
        screen_size=16.0,           # Custom screen size
        camera=False,               # Not from camera
        white_balance=False          # DO not Apply white balance
    )
    
    # Initialize pipeline with custom filter processor
    pipeline = FilteredInferencePipeline(
        model_type='qwen2',
        model_path='Qwen/Qwen2.5-VL-7B-Instruct',
        filter_processor=filter_processor
    )
    
    # Perform inference
    result = pipeline.infer_single_image(
        image_path='path/to/your/image.jpg',
        prompt=PROMPT,
        hshift=0.5,
        vshift=0.5
    )
    
    print(f"Result: {result}")
    
    pipeline.cleanup()


def example_5_direct_model_usage():
    """
    Example 5: Using model classes directly with lazy import
    """
    print("\n" + "=" * 60)
    print("Example 5: Direct Model Usage with Filter")
    print("=" * 60)
    
    from models import ModelFactory
    from utils.image_processor import ImageFilterProcessor
    
    # Create model instance using factory (lazy import)
    model = ModelFactory.create_model(
        model_type='qwen2',
        model_path='Qwen/Qwen2.5-VL-7B-Instruct'
    )
    
    # Initialize filter processor
    filter_processor = ImageFilterProcessor()
    
    # Perform inference with filter
    result = model.infer_with_filter(
        image='path/to/your/image.jpg',
        prompt=PROMPT,
        hshift=0.288,
        vshift=0.288,
        filter_processor=filter_processor
    )
    
    print(f"Result: {result}")
    
    model.cleanup()


def example_6_process_multiple_filters():
    """
    Example 6: Process the same image with multiple filters
    """
    print("\n" + "=" * 60)
    print("Example 6: Multiple Filters on Same Image")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = FilteredInferencePipeline(
        model_type='gemini',
        model_path='gemini-1.5-flash'
    )
    
    # Process with multiple filter IDs
    filter_ids = [1, 2, 3, 4, 5]
    image_path = 'path/to/your/image.jpg'
    
    results = {}
    for filter_id in filter_ids:
        hshift, vshift = ImageFilterProcessor.convert_filter_id_to_shifts(filter_id)
        
        result = pipeline.infer_single_image(
            image_path=image_path,
            prompt=PROMPT,
            hshift=hshift,
            vshift=vshift
        )
        
        results[filter_id] = result
        print(f"Filter {filter_id}: {result}")
    
    pipeline.cleanup()


def example_7_batch_process_subjects():
    """
    Example 7: Batch process all subjects with their individual parameters
    Uses the pre-generated sub_vision_rec_to_hvshift.csv file
    """
    print("\n" + "=" * 60)
    print("Example 7: Batch Process All Subjects")
    print("=" * 60)
    
    import pandas as pd
    
    # Load subject parameters from CSV
    subject_params_path = 'data/human/sub_vision_rec_to_hvshift.csv'
    subject_df = pd.read_csv(subject_params_path)
    
    print(f"Loaded {len(subject_df)} subjects from {subject_params_path}")
    
    # Configuration
    model_type = 'dsocr'
    model_path = 'deepseek-ai/DeepSeek-OCR'
    model_size = 'Tiny'
    
    # Image directory (assumes images are named like SubXXX.jpg or in subdirectories)
    img_dir = 'data/totaltext/16'  # Modify to your actual image directory
    
    # Output directory
    output_dir = f'output/{model_type}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize pipeline
    pipeline = FilteredInferencePipeline(
        model_type=model_type,
        model_path=model_path,
        size=model_size
    )
    
    # Process each subject
    for idx, row in subject_df.iterrows():
        subject_id = row['subject_id']
        hshift = row['hshift']
        vshift = row['vshift']
        va = row['va_measured']
        cs = row['cs_measured']
        
        print(f"\n[{idx+1}/{len(subject_df)}] Processing {subject_id}...")
        print(f"  VA: {va:.2f}, CS: {cs:.2f}, hshift: {hshift:.4f}, vshift: {vshift:.4f}")
        
        # Find image for this subject
        # Try multiple possible image paths
        possible_paths = [
            os.path.join(img_dir, f'{subject_id}.jpg'),
            os.path.join(img_dir, f'{subject_id}.png'),
            os.path.join(img_dir, subject_id, 'image.jpg'),
            os.path.join('data/human/lowviz/images', f'{subject_id}.jpg'),
        ]
        
        image_path = None
        for path in possible_paths:
            if os.path.exists(path):
                image_path = path
                break
        
        if image_path is None:
            print(f"  ⚠️  Warning: Image not found for {subject_id}, skipping...")
            continue
        
        try:
            # Perform inference
            prompt = PROMPT if model_type != 'dsocr' else '<image>\nFree OCR.'
            result = pipeline.infer_single_image(
                image_path=image_path,
                prompt=prompt,
                hshift=hshift,
                vshift=vshift
            )
            
            # Prepare result dictionary
            result_dict = {
                'subject_id': subject_id,
                'va_measured': float(va),
                'cs_measured': float(cs),
                'hshift': float(hshift),
                'vshift': float(vshift),
                'image_path': image_path,
                'inference_result': result
            }
            
            # Save to JSONL (one subject per line)
            output_jsonl = os.path.join(output_dir, f'{subject_id}.jsonl')
            with open(output_jsonl, 'w', encoding='utf-8') as f:
                f.write(json.dumps(result_dict, ensure_ascii=False) + '\n')
            
            # Also save to JSON for easier reading
            output_json = os.path.join(output_dir, f'{subject_id}.json')
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dumps(result_dict, f, indent=2, ensure_ascii=False)
            
            print(f"  ✓ Saved to {output_json}")
            print(f"  Result: {result[:100]}..." if len(result) > 100 else f"  Result: {result}")
            
        except Exception as e:
            print(f"  ✗ Error processing {subject_id}: {e}")
            continue
    
    pipeline.cleanup()
    
    print("\n" + "=" * 60)
    print(f"Processing complete! Results saved to {output_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Filtered Inference Pipeline - Usage Examples")
    print("=" * 60)
    
    # Note: These examples are for demonstration purposes.
    # You'll need to modify the paths and ensure you have the required models.
    
    print("\nThese are example functions. Please modify the paths and")
    print("uncomment the function you want to run.")
    print("\nAvailable examples:")
    print("  1. example_1_basic_usage() - Basic single image inference")
    print("  2. example_2_using_filter_id() - Using predefined filter IDs")
    print("  3. example_3_batch_processing() - Batch processing")
    print("  4. example_4_custom_filter_settings() - Custom filter settings")
    print("  5. example_5_direct_model_usage() - Direct model usage")
    print("  6. example_6_process_multiple_filters() - Multiple filters")
    print("  7. example_7_batch_process_subjects() - ✨ Batch process all subjects")
    
    # Uncomment the example you want to run:
    # example_1_basic_usage()
    # example_2_using_filter_id()
    # example_3_batch_processing()
    # example_4_custom_filter_settings()
    # example_5_direct_model_usage()
    # example_6_process_multiple_filters()
    # example_7_batch_process_subjects()

