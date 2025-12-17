#!/bin/bash

# Example Command-Line Usage of the Filtered Inference Pipeline
# These examples demonstrate how to use the inference_pipeline.py script

echo "============================================================"
echo "Filtered Inference Pipeline - Command Line Examples"
echo "============================================================"

# Example 1: Basic usage with Qwen2 model
echo ""
echo "Example 1: Basic inference with Qwen2 model"
echo "------------------------------------------------------------"
cat << 'EOF'
python inference_pipeline.py \
    --infile data/totaltext/anno.json \
    --outfile output/totaltext_filtered \
    --img_dir data/totaltext/16 \
    --model_type qwen2 \
    --model_path Qwen/Qwen2.5-VL-7B-Instruct \
    --hshift 3.4722 \
    --vshift 0.288 \
    --filter_id 2 \
    --prompt "Read all text in this image."
EOF

# Example 2: Using Gemini model
echo ""
echo "Example 2: Using Gemini model"
echo "------------------------------------------------------------"
cat << 'EOF'
python inference_pipeline.py \
    --infile data/viocr/anno.json \
    --outfile output/viocr_filtered \
    --img_dir data/viocr/selected_images_new \
    --model_type gemini \
    --model_path gemini-1.5-flash \
    --hshift 6.3694 \
    --vshift 0.157 \
    --filter_id 3
EOF

# Example 3: Using DeepSeek OCR model
echo ""
echo "Example 3: Using DeepSeek OCR model"
echo "------------------------------------------------------------"
cat << 'EOF'
python inference_pipeline.py \
    --infile data/totaltext/anno.json \
    --outfile output/dsocr_filtered \
    --img_dir data/totaltext/16 \
    --model_type dsocr \
    --model_path deepseek-ai/DeepSeek-OCR \
    --size Gundam \
    --hshift 11.6279 \
    --vshift 0.086 \
    --filter_id 4
EOF

# Example 4: Processing with multiple filter levels (using a loop)
echo ""
echo "Example 4: Processing with multiple filters (loop)"
echo "------------------------------------------------------------"
cat << 'EOF'
# Define filter parameters (filter_id: hshift, vshift)
declare -A FILTERS=(
    [2]="3.4722 0.288"
    [3]="6.3694 0.157"
    [4]="11.6279 0.086"
    [5]="20.8333 0.048"
)

for filter_id in "${!FILTERS[@]}"; do
    read hshift vshift <<< "${FILTERS[$filter_id]}"
    
    python inference_pipeline.py \
        --infile data/totaltext/anno.json \
        --outfile output/totaltext_filter_${filter_id} \
        --img_dir data/totaltext/16 \
        --model_type qwen2 \
        --model_path Qwen/Qwen2.5-VL-7B-Instruct \
        --hshift $hshift \
        --vshift $vshift \
        --filter_id $filter_id
done
EOF

# Example 5: Using custom prompt from command line
echo ""
echo "Example 5: Using custom prompt"
echo "------------------------------------------------------------"
cat << 'EOF'
python inference_pipeline.py \
    --infile data/mnread/anno.json \
    --outfile output/mnread_filtered \
    --img_dir data/mnread \
    --model_type qwen2 \
    --model_path Qwen/Qwen2.5-VL-7B-Instruct \
    --hshift 3.4722 \
    --vshift 0.288 \
    --filter_id 2 \
    --prompt "Please read all the words you can see in this image, from left to right, top to bottom."
EOF

# Example 6: Disable resume (start fresh)
echo ""
echo "Example 6: Disable resume to start fresh"
echo "------------------------------------------------------------"
cat << 'EOF'
python inference_pipeline.py \
    --infile data/totaltext/anno.json \
    --outfile output/totaltext_fresh \
    --img_dir data/totaltext/16 \
    --model_type gemini \
    --model_path gemini-1.5-flash \
    --hshift 3.4722 \
    --vshift 0.288 \
    --filter_id 2 \
    --no_resume
EOF

echo ""
echo "============================================================"
echo "Helper: Convert Filter ID to HShift/VShift"
echo "============================================================"
echo ""
echo "You can use Python to convert filter IDs to shift values:"
cat << 'EOF'

python -c "
from utils.image_processor import ImageFilterProcessor

filter_id = 2
hshift, vshift = ImageFilterProcessor.convert_filter_id_to_shifts(filter_id)
print(f'Filter ID {filter_id}: hshift={hshift}, vshift={vshift}')
"
EOF

echo ""
echo "============================================================"
echo "Note: These are example commands. Please modify paths and"
echo "parameters according to your specific needs."
echo "============================================================"




