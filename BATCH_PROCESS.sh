#!/bin/bash

# Batch Process Subjects - Complete Example Workflow
# This script demonstrates the complete workflow for processing all subjects

echo "========================================================================"
echo "Batch Process Subjects - Complete Workflow Example"
echo "========================================================================"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Generate subject parameters
echo -e "${BLUE}Step 1: Generate Subject Parameters${NC}"
echo "------------------------------------------------------------------------"
echo "Command: python utils/vision_parameter_lookup.py --export data/human/sub_vision_rec_to_hvshift.csv"
echo ""

read -p "Execute Step 1? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python utils/vision_parameter_lookup.py --export data/human/sub_vision_rec_to_hvshift.csv
    echo ""
fi

# Check if CSV exists
if [ ! -f "data/human/sub_vision_rec_to_hvshift.csv" ]; then
    echo -e "${YELLOW}⚠️  Parameter file not found. Please run Step 1 first.${NC}"
    exit 1
fi

# Step 2: Preview parameters
echo -e "${BLUE}Step 2: Preview Parameter File${NC}"
echo "------------------------------------------------------------------------"
echo "First 5 rows:"
head -5 data/human/sub_vision_rec_to_hvshift.csv
echo ""
echo "Total rows: $(wc -l < data/human/sub_vision_rec_to_hvshift.csv) (including header)"
echo ""

# Step 3: Choose processing mode
echo -e "${BLUE}Step 3: Choose Processing Mode${NC}"
echo "------------------------------------------------------------------------"
echo "IMPORTANT: All subjects will process the SAME set of images."
echo "           Only the filter parameters (hshift/vshift) differ per subject."
echo ""
echo "Available modes:"
echo "  1. Mode 1: Use pre-saved filtered images (Fast)"
echo "     - Each subject's filtered images are already saved"
echo "     - Structure: filtered_dir/SubjectID/images/"
echo ""
echo "  2. Mode 2: Real-time filtering (Flexible)"
echo "     - All subjects use same original images from anno.json"
echo "     - Each subject's filter is applied on-the-fly"
echo "     - Optional: Save filtered images for future use"
echo ""

read -p "Choose mode (1-2): " mode_choice

case $mode_choice in
    1)
        MODE="prefiltered"
        MODE_FLAG="--filtered_image_dir"
        echo ""
        echo -e "${GREEN}Selected: Mode 1 - Using pre-saved filtered images${NC}"
        echo ""
        read -p "Enter filtered images base directory: " FILTERED_DIR
        
        if [ ! -d "$FILTERED_DIR" ]; then
            echo -e "${RED}✗ Directory not found: $FILTERED_DIR${NC}"
            exit 1
        fi
        
        echo ""
        echo "Expected structure:"
        echo "  $FILTERED_DIR/"
        echo "  ├── Sub123/"
        echo "  │   ├── img1.jpg"
        echo "  │   └── img2.jpg"
        echo "  └── Sub150/"
        echo "      ├── img1.jpg"
        echo "      └── img2.jpg"
        echo ""
        
        MODE_PARAM="$MODE_FLAG $FILTERED_DIR"
        SAVE_IMAGES_FLAG=""
        ;;
    2)
        MODE="realtime"
        echo ""
        echo -e "${GREEN}Selected: Mode 2 - Real-time filtering${NC}"
        echo ""
        echo "All subjects will use the SAME original images."
        echo "Filter parameters will be applied per subject."
        echo ""
        echo "Default settings:"
        echo "  - anno.json: data/totaltext/anno.json"
        echo "  - img_dir:   data/totaltext/16"
        echo ""
        
        read -p "Use default settings? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            ANNO_JSON="data/totaltext/anno.json"
            IMG_DIR="data/totaltext/16"
        else
            read -p "Enter anno.json path: " ANNO_JSON
            read -p "Enter image directory: " IMG_DIR
        fi
        
        MODE_PARAM="--anno_json $ANNO_JSON --img_dir $IMG_DIR"
        
        echo ""
        read -p "Save filtered images for future use? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            SAVE_IMAGES_FLAG="--save_filtered_images"
            echo ""
            echo "Filtered images will be saved to: <output_dir>/filtered_images/SubjectID/"
        else
            SAVE_IMAGES_FLAG=""
        fi
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

# Step 4: Choose model
echo ""
echo -e "${BLUE}Step 4: Choose Model and Environment${NC}"
echo "------------------------------------------------------------------------"
echo "Available models:"
echo "  1. DeepSeek OCR (dsocr_env)"
echo "  2. Qwen2.5-VL (qwen2_env)"
echo "  3. Google Gemini (gemini_env)"
echo "  4. Claude (claude_env)"
echo "  5. GPT-4 (gpt_env)"
echo "  6. CogVLM (cogvlm_env)"
echo ""

read -p "Choose model (1-6): " model_choice

case $model_choice in
    1)
        MODEL_TYPE="dsocr"
        MODEL_PATH="deepseek-ai/DeepSeek-OCR"
        ENV_NAME="dsocr_env"
        MODEL_SIZE="--size Tiny"
        ;;
    2)
        MODEL_TYPE="qwen2"
        MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
        ENV_NAME="qwen2_env"
        MODEL_SIZE=""
        ;;
    3)
        MODEL_TYPE="gemini"
        MODEL_PATH="gemini-2.5-flash"
        ENV_NAME="gemini_env"
        MODEL_SIZE=""
        ;;
    4)
        MODEL_TYPE="claude"
        MODEL_PATH="claude-3-5-sonnet-20240620"
        ENV_NAME="claude_env"
        MODEL_SIZE=""
        ;;
    5)
        MODEL_TYPE="gpt"
        MODEL_PATH="gpt-4o"
        ENV_NAME="gpt_env"
        MODEL_SIZE=""
        ;;
    6)
        MODEL_TYPE="cogvlm"
        MODEL_PATH="THUDM/cogagent-chat-hf"
        ENV_NAME="cogvlm_env"
        MODEL_SIZE=""
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Selected model: $MODEL_TYPE${NC}"
echo "Model path: $MODEL_PATH"
echo "Recommended environment: $ENV_NAME"
echo ""

# Step 5: Configure output
echo -e "${BLUE}Step 5: Configure Output Directory${NC}"
echo "------------------------------------------------------------------------"

OUTPUT_DIR="outputs/${MODEL_TYPE}_results"
SHARED_FILTERED_DIR="outputs/filtered_images"
echo "Output directory:          $OUTPUT_DIR"
echo "Shared filtered images:    $SHARED_FILTERED_DIR"
echo "  (Shared across all models to avoid duplication)"
echo ""

# Step 6: Build command
echo -e "${BLUE}Step 6: Build Command${NC}"
echo "------------------------------------------------------------------------"

COMMAND="python batch_process_subjects.py \
    --model_type $MODEL_TYPE \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --shared_filtered_dir $SHARED_FILTERED_DIR \
    $MODEL_SIZE \
    $MODE_PARAM \
    $SAVE_IMAGES_FLAG"

echo "Full command:"
echo ""
echo "$COMMAND"
echo ""

# Step 7: Display command summary
echo -e "${BLUE}Step 7: Command Summary${NC}"
echo "------------------------------------------------------------------------"
echo "Processing mode:     ${MODE}"
echo "Model:               ${MODEL_TYPE}"
echo "Output directory:    ${OUTPUT_DIR}"
if [ "$MODE" = "realtime" ] && [ -n "$SAVE_IMAGES_FLAG" ]; then
    echo "Filtered images:     ${SHARED_FILTERED_DIR}/"
    echo "  (Will be saved and shared across all models)"
fi
echo ""

# Step 8: Execute
echo -e "${BLUE}Step 8: Execute Batch Processing${NC}"
echo "------------------------------------------------------------------------"
echo ""
echo -e "${YELLOW}⚠️  Important: Make sure you have activated the correct virtual environment!${NC}"
echo "Example: conda activate $ENV_NAME"
echo ""

read -p "Execute batch processing now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Starting processing..."
    echo ""
    
    eval $COMMAND
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo -e "${GREEN}========================================================================"
        echo "✓ Batch processing completed successfully!"
        echo "========================================================================${NC}"
        echo ""
        echo "Results saved in: $OUTPUT_DIR"
        echo ""
        echo "View results:"
        echo "  - Summary (JSON):   cat $OUTPUT_DIR/all_results.json | jq"
        echo "  - Single subject:   cat $OUTPUT_DIR/Sub123/results.json | jq"
        echo "  - Count results:    cat $OUTPUT_DIR/all_results.json | jq 'length'"
        echo ""
        
        if [ "$MODE" = "realtime" ] && [ -n "$SAVE_IMAGES_FLAG" ]; then
            echo "Filtered images saved in: $SHARED_FILTERED_DIR/"
            echo "These images are shared across all models."
            echo "You can reuse them with Mode 1 (--filtered_image_dir $SHARED_FILTERED_DIR) for faster processing."
            echo ""
        fi
    else
        echo ""
        echo -e "${RED}✗ Error occurred during processing (exit code: $EXIT_CODE)${NC}"
        echo ""
    fi
else
    echo ""
    echo "Command generated but not executed."
    echo "You can copy the command above and run it manually."
    echo ""
fi

echo "========================================================================"
echo "Workflow Complete"
echo "========================================================================"

