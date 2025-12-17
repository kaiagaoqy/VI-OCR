# VIOCR
 Visually-impaird OCR framework featuring automatic low-vision filtering based on human vision parameters.

## Updates

- **6 Vision Models Supported**: DeepSeek OCR, Qwen2.5-VL, Gemini, Claude, GPT-4, CogVLM
- **Automatic Low-Vision Filtering**: Apply perceptual filters based on VA (Visual Acuity) and CS (Contrast Sensitivity)
- **Subject-Specific Processing**: Each subject has unique vision parameters (hshift/vshift)
- **Shared Filtered Images**: Efficient storage - filter once, reuse across all models
- **Batch Processing**: Process multiple subjects and images automatically
- **Resume Support**: Continue processing after interruptions
- **Flexible Workflows**: Pre-filtered or real-time filtering modes

## Project Structure

```
VIOCR_infer_models/
├── models/                          # Model implementations
│   ├── __init__.py                 # ModelFactory for lazy loading
│   ├── base_model.py               # Abstract base class
│   ├── dsocr_model.py              # DeepSeek OCR wrapper
│   ├── qwen2_model.py              # Qwen2.5-VL wrapper
│   ├── gemini_model.py             # Google Gemini wrapper
│   ├── claude_model.py             # Anthropic Claude wrapper
│   ├── gpt_model.py                # OpenAI GPT wrapper
│   └── cogvlm_model.py             # CogVLM wrapper
│
├── utils/                           # Utility modules
│   ├── filter.py                   # Core filtering algorithms (original)
│   ├── image_processor.py          # Image filtering wrapper
│   └── vision_parameter_lookup.py  # Subject vision parameter management
│
├── examples/                        # Usage examples
│   ├── example_usage.py            # Python API examples
│   └── example_command_line.sh     # CLI examples
│
├── data/                            # Data directory
│   └── human/                      
│       ├── param.matrix.csv        # VA/CS to hshift/vshift mapping
│       ├── human_measured_vision_cleaned.csv  # Subject measurements
│       └── sub_vision_rec_to_hvshift.csv      # Generated parameters
│
├── outputs/                         # Output directory
│   ├── filtered_images/            # Shared filtered images (all models)
│   ├── dsocr_results/              # DeepSeek results
│   ├── qwen2_results/              # Qwen2 results
│   ├── claude_results/             # Claude results
│   └── ...                         # Other model results
│
├── inference_pipeline.py            # Main inference orchestration
├── batch_process_subjects.py        # Batch processing script
└── BATCH_PROCESS.sh         # Interactive workflow script
```

## Quick Start

### Installation

```bash
# Clone repository
cd /path/to/VIOCR_infer_models

# Install core dependencies
pip install opencv-python pillow numpy pandas tqdm python-dotenv

# Install model-specific dependencies (in separate environments)
# For DeepSeek OCR:
pip install deepseek-ocr

# For Qwen2:
pip install transformers torch

# For Gemini:
pip install google-generativeai

# For Claude:
pip install anthropic

# For GPT-4:
pip install openai

# For CogVLM:
pip install transformers torch
```

### Environment Setup

Create a `.env` file:

```bash
# Prompt for inference
Prompt="What are all the English words visible in the image? Please output each word in order from left to right and top to bottom. If you can not recognize it, please leave it blank. Please return strictly in the following format (do not output anything else): ['word1', 'word2', 'word3', ...] Note that the data such as 'word1' above should be replaced with a real word."
# word with grounding box
Prompt="What are all the English words visible in the image? Please output each word in order from left to right and top to bottom and include the coordinates [[x0,y0,x1,y1]] for each word. If you can not recognize it, please leave it blank. Please return strictly in the following format (do not output anything else): {'word1':[x0,y0,x1,y1], 'word2':[x0,y0,x1,y1], 'word3':[x0,y0,x1,y1], ...} Note that the data such as 'word1' above should be replaced with a real word."

# API Keys (if using API-based models)
ANTHROPIC_API_KEY="your-anthropic-key"
OPENAI_API_KEY="your-openai-key"
GOOGLE_API_KEY="your-google-key"
```

### Generate Subject Parameters

```bash
# Generate hshift/vshift for all subjects
python utils/vision_parameter_lookup.py --export data/human/sub_vision_rec_to_hvshift.csv
```

### Run Interactive Workflow

```bash
./BATCH_PROCESS.sh
```

Follow the prompts to:
1. Generate/verify subject parameters
2. Choose processing mode (pre-filtered or real-time)
3. Select model (1-6)
4. Execute batch processing

## Complete Paramter Inference

```bash
python batch_process_subjects.py \
    # Required
    --model_type {dsocr,qwen2,gemini,claude,gpt,cogvlm} \
    --model_path MODEL_PATH \
    --output_dir OUTPUT_DIR \
    
    # Mode 1: Pre-filtered images
    --filtered_image_dir FILTERED_DIR \
    
    # Mode 2: Real-time filtering
    --anno_json ANNO_JSON \
    --img_dir IMG_DIR \
    
    # Optional
    --subject_csv data/human/sub_vision_rec_to_hvshift.csv \
    --shared_filtered_dir outputs/filtered_images \
    --save_filtered_images \
    --prompt "Custom prompt" \
    --size {Tiny,Small,Base,Large,Gundam}  # DeepSeek only
    --no_resume \
    
    # Advanced filter options
    --screen_reso WIDTH HEIGHT \
    --screen_size WIDTH_CM HEIGHT_CM \
    --camera \
    --white_balance
```

## Core Concepts

### Vision Parameters

Each subject has unique vision impairment characteristics:

- **VA (Visual Acuity)**: Measured in logMAR (e.g., 1.0, 1.3, 2.9)
- **CS (Contrast Sensitivity)**: Measured in logCS (e.g., 0.85, 1.25, 0.35)
- **hshift**: Horizontal blur parameter (derived from VA/CS)
- **vshift**: Vertical blur parameter (derived from VA/CS)

### Filtering Process

```
Original Image → Apply Subject's Filter (hshift, vshift) → Filtered Image → Model Inference
```

**Key Point**: All subjects process the **same** set of original images, but each uses their unique filter parameters.

### Two Processing Modes

#### Mode 1: Pre-filtered Images (Fast)

Use pre-saved filtered images for direct inference.

```bash
python batch_process_subjects.py \
    --model_type dsocr \
    --model_path deepseek-ai/DeepSeek-OCR \
    --output_dir outputs/dsocr_results \
    --filtered_image_dir outputs/filtered_images
```

**Use when**: Testing multiple models, want fastest processing.

#### Mode 2: Real-time Filtering (Flexible)

Apply filters on-the-fly during inference.

```bash
python batch_process_subjects.py \
    --model_type dsocr \
    --model_path deepseek-ai/DeepSeek-OCR \
    --output_dir outputs/dsocr_results \
    --anno_json data/totaltext/anno.json \
    --img_dir data/totaltext/16 \
    --save_filtered_images  # Optional: save for future use
```

**Use when**: First run, want to save filtered images for reuse.

## Supported Models

| Model | Type | Model Path | API/Local | Environment |
|-------|------|------------|-----------|-------------|
| **DeepSeek OCR** | `dsocr` | `deepseek-ai/DeepSeek-OCR` | Local | `dsocr_env` |
| **Qwen2.5-VL** | `qwen2` | `Qwen/Qwen2.5-VL-7B-Instruct` | Local | `qwen2_env` |
| **Gemini** | `gemini` | `gemini-2.5-flash` | API | `gemini_env` |
| **Claude** | `claude` | `claude-3-5-sonnet-20240620` | API | `claude_env` |
| **GPT-4** | `gpt` | `gpt-4o` | API | `gpt_env` |
| **CogVLM** | `cogvlm` | `THUDM/cogagent-chat-hf` | Local | `cogvlm_env` |

### Model-Specific Options

**DeepSeek OCR**:
```bash
--size {Tiny,Small,Base,Large,Gundam}
```

**CogVLM**:
```bash
--quant 4  # 4-bit quantization
--bf16     # Use bfloat16
```

## Usage Examples
More examples can be found in examples/

### Example 1: Single Model, Real-time Filtering

```bash
python batch_process_subjects.py \
    --model_type dsocr \
    --model_path deepseek-ai/DeepSeek-OCR \
    --output_dir outputs/dsocr_results \
    --anno_json data/totaltext/anno.json \
    --img_dir data/totaltext/16 \
    --size Tiny \
    --save_filtered_images
```

## 📊 Output Format

### Directory Structure

```
outputs/
├── filtered_images/              # Shared across all models
│   ├── Sub123/
│   │   ├── img_001.jpg
│   │   └── img_002.jpg
│   └── Sub150/
│       ├── img_001.jpg
│       └── img_002.jpg
│
├── dsocr_results/
│   ├── all_results.jsonl        # Stream format (one JSON per line)
│   ├── all_results.json         # Array format
│   ├── Sub123/
│   │   └── results.json         # All images for Sub123
│   └── Sub150/
│       └── results.json
│
└── claude_results/
    ├── all_results.jsonl
    ├── all_results.json
    ├── Sub123/
    │   └── results.json
    └── Sub150/
        └── results.json
```

### Result JSON Format

**Per-subject results** (`SubjectID/results.json`):
```json
[
  {
    "subject_id": "Sub123",
    "image_id": "1",
    "image_filename": "img_001.jpg",
    "va_measured": 1.0,
    "cs_measured": 0.85,
    "hshift": 7.0771,
    "vshift": 0.0692,
    "rec_texts": "recognized text here",
    "mode": "real-time-filter"
  },
  {
    "subject_id": "Sub123",
    "image_id": "2",
    "image_filename": "img_002.jpg",
    ...
  }
]
```

## Advanced Usage

### Custom Prompts

```bash
python batch_process_subjects.py \
    --model_type gpt \
    --prompt "List all visible words in the image" \
    ...
```

### Custom Filter Parameters

```bash
python batch_process_subjects.py \
    --screen_reso 1920 1080 \
    --screen_size 52.7 29.6 \
    --camera \
    --white_balance \
    ...
```


## Contributing

To add a new model:

1. Create a new model class in `models/`:
```python
from .base_model import BaseModelInference

class YourModelInference(BaseModelInference):
    def load_model(self):
        # Load your model
        pass
    
    def infer(self, image: str, prompt: str, **kwargs) -> str:
        # Inference logic
        pass
```

2. Register in `models/__init__.py`:
```python
MODEL_MAP = {
    ...
    'yourmodel': ('yourmodel_model', 'YourModelInference'),
}
```

3. Add to `BATCH_PROCESS_EXAMPLE.sh`:
```bash
7)
    MODEL_TYPE="yourmodel"
    MODEL_PATH="your/model/path"
    ENV_NAME="yourmodel_env"
    ;;
```


## Cite Us
```bibtex
@article{gao2025vi,
  title={VI-OCR:“Visually Impaired” optical character recognition pipeline for text accessibility assessment},
  author={Gao, Qingying and Manduchi, Roberto and Ramulu, Pradeep Y and Legge, Gordon E and Xiong, Yingzi},
  journal={Scientific Reports},
  year={2025},
  publisher={Nature Publishing Group}
}
```


