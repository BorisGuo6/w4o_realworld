## Usage (Google GenAI Image Generation and Editing)

Set the environment variable `GEMINI_API_KEY` first.

**🚀 Automatic Model Selection**
All scripts automatically select the best available model:
- **Default**: `gemini-2.5-flash-image-preview` (high-performance)
- **Fallback**: `gemini-2.0-flash-preview-image-generation` (standard)

```bash
export GEMINI_API_KEY=YOUR_API_KEY
```

### Generate image: gen_img.py

**Function**: Generate images from text prompts using Google GenAI.

**Basic usage:**
```bash
python /home/boris/workspace/World4Omni/scripts/gen_img.py "a red circle on white background"
```

**With optional arguments:**
```bash
python /home/boris/workspace/World4Omni/scripts/gen_img.py \
  "a red circle on white background" \
  --prefix my_gen
```

**Arguments:**
- `prompt` (required): text instruction
- `--prefix` (optional): output filename prefix (default: `generated_image`)

**Output**: Images saved to `World4Omni/outputs/image_generation/gen_img/`

### Edit image: edit_img.py

**Function**: Edit images with text instructions using Google GenAI.

**Basic usage:**
```bash
python /home/boris/workspace/World4Omni/scripts/edit_img.py \
  /home/boris/workspace/World4Omni/images/move_tomato_to_pan.png \
  "revise the image, move the tomato to pan"
```

**With optional arguments:**
```bash
python /home/boris/workspace/World4Omni/scripts/edit_img.py \
  /path/to/input.png \
  "your edit instruction" \
  --prefix my_edit
```

**Arguments:**
- `image` (required): input image path
- `instruction` (required): text instruction
- `--prefix` (optional): output filename prefix (default: `edited_image`)

**Output**: Edited images saved to `World4Omni/outputs/image_generation/edit_img/`

### Enhance prompts: enhancer.py

**Function**: Convert natural language to precise image editing prompts.

**Basic usage:**
```bash
export GEMINI_API_KEY=YOUR_KEY
python /home/boris/workspace/World4Omni/scripts/enhancer.py "move the tomato to pan"
```

**Arguments:**
- `text` (required): input text instruction
- `--api-key` (optional): Gemini API key

**Output**: Enhanced prompt text

### Extract objects: extract_objects.py

**Function**: Extract moved objects from text instructions.

**Basic usage:**
```bash
export PYTHONNOUSERSITE=1
python /home/boris/workspace/World4Omni/scripts/extract_objects.py \
  "revise the image, move the tomato to pan"
```

**Arguments:**
- `text` (required): input text instruction
- `--json` (optional): output as JSON array

**Output**: Comma-separated object names or JSON array

### Run Grounded SAM: test_grounded_sam.py

**Function**: Segment objects in images using GroundingDINO + SAM2.

**Basic usage:**
```bash
export PYTHONNOUSERSITE=1
python /home/boris/workspace/World4Omni/scripts/test_grounded_sam.py \
  /home/boris/workspace/World4Omni/images/move_tomato_to_pan.png \
  "tomato, pan"
```

**Arguments:**
- `image` (required): input image path
- `items` (required): objects to segment
- `--output-dir` (optional): output directory
- `--box-threshold` (optional): box threshold (default: 0.35)
- `--text-threshold` (optional): text threshold (default: 0.25)

**Output**: Annotated images and JSON results

### Synthesize images: Synthesis.py

**Function**: End-to-end pipeline for image editing with object segmentation and overlay.

**Basic usage:**
```bash
export PYTHONNOUSERSITE=1
export GEMINI_API_KEY=YOUR_KEY
python /home/boris/workspace/World4Omni/scripts/Synthesis.py \
  /home/boris/workspace/World4Omni/images/move_tomato_to_pan.png \
  "revise the image, move the tomato to pan"
```

**Arguments:**
- `image` (required): input image path
- `instruction` (required): text instruction
- `--alpha` (optional): overlay transparency (0-1, default: 0.5)
- `--out-prefix` (optional): output prefix (default: "synthesis")

**Output**: Edited image, segmentation results, and final overlay

### Iterative editing: Reflector.py

**Function**: Advanced pipeline with AI validation and iterative improvement.

**Basic usage:**
```bash
export PYTHONNOUSERSITE=1
export GEMINI_API_KEY=YOUR_KEY
python /home/boris/workspace/World4Omni/scripts/Reflector.py \
  /home/boris/workspace/World4Omni/images/move_tomato_to_pan.png \
  "move the tomato to pan"
```

**Arguments:**
- `image` (required): input image path
- `instruction` (required): text instruction
- `--max-iterations` (optional): max attempts (default: 3)
- `--out-prefix` (optional): output prefix (default: "reflector")

**Output**: Iterative results with final validated overlay

### Test World Model: test/test_world_model.py

**Function**: Test the World Model pipeline with iterative validation, reflection, mask generation and object segmentation.

**Basic usage:**
```bash
export PYTHONNOUSERSITE=1
export GEMINI_API_KEY=YOUR_KEY
python /home/boris/workspace/World4Omni/test/test_world_model.py
```

**Features:**
- **Iterative Validation**: Uses Gemini to validate and refine results (up to 3 iterations)
- **Reflection Loop**: Automatically revises instructions based on validation feedback
- **Mask Generation**: Generates masks for both original and edited images
- **Object Extraction**: Extracts target objects from text instructions
- **Organized Output**: Saves results to `outputs/mask/` directory with timestamped subfolders
- **Cleanup**: Automatically cleans up intermediate files
- **⏱️ Timing**: Displays detailed timing information for each process

**Process Flow:**
1. **Enhance Instruction**: Convert natural language to precise editing prompts
2. **Iterative Synthesis**: Run image editing with validation loop
3. **Gemini Validation**: AI validates if result meets requirements
4. **Reflection**: If validation fails, generate revised instructions
5. **Mask Generation**: Create object masks for both original and edited images
6. **Cleanup**: Remove intermediate files, keep only final results

**Output**: 
- Organized mask directory with timestamped subfolders
- Each run creates exactly 4 files: original.png, edited.png, original_mask.png, edited_mask.png
- Intermediate files stored in outputs/reflector/
- **Timing Information**: Detailed breakdown of processing time

### Test Different Modes: test/test_world_model_modes.py

**Function**: Test different World Model configurations with timing analysis.

**Basic usage:**
```bash
export PYTHONNOUSERSITE=1
export GEMINI_API_KEY=YOUR_KEY
python /home/boris/workspace/World4Omni/test/test_world_model_modes.py
```

**Features:**
- **4 Test Modes**:
  - Full mode (enhancer + reflector + masks)
  - Simple mode (no enhancer, no reflector, no masks)
  - Enhanced only (enhancer + no reflector + no masks)
  - Reflector only (no enhancer + reflector + no masks)
- **⏱️ Timing**: Displays detailed timing information for each test mode
- **Performance Comparison**: Compare processing times across different configurations

**Output**: 
- Results for each test mode
- Comprehensive timing analysis
- Performance comparison between different configurations

### Model Usage

**Text Processing**: `gemini-2.5-pro` (all text tasks)
**Image Generation**: Auto-selects best available model
**Image Analysis**: `gemini-1.5-flash-latest` (fallback for validation)

### Output Structure

```
World4Omni/outputs/
├── image_generation/
│   ├── gen_img/          # Generated images
│   └── edit_img/         # Edited images
├── grounded_sam/
│   └── segmentation/     # SAM results
├── synthesis/            # Synthesis outputs
├── reflector/            # Reflector outputs & intermediate files
└── mask/                 # World Model mask outputs (organized)
    └── run_YYYYMMDD_HHMMSS/  # Timestamped subfolders
        ├── original.png      # Original image
        ├── edited.png        # Edited image
        ├── original_mask.png # Original mask
        └── edited_mask.png   # Edited mask
```

**File naming**: `{prefix}_{timestamp}_c{candidate:02d}_p{part:02d}.{ext}`