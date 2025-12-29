# Test Directory

This directory contains test scripts for the World4Omni project.

## Test Scripts

### test_world_model.py
Main test script for the World Model pipeline with timing analysis.

**Features:**
- Tests full World Model pipeline (enhancer + reflector + masks)
- Displays detailed timing information
- Generates organized mask outputs
- Cleans up intermediate files

**Usage:**
```bash
python test/test_world_model.py
```

### test_world_model_modes.py
Comprehensive test script for different World Model configurations.

**Features:**
- Tests 4 different modes:
  - Full mode (enhancer + reflector + masks)
  - Simple mode (no enhancer, no reflector, no masks)
  - Enhanced only (enhancer + no reflector + no masks)
  - Reflector only (no enhancer + reflector + no masks)
- Performance comparison with timing analysis
- Detailed timing breakdown for each test mode

**Usage:**
```bash
python test/test_world_model_modes.py
```

## Timing Information

Both test scripts provide detailed timing information including:
- Individual process timing
- Total execution time
- Minutes and seconds breakdown
- Performance comparison between different modes

## Requirements

- Set `GEMINI_API_KEY` environment variable
- Ensure all dependencies are installed
- Run from the project root directory
