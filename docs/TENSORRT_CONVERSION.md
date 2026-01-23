# TensorRT Conversion Guide for YOLOE-26 on Orin-Nano

This guide explains how to convert the YOLOE-26 PyTorch model to TensorRT format on the Jetson Orin-Nano to achieve 2-5x faster inference and reduced GPU memory usage.

## Prerequisites

1. **Model Location**: Ensure `yoloe-26n-seg.pt` is in `/home/nudro/Documents/snowcrash-orin3/models/`
2. **Environment**: Run this on the Orin-Nano (TensorRT conversion is hardware-specific)
3. **Dependencies**: Ensure `ultralytics` is installed:
   ```bash
   pip install ultralytics
   ```

## Quick Start

### Step 1: Convert to TensorRT

Run the conversion script on the Orin-Nano:

```bash
cd /home/nudro/Documents/snowcrash-orin3
python scripts/convert_yolo_to_tensorrt.py models/yoloe-26n-seg.pt --precision fp16 --imgsz 1280
```

**Options:**
- `--precision fp16`: Recommended for best speed/accuracy balance (default)
- `--precision int8`: Maximum speed with slight accuracy loss
- `--imgsz 1280`: Input image size (must match your usage - 1280 is recommended for YOLOE-26)

### Step 2: Verify Conversion

After conversion, you should see:
- `models/yoloe-26n-seg.engine` file created
- Success message with file size

### Step 3: Automatic Detection

The code **automatically detects** TensorRT engines! No code changes needed:
- If `yoloe-26n-seg.engine` exists, it will be used automatically
- If not, it falls back to the PyTorch `.pt` file
- The `load_yolo_model()` function in `tools/yolo_utils.py` handles this

## Performance Comparison

| Format | Inference Speed | GPU Memory | Accuracy |
|--------|----------------|------------|----------|
| PyTorch (.pt) | Baseline | Higher | Full |
| TensorRT FP16 | 2-3x faster | Lower | ~99.9% |
| TensorRT INT8 | 3-5x faster | Lowest | ~99% |

## Detailed Usage

### Basic Conversion (FP16 - Recommended)

```bash
python scripts/convert_yolo_to_tensorrt.py models/yoloe-26n-seg.pt
```

This creates `models/yoloe-26n-seg.engine` with FP16 precision.

### Maximum Speed (INT8)

For maximum speed with slight accuracy trade-off:

```bash
python scripts/convert_yolo_to_tensorrt.py models/yoloe-26n-seg.pt --precision int8
```

### Custom Image Size

If you need a different input size:

```bash
python scripts/convert_yolo_to_tensorrt.py models/yoloe-26n-seg.pt --imgsz 640
```

**Note**: The image size must match what you use in your code. The default is 1280 for YOLOE-26.

### Custom Output Path

```bash
python scripts/convert_yolo_to_tensorrt.py models/yoloe-26n-seg.pt --output models/yoloe-26n-seg-custom.engine
```

## How It Works

1. **Auto-Detection**: The `load_yolo_model()` function in `tools/yolo_utils.py` checks for `.engine` file first
2. **Fallback**: If no `.engine` file exists, it uses the `.pt` file
3. **Image Size**: TensorRT engines are compiled for a specific size (default: 1280x1280)
   - The code automatically uses `imgsz=1280` when TensorRT engine is detected
   - PyTorch models use `imgsz=640` for memory efficiency

## Troubleshooting

### Conversion Takes Too Long

- First-time conversion can take 10-30 minutes (normal)
- TensorRT optimizes the model for your specific hardware
- Subsequent conversions are faster if using the same settings

### Out of Memory During Conversion

- Reduce workspace size in the script (currently 4GB)
- Close other applications
- Use INT8 precision instead of FP16

### Model Not Found

Ensure the model path is correct:
```bash
ls -lh models/yoloe-26n-seg.pt
```

### Engine File Not Created

- Check for errors in the conversion output
- Ensure you have write permissions in the models directory
- Check disk space: `df -h`

## Verification

After conversion, verify the engine file exists:

```bash
ls -lh models/yoloe-26n-seg.engine
```

You should see a file size around 20-50 MB (smaller than the .pt file due to optimization).

## Running with TensorRT

Once the `.engine` file exists, your code will automatically use it:

```python
# This automatically uses .engine if available, otherwise .pt
from tools.yolo_utils import load_yolo_model
model = load_yolo_model("models/yoloe-26n-seg.pt", verbose=True)
# Output: [YOLO] Loading TensorRT engine: models/yoloe-26n-seg.engine
```

## Performance Tips

1. **Use FP16**: Best balance of speed and accuracy
2. **Match Image Size**: Use `--imgsz 1280` to match YOLOE-26's optimal size
3. **Batch Processing**: TensorRT excels at batch inference (if you process multiple frames)
4. **Monitor GPU**: Use `tegrastats` to monitor GPU usage and memory

## Next Steps

1. Convert the model using the steps above
2. Test your application - it should automatically use the TensorRT engine
3. Monitor performance improvements (2-5x speedup expected)
4. If needed, convert other YOLOE models (yoloe-26s-seg.pt, etc.) using the same process

## Notes

- TensorRT engines are **hardware-specific** - must be created on the Orin-Nano
- Engines are **size-specific** - if you change `imgsz`, you need to reconvert
- The `.pt` file is still needed as a fallback
- Both files can coexist - the code automatically chooses the best one

