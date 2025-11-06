# Mask Merging Utility

This utility script combines multiple defect mask files that belong to the same image into a single unified mask file using pixel-wise bitwise OR operation.

## Features

- **Automatic grouping**: Groups multiple masks by extracting the base image ID from filenames
- **Bitwise OR merging**: Combines masks using pixel-wise bitwise OR to preserve all defect regions
- **Multiple format support**: Supports PNG, JPG, JPEG, and BMP image formats
- **Grayscale output**: Saves merged masks as grayscale images
- **Extension preservation**: Output files use the same extension as input files
- **Error handling**: Robust error handling for missing files and dimension mismatches

## Installation

The script requires OpenCV (cv2) which can be installed via:

```bash
pip install opencv-python
```

## Usage

### Basic Usage

```bash
python merge_masks.py --input <input_folder> --output <output_folder>
```

### Example

```bash
python merge_masks.py --input "D:\qw\Code_github\cicai_all\cicaiquexian_seg\train\masks" \
                      --output "D:\qw\Code_github\cicai_all\merged_masks_images"
```

### Arguments

- `--input`: Path to the folder containing mask files to be merged (default: `D:\qw\Code_github\cicai_all\cicaiquexian_seg\train\masks`)
- `--output`: Path to the folder where merged masks will be saved (default: `D:\qw\Code_github\cicai_all\merged_masks_images`)

### Help

```bash
python merge_masks.py --help
```

## How It Works

### File Naming Convention

The script extracts the base image ID by removing the mask suffix and numbering pattern from filenames.

**Example:**

Input files:
- `20251008新数据-磷化倒角_102011703_白茬_3_mask.png`
- `20251008新数据-磷化倒角_102011703_白茬_5_mask.png`
- `20251008新数据-磷化倒角_102011703_白茬_7_mask.png`

Output file:
- `20251008新数据-磷化倒角_102011703.png`

The script identifies masks belonging to the same image by extracting the base ID (`20251008新数据-磷化倒角_102011703`) from the filename pattern.

### Merging Process

1. **Group masks**: All mask files in the input folder are grouped by their base image ID
2. **Initialize**: The first mask for each image is loaded as the base
3. **Combine**: Each subsequent mask is combined with the base using `cv2.bitwise_or()`
4. **Save**: The merged mask is saved with the base image ID as the filename

### Bitwise OR Operation

The bitwise OR operation ensures that any pixel that is white (255) in any of the masks will be white in the merged mask. This preserves all defect regions from all masks:

```
Mask 1:  [0, 0, 255, 0]
Mask 2:  [0, 255, 0, 0]
Result:  [0, 255, 255, 0]  (bitwise OR)
```

## Example Output

```
============================================================
Mask Merging Utility
============================================================
Input folder:  D:\qw\Code_github\cicai_all\cicaiquexian_seg\train\masks
Output folder: D:\qw\Code_github\cicai_all\merged_masks_images
============================================================
Found 5 mask files in input folder.
Grouped into 2 unique images.

Processing image: 20251008新数据-磷化倒角_102011703
  Merging 3 masks: ['..._3_mask.png', '..._5_mask.png', '..._7_mask.png']
  Saved merged mask to: 20251008新数据-磷化倒角_102011703.png

Processing image: 20251008新数据-磷化倒角_102011704
  Merging 2 masks: ['..._1_mask.png', '..._2_mask.png']
  Saved merged mask to: 20251008新数据-磷化倒角_102011704.png

============================================================
Processing complete!
Successfully merged 2 image masks.
Output saved to: D:\qw\Code_github\cicai_all\merged_masks_images
============================================================
```

## Error Handling

The script handles several error conditions:

- **Missing input folder**: Exits with an error message
- **No mask files found**: Warns the user and exits gracefully
- **Unreadable mask files**: Skips the file and continues processing
- **Dimension mismatches**: Skips masks with different dimensions and logs a warning
- **Write failures**: Reports failed save operations

## Testing

To test the script with sample data:

```bash
# Create test directories
mkdir -p /tmp/test_masks_input /tmp/test_masks_output

# Create sample mask files (using Python)
python3 << 'EOF'
import cv2
import numpy as np

mask1 = np.zeros((100, 100), dtype=np.uint8)
mask1[10:30, 10:30] = 255

mask2 = np.zeros((100, 100), dtype=np.uint8)
mask2[40:60, 40:60] = 255

cv2.imwrite('/tmp/test_masks_input/test_001_defect_1_mask.png', mask1)
cv2.imwrite('/tmp/test_masks_input/test_001_defect_2_mask.png', mask2)
EOF

# Run the merge script
python merge_masks.py --input /tmp/test_masks_input --output /tmp/test_masks_output
```

## Technical Details

- **Language**: Python 3
- **Dependencies**: OpenCV (cv2), numpy
- **Image format**: Grayscale (single channel)
- **Pixel depth**: 8-bit unsigned integer (0-255)

## Author

Created as part of the DWTFreqNet project for infrared small target detection.

## License

Follow the same license as the main DWTFreqNet project.
