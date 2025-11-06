"""
Script to merge multiple mask files for the same base image into a single mask.

This script processes mask files with names like:
    '20251008新数据-大面龟裂_42586468_裂纹_1_mask.png'
and merges them into a single mask file named after the base image:
    '20251008新数据-大面龟裂_42586468.png'

Multiple masks for the same image are combined using pixel-wise OR operations.
"""

import os
import re
from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict


def extract_base_name(filename):
    """
    Extract the base image name from a mask filename.
    
    Examples:
        '20251008新数据-大面龟裂_42586468_裂纹_1_mask.png' -> '20251008新数据-大面龟裂_42586468'
        '20251008新数据-大面龟裂_42586468_破损_2_mask.png' -> '20251008新数据-大面龟裂_42586468'
    
    Args:
        filename (str): The mask filename
    
    Returns:
        str: The base image name without extension
    """
    # Remove the file extension
    name_without_ext = os.path.splitext(filename)[0]
    
    # Pattern to match the base name before the defect type suffix
    # The pattern matches everything up to the last underscore followed by a defect type
    # and optional number/mask suffix
    pattern = r'^(.+?)_[^_]+_\d+_mask$'
    match = re.match(pattern, name_without_ext)
    
    if match:
        return match.group(1)
    
    # If the pattern doesn't match, try a simpler pattern
    # that removes '_mask' suffix and any trailing numbers
    pattern2 = r'^(.+?)(?:_[^_]+)?_mask$'
    match2 = re.match(pattern2, name_without_ext)
    
    if match2:
        return match2.group(1)
    
    # Fallback: remove '_mask' suffix if present
    if name_without_ext.endswith('_mask'):
        return name_without_ext[:-5]
    
    # If no pattern matches, return the filename without extension
    return name_without_ext


def group_masks_by_base_image(mask_folder):
    """
    Group mask files by their base image name.
    
    Args:
        mask_folder (str or Path): Path to the folder containing mask files
    
    Returns:
        dict: Dictionary mapping base image names to lists of mask file paths
    """
    mask_folder = Path(mask_folder)
    masks_by_base = defaultdict(list)
    
    # Get all PNG files in the folder
    mask_files = list(mask_folder.glob('*.png'))
    
    for mask_file in mask_files:
        base_name = extract_base_name(mask_file.name)
        masks_by_base[base_name].append(mask_file)
    
    return masks_by_base


def merge_masks(mask_paths):
    """
    Merge multiple mask images using pixel-wise OR operation.
    
    Args:
        mask_paths (list): List of paths to mask images to merge
    
    Returns:
        numpy.ndarray: Merged mask as a grayscale numpy array
    """
    if not mask_paths:
        return None
    
    # Load the first mask to get dimensions
    first_mask = Image.open(mask_paths[0]).convert('L')
    merged_mask = np.array(first_mask, dtype=np.uint8)
    
    # Merge remaining masks using pixel-wise OR
    for mask_path in mask_paths[1:]:
        mask = Image.open(mask_path).convert('L')
        mask_array = np.array(mask, dtype=np.uint8)
        
        # Ensure dimensions match
        if mask_array.shape != merged_mask.shape:
            print(f"Warning: Mask {mask_path.name} has different dimensions. Skipping.")
            continue
        
        # Pixel-wise OR operation
        merged_mask = np.maximum(merged_mask, mask_array)
    
    return merged_mask


def process_masks(input_folder, output_folder):
    """
    Process all mask files in the input folder and save merged masks to output folder.
    
    Args:
        input_folder (str or Path): Path to folder containing mask files
        output_folder (str or Path): Path to folder where merged masks will be saved
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    
    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Group masks by base image
    masks_by_base = group_masks_by_base_image(input_folder)
    
    print(f"Found {len(masks_by_base)} unique base images")
    print(f"Processing {sum(len(v) for v in masks_by_base.values())} mask files")
    
    # Process each group
    for base_name, mask_paths in masks_by_base.items():
        print(f"Merging {len(mask_paths)} masks for '{base_name}'")
        
        # Merge the masks
        merged_mask = merge_masks(mask_paths)
        
        if merged_mask is not None:
            # Save the merged mask
            output_path = output_folder / f"{base_name}.png"
            merged_image = Image.fromarray(merged_mask, mode='L')
            merged_image.save(output_path)
            print(f"  Saved: {output_path.name}")
    
    print(f"\nProcessing complete! Merged masks saved to: {output_folder}")


def main():
    """
    Main function to run the mask merging process.
    """
    # Default paths from the problem statement
    input_folder = r'D:\qw\Code_github\cicai_all\cicaiquexian_seg\train\masks'
    output_folder = r'D:\qw\Code_github\cicai_all\merged_masks_images'
    
    # Check if running in a different environment
    if not os.path.exists(input_folder):
        print(f"Warning: Input folder not found: {input_folder}")
        print("Please provide the correct paths as command line arguments.")
        print("\nUsage: python merge_masks.py <input_folder> <output_folder>")
        
        # Try to use command line arguments if provided
        import sys
        if len(sys.argv) >= 3:
            input_folder = sys.argv[1]
            output_folder = sys.argv[2]
            print(f"Using provided paths:")
            print(f"  Input: {input_folder}")
            print(f"  Output: {output_folder}")
        else:
            print("\nNo command line arguments provided. Exiting.")
            return
    
    # Process the masks
    process_masks(input_folder, output_folder)


if __name__ == '__main__':
    main()
