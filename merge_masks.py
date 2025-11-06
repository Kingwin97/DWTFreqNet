#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merge multiple defect masks into unified masks per image.

This script combines multiple mask files that belong to the same image into a single
unified mask file. The combining is done using pixel-wise bitwise_or operation.

Example:
    Input masks: 
        - 20251008新数据-磷化倒角_102011703_白茬_3_mask.png
        - 20251008新数据-磷化倒角_102011703_白茬_5_mask.png
    Output mask:
        - 20251008新数据-磷化倒角_102011703.png
"""

import cv2
import os
import argparse
import re
from collections import defaultdict
from pathlib import Path
import sys


def extract_image_id(mask_filename):
    """
    Extract the base image ID from a mask filename.
    
    The function removes the mask suffix and any numbering pattern to get the base image ID.
    For example:
        '20251008新数据-磷化倒角_102011703_白茬_3_mask.png' -> '20251008新数据-磷化倒角_102011703'
        
    Args:
        mask_filename (str): The mask filename
        
    Returns:
        str: The extracted image ID
    """
    # Remove file extension
    base_name = os.path.splitext(mask_filename)[0]
    
    # Remove '_mask' suffix if present
    if base_name.endswith('_mask'):
        base_name = base_name[:-5]
    
    # Pattern to match the numbering suffix (e.g., _白茬_3, _白茬_5, etc.)
    # This will match patterns like: _[text]_[number] at the end
    pattern = r'_[^_]+_\d+$'
    match = re.search(pattern, base_name)
    
    if match:
        # Remove the matched suffix to get the base image ID
        image_id = base_name[:match.start()]
    else:
        # If no pattern matches, use the whole base_name
        image_id = base_name
    
    return image_id


def merge_masks(input_folder, output_folder):
    """
    Merge multiple mask files into unified masks per image.
    
    Args:
        input_folder (str): Path to folder containing mask files
        output_folder (str): Path to folder where merged masks will be saved
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        sys.exit(1)
    
    # Get all mask files
    mask_files = [f for f in os.listdir(input_folder) 
                  if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not mask_files:
        print(f"Warning: No image files found in '{input_folder}'.")
        return
    
    print(f"Found {len(mask_files)} mask files in input folder.")
    
    # Group masks by image ID
    masks_by_image = defaultdict(list)
    for mask_file in mask_files:
        image_id = extract_image_id(mask_file)
        masks_by_image[image_id].append(mask_file)
    
    print(f"Grouped into {len(masks_by_image)} unique images.")
    
    # Process each image group
    processed_count = 0
    for image_id, mask_list in masks_by_image.items():
        print(f"\nProcessing image: {image_id}")
        print(f"  Merging {len(mask_list)} masks: {mask_list}")
        
        # Initialize merged mask with first mask
        first_mask_path = os.path.join(input_folder, mask_list[0])
        merged_mask = cv2.imread(first_mask_path, cv2.IMREAD_GRAYSCALE)
        
        if merged_mask is None:
            print(f"  Error: Could not read mask file '{mask_list[0]}'")
            continue
        
        # Merge remaining masks using bitwise_or
        for mask_file in mask_list[1:]:
            mask_path = os.path.join(input_folder, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                print(f"  Warning: Could not read mask file '{mask_file}', skipping...")
                continue
            
            # Check if dimensions match
            if mask.shape != merged_mask.shape:
                print(f"  Warning: Mask '{mask_file}' has different dimensions ({mask.shape}) "
                      f"than the first mask ({merged_mask.shape}), skipping...")
                continue
            
            # Combine using bitwise OR
            merged_mask = cv2.bitwise_or(merged_mask, mask)
        
        # Determine output file extension (use same as input)
        input_ext = os.path.splitext(mask_list[0])[1]
        output_filename = f"{image_id}{input_ext}"
        output_path = os.path.join(output_folder, output_filename)
        
        # Save merged mask
        success = cv2.imwrite(output_path, merged_mask)
        
        if success:
            print(f"  Saved merged mask to: {output_filename}")
            processed_count += 1
        else:
            print(f"  Error: Failed to save merged mask to '{output_path}'")
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successfully merged {processed_count} image masks.")
    print(f"Output saved to: {output_folder}")
    print(f"{'='*60}")


def main():
    """Main function to parse arguments and execute mask merging."""
    parser = argparse.ArgumentParser(
        description='Merge multiple defect masks into unified masks per image.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python merge_masks.py --input "D:\\qw\\Code_github\\cicai_all\\cicaiquexian_seg\\train\\masks" \\
                        --output "D:\\qw\\Code_github\\cicai_all\\merged_masks_images"
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default=r'D:\qw\Code_github\cicai_all\cicaiquexian_seg\train\masks',
        help='Input folder containing mask files (Note: Default path is for Windows. Adjust for your system.)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=r'D:\qw\Code_github\cicai_all\merged_masks_images',
        help='Output folder for merged mask files (Note: Default path is for Windows. Adjust for your system.)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Mask Merging Utility")
    print("="*60)
    print(f"Input folder:  {args.input}")
    print(f"Output folder: {args.output}")
    print("="*60)
    
    merge_masks(args.input, args.output)


if __name__ == '__main__':
    main()
