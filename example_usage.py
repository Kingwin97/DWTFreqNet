"""
Example usage script demonstrating how to use merge_masks.py

This script shows different ways to use the mask merging functionality.
"""

from merge_masks import process_masks, extract_base_name, group_masks_by_base_image
from pathlib import Path


def example_1_basic_usage():
    """
    Example 1: Basic usage with custom paths
    """
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Define input and output folders
    input_folder = '/path/to/masks'
    output_folder = '/path/to/merged_output'
    
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print("\nUsage:")
    print(f"  from merge_masks import process_masks")
    print(f"  process_masks('{input_folder}', '{output_folder}')")
    print()


def example_2_problem_statement_paths():
    """
    Example 2: Using the exact paths from the problem statement
    """
    print("=" * 60)
    print("Example 2: Problem Statement Paths")
    print("=" * 60)
    
    # Paths from the problem statement
    input_folder = r'D:\qw\Code_github\cicai_all\cicaiquexian_seg\train\masks'
    output_folder = r'D:\qw\Code_github\cicai_all\merged_masks_images'
    
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print("\nCommand line usage:")
    print(f'  python merge_masks.py "{input_folder}" "{output_folder}"')
    print("\nOr in Python:")
    print(f"  from merge_masks import process_masks")
    print(f"  process_masks(r'{input_folder}', r'{output_folder}')")
    print()


def example_3_filename_extraction():
    """
    Example 3: Demonstrate filename extraction
    """
    print("=" * 60)
    print("Example 3: Filename Extraction Examples")
    print("=" * 60)
    
    test_filenames = [
        '20251008新数据-大面龟裂_42586468_裂纹_1_mask.png',
        '20251008新数据-大面龟裂_42586468_破损_2_mask.png',
        '20251008新数据-大面龟裂_42586468_缺陷_3_mask.png',
    ]
    
    print("Input mask filenames:")
    for filename in test_filenames:
        print(f"  - {filename}")
    
    print("\nExtracted base name:")
    base_name = extract_base_name(test_filenames[0])
    print(f"  {base_name}")
    
    print("\nOutput filename:")
    print(f"  {base_name}.png")
    print()


def example_4_inspect_before_merge():
    """
    Example 4: Inspect mask groupings before merging
    """
    print("=" * 60)
    print("Example 4: Inspect Mask Groupings")
    print("=" * 60)
    
    print("To inspect how masks will be grouped before merging:")
    print()
    print("```python")
    print("from merge_masks import group_masks_by_base_image")
    print("from pathlib import Path")
    print()
    print("input_folder = Path('/path/to/masks')")
    print("groups = group_masks_by_base_image(input_folder)")
    print()
    print("for base_name, mask_files in groups.items():")
    print("    print(f'Base: {base_name}')")
    print("    print(f'  Number of masks: {len(mask_files)}')")
    print("    for mask_file in mask_files:")
    print("        print(f'    - {mask_file.name}')")
    print("```")
    print()


def example_5_custom_processing():
    """
    Example 5: Custom processing with selective merging
    """
    print("=" * 60)
    print("Example 5: Custom Processing")
    print("=" * 60)
    
    print("For advanced use cases where you need custom processing:")
    print()
    print("```python")
    print("from merge_masks import group_masks_by_base_image, merge_masks")
    print("from pathlib import Path")
    print("from PIL import Image")
    print()
    print("input_folder = Path('/path/to/masks')")
    print("output_folder = Path('/path/to/output')")
    print("output_folder.mkdir(parents=True, exist_ok=True)")
    print()
    print("# Group masks")
    print("groups = group_masks_by_base_image(input_folder)")
    print()
    print("# Process each group with custom logic")
    print("for base_name, mask_files in groups.items():")
    print("    # Custom filtering (e.g., only merge specific defect types)")
    print("    filtered_masks = [m for m in mask_files if '裂纹' in m.name]")
    print("    ")
    print("    if filtered_masks:")
    print("        merged = merge_masks(filtered_masks)")
    print("        output_path = output_folder / f'{base_name}_filtered.png'")
    print("        Image.fromarray(merged, mode='L').save(output_path)")
    print("```")
    print()


def example_6_batch_processing():
    """
    Example 6: Batch process multiple folders
    """
    print("=" * 60)
    print("Example 6: Batch Processing Multiple Folders")
    print("=" * 60)
    
    print("To process multiple mask folders at once:")
    print()
    print("```python")
    print("from merge_masks import process_masks")
    print()
    print("folders = [")
    print("    ('dataset1/train/masks', 'dataset1/train/merged'),")
    print("    ('dataset1/val/masks', 'dataset1/val/merged'),")
    print("    ('dataset2/train/masks', 'dataset2/train/merged'),")
    print("]")
    print()
    print("for input_folder, output_folder in folders:")
    print("    print(f'Processing {input_folder}...')")
    print("    process_masks(input_folder, output_folder)")
    print("    print('Done!\\n')")
    print("```")
    print()


def main():
    """
    Display all examples
    """
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "MERGE MASKS - USAGE EXAMPLES" + " " * 20 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    example_1_basic_usage()
    example_2_problem_statement_paths()
    example_3_filename_extraction()
    example_4_inspect_before_merge()
    example_5_custom_processing()
    example_6_batch_processing()
    
    print("=" * 60)
    print("For more information, see MERGE_MASKS_README.md")
    print("=" * 60)
    print()


if __name__ == '__main__':
    main()
