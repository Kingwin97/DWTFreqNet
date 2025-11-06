"""
Unit tests for the merge_masks.py script.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

# Add the current directory to the path to import merge_masks
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merge_masks import extract_base_name, group_masks_by_base_image, merge_masks, process_masks


def test_extract_base_name():
    """Test the base name extraction function."""
    print("Testing extract_base_name()...")
    
    # Test cases
    test_cases = [
        ('20251008新数据-大面龟裂_42586468_裂纹_1_mask.png', '20251008新数据-大面龟裂_42586468'),
        ('20251008新数据-大面龟裂_42586468_破损_2_mask.png', '20251008新数据-大面龟裂_42586468'),
        ('testimage_12345678_缺陷A_1_mask.png', 'testimage_12345678'),
        ('image_999_defect_3_mask.png', 'image_999'),
        ('simple_name_type_5_mask.png', 'simple_name'),
    ]
    
    passed = 0
    for filename, expected in test_cases:
        result = extract_base_name(filename)
        if result == expected:
            print(f"  ✓ {filename} -> {result}")
            passed += 1
        else:
            print(f"  ✗ {filename} -> {result} (expected: {expected})")
    
    print(f"  Passed {passed}/{len(test_cases)} tests\n")
    return passed == len(test_cases)


def test_merge_masks():
    """Test the mask merging function."""
    print("Testing merge_masks()...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test masks
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask1[10:30, 10:30] = 255
        
        mask2 = np.zeros((100, 100), dtype=np.uint8)
        mask2[70:90, 70:90] = 255
        
        mask3 = np.zeros((100, 100), dtype=np.uint8)
        mask3[40:60, 40:60] = 128  # Different intensity
        
        # Save masks
        mask1_path = tmpdir / 'mask1.png'
        mask2_path = tmpdir / 'mask2.png'
        mask3_path = tmpdir / 'mask3.png'
        
        Image.fromarray(mask1, mode='L').save(mask1_path)
        Image.fromarray(mask2, mode='L').save(mask2_path)
        Image.fromarray(mask3, mode='L').save(mask3_path)
        
        # Test merging
        merged = merge_masks([mask1_path, mask2_path, mask3_path])
        
        # Check if all regions are present
        has_region1 = np.any(merged[10:30, 10:30] == 255)
        has_region2 = np.any(merged[70:90, 70:90] == 255)
        has_region3 = np.any(merged[40:60, 40:60] > 0)
        
        if has_region1 and has_region2 and has_region3:
            print("  ✓ All regions correctly merged")
            print(f"  ✓ Non-zero pixels: {np.count_nonzero(merged)}")
            return True
        else:
            print("  ✗ Merge failed")
            print(f"    Region 1: {has_region1}, Region 2: {has_region2}, Region 3: {has_region3}")
            return False


def test_group_masks_by_base_image():
    """Test the mask grouping function."""
    print("\nTesting group_masks_by_base_image()...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test mask files
        test_files = [
            'image1_42586468_defect1_1_mask.png',
            'image1_42586468_defect2_2_mask.png',
            'image2_12345678_defect1_1_mask.png',
            'image3_99999999_defect1_1_mask.png',
            'image3_99999999_defect2_2_mask.png',
            'image3_99999999_defect3_3_mask.png',
        ]
        
        for filename in test_files:
            # Create empty files
            (tmpdir / filename).touch()
        
        # Group the masks
        groups = group_masks_by_base_image(tmpdir)
        
        # Verify grouping
        expected_groups = {
            'image1_42586468': 2,
            'image2_12345678': 1,
            'image3_99999999': 3,
        }
        
        all_correct = True
        for base_name, expected_count in expected_groups.items():
            actual_count = len(groups.get(base_name, []))
            if actual_count == expected_count:
                print(f"  ✓ {base_name}: {actual_count} masks")
            else:
                print(f"  ✗ {base_name}: {actual_count} masks (expected: {expected_count})")
                all_correct = False
        
        return all_correct


def test_process_masks_integration():
    """Test the complete process_masks function."""
    print("\nTesting process_masks() integration...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / 'input'
        output_dir = Path(tmpdir) / 'output'
        input_dir.mkdir()
        
        # Create test masks with actual image data
        base_images = {
            'image_A': 2,
            'image_B': 3,
            'image_C': 1,
        }
        
        for base_name, num_masks in base_images.items():
            for i in range(num_masks):
                # Create a mask with a unique pattern
                mask = np.zeros((50, 50), dtype=np.uint8)
                mask[i*10:(i+1)*10, i*10:(i+1)*10] = 255
                
                filename = f"{base_name}_defect{i+1}_{i+1}_mask.png"
                Image.fromarray(mask, mode='L').save(input_dir / filename)
        
        # Process the masks
        process_masks(input_dir, output_dir)
        
        # Check output
        output_files = list(output_dir.glob('*.png'))
        
        if len(output_files) == len(base_images):
            print(f"  ✓ Created {len(output_files)} merged masks")
            
            # Verify filenames
            expected_names = {f"{name}.png" for name in base_images.keys()}
            actual_names = {f.name for f in output_files}
            
            if expected_names == actual_names:
                print(f"  ✓ All filenames correct")
                return True
            else:
                print(f"  ✗ Filename mismatch")
                print(f"    Expected: {expected_names}")
                print(f"    Actual: {actual_names}")
                return False
        else:
            print(f"  ✗ Wrong number of output files: {len(output_files)} (expected: {len(base_images)})")
            return False


def test_edge_cases():
    """Test edge cases and special scenarios."""
    print("\nTesting edge cases...")
    
    # Test with Chinese characters in filename
    filename1 = '20251008新数据-大面龟裂_42586468_裂纹_1_mask.png'
    base1 = extract_base_name(filename1)
    print(f"  ✓ Chinese characters: '{filename1}' -> '{base1}'")
    
    # Test with long numeric IDs
    filename2 = 'test_123456789012345_defect_1_mask.png'
    base2 = extract_base_name(filename2)
    print(f"  ✓ Long numeric ID: '{filename2}' -> '{base2}'")
    
    # Test with special characters
    filename3 = 'test-name_with-dash_456_type_1_mask.png'
    base3 = extract_base_name(filename3)
    print(f"  ✓ Special characters: '{filename3}' -> '{base3}'")
    
    return True


def run_all_tests():
    """Run all test functions."""
    print("=" * 60)
    print("Running merge_masks.py Test Suite")
    print("=" * 60 + "\n")
    
    results = []
    
    results.append(("extract_base_name", test_extract_base_name()))
    results.append(("merge_masks", test_merge_masks()))
    results.append(("group_masks_by_base_image", test_group_masks_by_base_image()))
    results.append(("process_masks_integration", test_process_masks_integration()))
    results.append(("edge_cases", test_edge_cases()))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed successfully!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
