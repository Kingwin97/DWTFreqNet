import numpy as np
import cv2


def calculate_iou(pred_img, gt_img):
    """
    Calculate IoU between two binary images

    Args:
        pred_img: Predicted binary image (numpy array with values 0 or 1)
        gt_img: Ground truth binary image (numpy array with values 0 or 1)

    Returns:
        iou: IoU score
    """
    # Ensure images are binary
    pred_img = pred_img.astype(bool)
    gt_img = gt_img.astype(bool)

    # Calculate intersection and union
    intersection = np.logical_and(pred_img, gt_img).sum()
    union = np.logical_or(pred_img, gt_img).sum()

    # Calculate IoU
    iou = intersection / union if union != 0 else 0

    return iou


# Example usage
if __name__ == "__main__":
    # Read images
    # Note: Ensure your images are already binary (0 or 1)
    pred_path = r"E:\Research_Topic\about_IR\duibi_image_visual\NUDT-SIRST\me\000296.png"
    gt_path = r"E:\Research_Topic\about_IR\Dataset\NUDT-SIRST\masks\000296.png"

    pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    # Convert to binary if images are not already binary
    # Assuming white (255) is the foreground
    pred_img = (pred_img > 127).astype(np.uint8)
    gt_img = (gt_img > 127).astype(np.uint8)

    # Calculate IoU
    iou_score = calculate_iou(pred_img, gt_img)
    print(f"IoU Score: {iou_score:.4f}")