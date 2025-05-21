import numpy as np
from scipy.ndimage import gaussian_filter

def binary_to_heatmap(mask: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    Convert a binary corner mask into a Gaussian heatmap.

    Args:
        mask: 2D numpy array (uint8 or bool), with 1s at corner pixels and 0s elsewhere.
        sigma: Standard deviation for the Gaussian kernel.

    Returns:
        2D float32 numpy array of the same shape, with values normalized to [0, 1].
    """
    # Ensure float32 for filtering
    mask_float = mask.astype(np.float32)
    # Apply Gaussian filter
    heatmap = gaussian_filter(mask_float, sigma=sigma)
    # Normalize to [0, 1]
    max_val = heatmap.max()
    if max_val > 0:
        heatmap /= max_val
    return heatmap

# Example usage
if __name__ == "__main__":
    # Create a sample binary mask
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30, 40] = 1
    mask[70, 80] = 1

    # Generate Gaussian heatmap
    heatmap = binary_to_heatmap(mask, sigma=3.0)

    # Display ranges
    print("Heatmap min/max:", heatmap.min(), heatmap.max())
