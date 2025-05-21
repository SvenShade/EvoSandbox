import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

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

def make_lorentzian_kernel(size, gamma):
    """
    Create a normalized 2D Lorentzian (Cauchy) kernel.
    
    Args:
        size: int, kernel will be (size x size), size should be odd.
        gamma: scale parameter (controls width).
    Returns:
        kernel: 2D float32 numpy array summing to 1.
    """
    assert size % 2 == 1, "Size must be odd"
    ax = np.arange(-size//2 + 1, size//2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    rr2 = xx**2 + yy**2
    # Unnormalized Lorentzian
    kernel = 1.0 / (1.0 + rr2/(gamma**2))
    # Normalize so sum == 1
    kernel /= kernel.sum()
    return kernel

def binary_to_lorentz_heatmap(mask, size=31, gamma=3.0):
    """
    Convolve binary mask with a Lorentzian kernel.
    """
    kernel = make_lorentzian_kernel(size, gamma)
    # Use fft-based convolution for speed & same-shape output
    heatmap = fftconvolve(mask.astype(np.float32), kernel, mode='same')
    # Optional: renormalize so peak == 1
    peak = heatmap.max()
    if peak > 0:
        heatmap /= peak
    return heatmap
