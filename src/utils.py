import numpy as np

def generate_qft_matrix(N: int) -> np.ndarray:
    """Generate the NxN Quantum Fourier Transform (QFT) matrix."""
    omega = np.exp(2j * np.pi / N)
    j, k = np.meshgrid(np.arange(N), np.arange(N))
    Q = omega ** (j * k) / np.sqrt(N)
    return Q

def qft2(image: np.ndarray) -> np.ndarray:
    """Apply 2D QFT to an image matrix."""
    N, M = image.shape
    Qx = generate_qft_matrix(N)
    Qy = generate_qft_matrix(M)
    return Qx @ image @ Qy.T

def iqft2(q_image: np.ndarray) -> np.ndarray:
    """Apply inverse 2D QFT to a transformed image."""
    N, M = q_image.shape
    Qx = generate_qft_matrix(N)
    Qy = generate_qft_matrix(M)
    return np.conjugate(Qx.T) @ q_image @ np.conjugate(Qy)

def add_random_phase(q_image: np.ndarray, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Encrypt the QFT image by adding a random phase key."""
    np.random.seed(seed)
    phase_key = np.exp(1j * 2 * np.pi * np.random.rand(*q_image.shape))
    encrypted = q_image * phase_key
    return encrypted, phase_key

def remove_random_phase(encrypted: np.ndarray, phase_key: np.ndarray) -> np.ndarray:
    """Decrypt the QFT image by removing the phase key."""
    return encrypted / phase_key

def mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Mean Squared Error between two images."""
    return np.mean((img1 - img2) ** 2)

def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio."""
    mse_val = mse(img1, img2)
    if mse_val == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse_val))
