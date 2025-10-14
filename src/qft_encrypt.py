import time
import matplotlib.pyplot as plt
from utils import qft2, iqft2, add_random_phase, remove_random_phase, mse, psnr

import os
from PIL import Image, ImageDraw
import numpy as np

def load_image(path: str, size=(128, 128)) -> np.ndarray:
    """Load grayscale image; if not found, generate one automatically."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print("⚙️ No image found — generating a sample grayscale image automatically...")
        # Create a synthetic gradient pattern
        img = Image.new("L", size)
        draw = ImageDraw.Draw(img)
        for x in range(size[0]):
            for y in range(size[1]):
                # Simple gradient pattern
                pixel_value = int((x + y) / (2 * size[0]) * 255)
                draw.point((x, y), fill=pixel_value)
        img.save(path)
    else:
        print(f"📸 Found existing image: {path}")
    
    img = Image.open(path).convert("L").resize(size)
    return np.array(img, dtype=float)



def save_image(array: np.ndarray, path: str):
    """Save a numpy array as grayscale image, creating folder if missing."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(np.uint8(np.clip(array, 0, 255))).save(path)


def visualize_results(original, encrypted, decrypted):
    """Display the images side by side."""
    fig, ax = plt.subplots(1, 3, figsize=(10, 4))
    ax[0].imshow(original, cmap="gray"); ax[0].set_title("Original")
    ax[1].imshow(np.abs(encrypted), cmap="gray"); ax[1].set_title("Encrypted (Magnitude)")
    ax[2].imshow(decrypted, cmap="gray"); ax[2].set_title("Decrypted")
    for a in ax: a.axis("off")
    plt.tight_layout(); plt.show()

def main():
    path = "data/sample_image.png"   # Put your grayscale image here
    img = load_image(path)
    print(f"Loaded image of shape: {img.shape}")

    start = time.time()
    q_img = qft2(img)
    encrypted, key = add_random_phase(q_img)
    enc_time = time.time() - start
    print(f"Encryption time: {enc_time:.4f} s")

    start = time.time()
    decrypted_q = remove_random_phase(encrypted, key)
    decrypted = np.real(iqft2(decrypted_q))
    dec_time = time.time() - start
    print(f"Decryption time: {dec_time:.4f} s")

    error = mse(img, decrypted)
    quality = psnr(img, decrypted)
    print(f"MSE: {error:.6f} | PSNR: {quality:.2f} dB")

    save_image(np.abs(encrypted), "outputs/encrypted.png")
    save_image(decrypted, "outputs/decrypted.png")
    visualize_results(img, encrypted, decrypted)

if __name__ == "__main__":
    main()
