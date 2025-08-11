#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np

# Use non-interactive backend suitable for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
import cv2


ASCII_CHARS = "@%#*+=-:. "


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def generate_synthetic_image(width: int = 320, height: int = 200) -> Image.Image:
    # Horizontal gradient
    x = np.linspace(0, 255, width, dtype=np.uint8)
    gradient = np.tile(x, (height, 1))

    # Add a vertical sine wave pattern for interest
    y = np.arange(height).reshape(-1, 1)
    sine = (127.5 * (1 + np.sin(2 * np.pi * y / 32))).astype(np.uint8)
    pattern = np.clip(0.6 * gradient + 0.4 * sine, 0, 255).astype(np.uint8)

    # Create RGB by stacking with phase shifts
    img_rgb = np.stack([
        pattern,
        np.roll(pattern, shift=10, axis=1),
        np.roll(pattern, shift=20, axis=1),
    ], axis=2)

    return Image.fromarray(img_rgb, mode="RGB")


def pil_read_image(image_path: Path) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    return img


def cv2_read_image(image_path: Path) -> np.ndarray:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image with OpenCV: {image_path}")
    return bgr


def convert_to_numpy_from_pil(img_pil: Image.Image) -> np.ndarray:
    return np.array(img_pil)  # RGB, shape (H, W, 3)


def convert_bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def to_grayscale_rgb(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


def image_to_ascii(gray: np.ndarray, width: int = 80) -> str:
    if gray.ndim != 2:
        raise ValueError("ASCII conversion expects a 2D grayscale array")

    h, w = gray.shape
    if w == 0 or h == 0:
        return ""

    # Characters are taller than wide; adjust height to preserve aspect
    aspect = h / float(w)
    new_w = max(4, width)
    new_h = max(1, int(aspect * new_w * 0.55))

    # Resize using PIL for demonstration
    pil_gray = Image.fromarray(gray, mode="L")
    pil_small = pil_gray.resize((new_w, new_h), resample=Image.BILINEAR)
    small = np.array(pil_small)

    idx = (small.astype(np.float32) / 255.0 * (len(ASCII_CHARS) - 1)).astype(np.int32)
    lines = ["".join(ASCII_CHARS[i] for i in row) for row in idx]
    return "\n".join(lines)


def plot_and_save(img_rgb: np.ndarray, gray: np.ndarray, edges: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original (RGB)")
    axes[0].axis("off")

    axes[1].imshow(gray, cmap="gray")
    axes[1].set_title("Grayscale")
    axes[1].axis("off")

    axes[2].imshow(edges, cmap="gray")
    axes[2].set_title("Canny Edges")
    axes[2].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_array_info(name: str, arr: np.ndarray) -> None:
    print(f"{name}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min()}, max={arr.max()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Intro to image processing with PIL and OpenCV")
    parser.add_argument("--image", type=str, default="", help="Path to an input image. If omitted, a synthetic sample is used.")
    parser.add_argument("--ascii-width", type=int, default=100, help="Width (characters) for ASCII rendering in console")
    parser.add_argument("--out-dir", type=str, default="outputs", help="Directory to save outputs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_output_dir(out_dir)

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        img_pil = pil_read_image(image_path)
    else:
        img_pil = generate_synthetic_image()
        sample_path = out_dir / "sample.png"
        img_pil.save(sample_path)
        print(f"No --image provided. Generated synthetic sample at: {sample_path}")

    # PIL -> numpy (RGB)
    img_rgb_from_pil = convert_to_numpy_from_pil(img_pil)
    print_array_info("PIL->numpy RGB", img_rgb_from_pil)

    # OpenCV read (BGR) for comparison
    # Save a temp file to ensure consistent source for cv2 when synthetic
    tmp_path = out_dir / "tmp_for_cv2.png"
    img_pil.save(tmp_path)
    img_bgr_from_cv2 = cv2_read_image(tmp_path)
    print_array_info("cv2 BGR", img_bgr_from_cv2)

    # Convert BGR->RGB for visualization
    img_rgb_from_cv2 = convert_bgr_to_rgb(img_bgr_from_cv2)

    # Use cv2 pipeline for grayscale and edges
    gray = to_grayscale_rgb(img_rgb_from_cv2)
    edges = cv2.Canny(gray, 100, 200)

    # Console ASCII visualization
    ascii_art = image_to_ascii(gray, width=args.ascii_width)
    print("\nASCII visualization (grayscale):")
    print(ascii_art)

    # Save individual artifacts
    Image.fromarray(img_rgb_from_pil).save(out_dir / "original_rgb.png")
    Image.fromarray(gray).save(out_dir / "grayscale.png")
    Image.fromarray(edges).save(out_dir / "edges_canny.png")

    # Plot and save comparison figure
    plot_and_save(img_rgb_from_pil, gray, edges, out_dir / "overview.png")

    print(f"\nSaved outputs to: {out_dir.resolve()}")
    print("- original_rgb.png")
    print("- grayscale.png")
    print("- edges_canny.png")
    print("- overview.png")


if __name__ == "__main__":
    main()