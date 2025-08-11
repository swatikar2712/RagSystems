# Introduction to Digital Image Processing (PIL + OpenCV)

This mini project demonstrates how to:
- Read images using PIL (`Pillow`) and OpenCV (`cv2`)
- Convert images to NumPy arrays
- Plot and save visualizations (original, grayscale, edges)
- Render an ASCII representation of the image in the console

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Usage

Run with a provided image path:

```bash
python3 image_processing_intro.py --image /absolute/path/to/your/image.jpg --ascii-width 100
```

Or run without an image to generate a synthetic sample:

```bash
python3 image_processing_intro.py
```

Outputs are saved in the `outputs/` directory:
- `original_rgb.png`
- `grayscale.png`
- `edges_canny.png`
- `overview.png` (side-by-side plot)

The console will also print an ASCII art rendering of the grayscale image.