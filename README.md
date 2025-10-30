# Voronoi Art

Transform images into artistic Voronoi diagrams with adaptive sampling and posterization effects. Creates vector (SVG) and raster (PNG) outputs.

## Features

- **Interactive Preview Mode**: Real-time parameter adjustment with instant visual feedback
- **Adaptive Sampling**: Places more Voronoi sites in high-detail areas using variance-based density
- **Posterization**: Optional color reduction for artistic effects
- **Vector Output**: Generates crisp SVG files that scale infinitely
- **Flexible Scaling**: Control output resolution independently from input
- **Universal Image Support**: Works with PNG, JPG, JPEG, BMP, TIFF, WebP, and other formats
- **Debug Mode**: Visualize Voronoi cell boundaries

## Installation

### Using Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/fritzip/voronoi-art.git
cd voronoi-art

# Install dependencies with Poetry
poetry install

# Run the tool
poetry run voronoi-art --input image.jpg
```

### System Installation

```bash
# Install globally with Poetry
poetry build
pip install dist/voronoi_art-*.whl

# Now use from anywhere
voronoi-art --input ~/pictures/photo.jpg
```

## Usage

### Interactive Preview Mode (Recommended)

```bash
# Launch interactive preview with real-time parameter adjustment
voronoi-art --input photo.jpg --preview

# Adjust sliders in the GUI window:
#   - Points: Change number of Voronoi sites (0-100,000)
#   - Strength: Control adaptive sampling intensity (0-10, exponential scale)
#              0=off, 5≈0.67, 10=100 (uses e^x/e^10 scaling for fine control)
#   - Blur: Adjust variance map smoothness (1-20)
#   - Edges: Toggle cell borders on/off (0 or 1)
#   - Posterize: Change color quantization (0-10: 0=off, 1=2 colors, 2=4, 3=8, ..., 10=1024)
#   - Scale x0.1: Output size multiplier (1=0.1x, 10=1.0x, 50=5.0x, 100=10.0x)
#   - Seed: Random seed for reproducible results (0-100)
# Press ENTER to save with current settings
# Press 'q' to quit without saving
```

### Basic Usage

```bash
# Process an image with default settings
voronoi-art --input photo.jpg

# Output will be photo_voronoi.svg and photo_voronoi.png
```

### Advanced Options

```bash
# High detail with posterization and custom output
voronoi-art \
  --input landscape.jpg \
  --output artistic_landscape \
  --points 10000 \
  --posterize 16 \
  --strength 4.0 \
  --scale 2.0

# Low-poly style with visible edges
voronoi-art \
  --input portrait.png \
  --points 500 \
  --edges \
  --posterize 8
```

## Parameters

| Parameter     | Type   | Default           | Description                                                       |
| ------------- | ------ | ----------------- | ----------------------------------------------------------------- |
| `--input`     | string | _required_        | Input image path (supports PNG, JPG, JPEG, BMP, TIFF, WebP, etc.) |
| `--output`    | string | `{input}_voronoi` | Output basename without extension                                 |
| `--preview`   | flag   | off               | Launch interactive preview mode with real-time adjustments        |
| `--points`    | int    | 6000              | Number of Voronoi sites (more = finer detail)                     |
| `--strength`  | float  | 3.0               | Adaptive sampling intensity (higher = more detail emphasis)       |
| `--blur`      | int    | 2                 | Variance map smoothness (lower = sharper transitions)             |
| `--posterize` | int    | 0                 | Color levels (0=off, 8/16/32 for artistic effects)                |
| `--scale`     | float  | 1.0               | Output size multiplier (2.0 = double resolution)                  |
| `--seed`      | int    | None              | Random seed for reproducible results (0-100)                      |
| `--edges`     | flag   | off               | Show Voronoi cell borders for debugging                           |

## Examples

### Natural Image (6000 points)

Balanced detail preservation with adaptive sampling.

### Posterized Art (500 points, 8 colors)

Create bold, geometric low-poly art.

### High Resolution (10000 points, 2x scale)

Maximum detail for large prints.

## How It Works

1. **Variance Analysis**: Calculates color variance across the image
2. **Adaptive Sampling**: Places more Voronoi sites in high-variance (detailed) regions
3. **Voronoi Tessellation**: Generates polygonal cells using scipy
4. **Color Sampling**: Each cell takes the color from its center point
5. **Vector Export**: Saves as SVG with optional posterization
6. **Raster Rendering**: Converts to PNG at specified scale using CairoSVG

## Requirements

- Python 3.8+
- NumPy
- OpenCV (cv2)
- SciPy
- svgwrite
- CairoSVG

## License

MIT License - feel free to use in your own projects!

## Contributing

Contributions welcome! Please open an issue or submit a pull request.
