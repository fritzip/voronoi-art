# Voronoi Art

Transform images into artistic Voronoi diagrams with adaptive sampling and posterization effects. Creates vector (SVG) and raster (PNG) outputs.

## Features

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
| `--points`    | int    | 6000              | Number of Voronoi sites (more = finer detail)                     |
| `--strength`  | float  | 3.0               | Adaptive sampling intensity (higher = more detail emphasis)       |
| `--blur`      | int    | 2                 | Variance map smoothness (lower = sharper transitions)             |
| `--posterize` | int    | 0                 | Color levels (0=off, 8/16/32 for artistic effects)                |
| `--scale`     | float  | 1.0               | Output size multiplier (2.0 = double resolution)                  |
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

## Tips

- Start with default settings, then adjust `--points` and `--strength`
- Use `--posterize 8` or `16` for pop art effects
- Add `--edges` to understand how the algorithm samples your image
- Increase `--scale` for high-resolution prints without adding more points
- Lower `--points` (500-1500) for abstract, low-poly style art
