# voronoi_art_cli_posterize.py
# Requirements: pip install numpy opencv-python scipy svgwrite cairosvg

import cv2, numpy as np, random, svgwrite, cairosvg, argparse
from scipy.spatial import Voronoi


def posterize_image(img, n_colors):
    """Posterize image to n_colors."""
    # Simple uniform quantization
    bins = np.linspace(0, 256, n_colors + 1)
    quantized = np.digitize(img, bins) - 1
    quantized = (bins[quantized] + bins[quantized + 1]) / 2
    return quantized.astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description="Adaptive Voronoi vectorization with optional posterization")
    p.add_argument("--input", required=True, help="Input image path")
    p.add_argument("--output", required=False, default="output", help="Output basename (no extension)")
    p.add_argument("--points", type=int, default=6000, help="Number of Voronoi sites")
    p.add_argument("--strength", type=float, default=3.0, help="Effect of color variance on density")
    p.add_argument("--blur", type=int, default=2, help="Smoothness of variance map")
    p.add_argument("--edges", action="store_true", help="Show Voronoi borders (debug)")
    p.add_argument("--posterize", type=int, default=0, help="Posterize colors (0=off, e.g., 8,16,32)")
    p.add_argument("--scale", type=float, default=1.0, help="Size multiplier relative to input (e.g. 2.0 = double size)")
    args = p.parse_args()

    IMG_PATH = args.input
    OUTPUT_SVG = args.output + ".svg"
    OUTPUT_PNG = args.output + ".png"

    # Load image
    img = cv2.imread(IMG_PATH)
    if img is None:
        raise FileNotFoundError(f"Image not found: {IMG_PATH}")

    if args.posterize > 1:
        img = posterize_image(img, args.posterize)

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Variance map
    blur = cv2.GaussianBlur(gray, (0, 0), args.blur)
    var = cv2.GaussianBlur(gray**2, (0, 0), args.blur) - blur**2
    var = cv2.normalize(var, None, 0, 1, cv2.NORM_MINMAX)

    # Weighted random sampling
    weights = var * args.strength + 1e-3
    weights /= weights.sum()
    ys, xs = np.indices(var.shape)
    coords = np.column_stack((xs.ravel(), ys.ravel()))
    chosen = coords[np.random.choice(len(coords), size=args.points, p=weights.ravel())]

    # Voronoi
    vor = Voronoi(chosen)

    # SVG
    dwg = svgwrite.Drawing(OUTPUT_SVG, size=(w, h))
    stroke_color = "black" if args.edges else "none"

    for i, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if not region or -1 in region:
            continue
        polygon = [vor.vertices[j] for j in region if 0 <= j < len(vor.vertices)]
        if len(polygon) < 3:
            continue
        px, py = map(int, chosen[i])
        px, py = np.clip(px, 0, w - 1), np.clip(py, 0, h - 1)
        b, g, r = img[int(py), int(px)]
        dwg.add(dwg.polygon(polygon, fill=svgwrite.rgb(r, g, b), stroke=stroke_color, stroke_width=0.3 if args.edges else 0))

    dwg.save()
    print(f"✅ Saved SVG: {OUTPUT_SVG}")

    # PNG export
    cairosvg.svg2png(url=OUTPUT_SVG, write_to=OUTPUT_PNG, output_width=w * args.scale, output_height=h * args.scale)
    print(f"✅ Saved PNG: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
