# voronoi_art.py
# Adaptive Voronoi vectorization with optional posterization

import cv2, numpy as np, random, svgwrite, cairosvg, argparse, os, sys
from scipy.spatial import Voronoi
from pathlib import Path


def posterize_image(img, n_colors):
    """Posterize image to n_colors."""
    if n_colors <= 1:
        return img
    # Simple uniform quantization
    bins = np.linspace(0, 256, n_colors + 1)
    quantized = np.digitize(img, bins) - 1
    quantized = (bins[quantized] + bins[quantized + 1]) / 2
    return quantized.astype(np.uint8)


def generate_voronoi_image(img, points, strength, blur, edges, posterize, seed=None):
    """Generate a Voronoi rendering of the image with given parameters."""
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Apply posterization if needed
    working_img = posterize_image(img.copy(), posterize) if posterize > 1 else img.copy()

    h, w = working_img.shape[:2]
    gray = cv2.cvtColor(working_img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Variance map
    blur_img = cv2.GaussianBlur(gray, (0, 0), blur)
    var = cv2.GaussianBlur(gray**2, (0, 0), blur) - blur_img**2
    var = cv2.normalize(var, None, 0, 1, cv2.NORM_MINMAX)

    # Weighted random sampling
    weights = var * strength + 1e-3
    weights /= weights.sum()
    ys, xs = np.indices(var.shape)
    coords = np.column_stack((xs.ravel(), ys.ravel()))
    chosen = coords[np.random.choice(len(coords), size=points, p=weights.ravel())]

    # Voronoi
    vor = Voronoi(chosen)

    # Create output image
    output = np.zeros_like(working_img)

    for i, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if not region or -1 in region:
            continue
        polygon = [vor.vertices[j] for j in region if 0 <= j < len(vor.vertices)]
        if len(polygon) < 3:
            continue

        # Get color from center point
        px, py = map(int, chosen[i])
        px, py = np.clip(px, 0, w - 1), np.clip(py, 0, h - 1)
        color = working_img[int(py), int(px)]

        # Draw filled polygon
        pts = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(output, [pts], color.tolist())

        # Draw edges if requested
        if edges:
            cv2.polylines(output, [pts], True, (0, 0, 0), 1)

    return output


def preview_mode(img_path):
    """Interactive preview mode with real-time parameter adjustment."""
    # Load image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found or unsupported format: {img_path}")

    # Resize if too large for display
    h, w = img.shape[:2]
    max_display = 800
    if h > max_display or w > max_display:
        scale = max_display / max(h, w)
        display_h, display_w = int(h * scale), int(w * scale)
        display_img = cv2.resize(img, (display_w, display_h))
    else:
        display_img = img.copy()
        display_h, display_w = h, w

    # Create window
    window_name = "Voronoi Art Preview - Press 'q' to quit, ENTER to save"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_w, display_h + 200)

    # Parameters (strength: 0-10 exponential, posterize: 0-10 powers of 2, scale: 1-10, seed: 0-100)
    params = {"points": 6000, "strength": 1, "blur": 2, "edges": 0, "posterize": 0, "scale": 1, "seed": 0}

    # Create trackbars
    cv2.createTrackbar("Points", window_name, params["points"], 100000, lambda x: None)
    cv2.createTrackbar("Strength", window_name, params["strength"], 10, lambda x: None)
    cv2.createTrackbar("Blur", window_name, params["blur"], 20, lambda x: None)
    cv2.createTrackbar("Edges (on/off)", window_name, params["edges"], 1, lambda x: None)
    cv2.createTrackbar("Posterize", window_name, params["posterize"], 10, lambda x: None)
    cv2.createTrackbar("Scale", window_name, params["scale"], 10, lambda x: None)
    cv2.createTrackbar("Seed", window_name, params["seed"], 100, lambda x: None)

    print("\n" + "=" * 60)
    print("VORONOI ART PREVIEW MODE")
    print("=" * 60)
    print("Adjust the sliders to preview different settings:")
    print("  • Points: Number of Voronoi sites (more = finer detail)")
    print("  • Strength: Adaptive sampling intensity (0-10, exponential scale)")
    print("  • Blur: Variance map smoothness")
    print("  • Edges: Toggle cell borders (0=off, 1=on)")
    print("  • Posterize: Color reduction (0=off, 1=2 colors, 2=4, 3=8, ..., 10=1024)")
    print("  • Scale: Output size multiplier (1-10)")
    print("  • Seed: Random seed for reproducible results (0-100)")
    print("\nPress ENTER to save with current settings")
    print("Press 'q' to quit without saving")
    print("=" * 60 + "\n")

    last_params = params.copy()
    result = display_img.copy()
    needs_update = True

    while True:
        # Get current trackbar values
        params["points"] = max(100, cv2.getTrackbarPos("Points", window_name))
        params["strength"] = cv2.getTrackbarPos("Strength", window_name)
        params["blur"] = max(1, cv2.getTrackbarPos("Blur", window_name))
        params["edges"] = cv2.getTrackbarPos("Edges (on/off)", window_name)
        params["posterize"] = cv2.getTrackbarPos("Posterize", window_name)
        params["scale"] = max(1, cv2.getTrackbarPos("Scale", window_name))
        params["seed"] = cv2.getTrackbarPos("Seed", window_name)

        # Check if parameters changed
        if params != last_params:
            needs_update = True
            last_params = params.copy()

        # Generate preview
        if needs_update:
            # Convert strength from 0-10 scale using exponential: e^strength / e^10
            # This gives a range from ~0 (e^0/e^10 ≈ 0.000045) to 1.0 (e^10/e^10)
            # with more fine control at lower values
            max_strength_exp = 10
            if params["strength"] == 0:
                actual_strength = 0
            else:
                actual_strength = np.exp(params["strength"]) / np.exp(max_strength_exp) * 100  # Scale to 0-100

            # Convert posterize slider (0-10) to power of 2
            actual_posterize = 2 ** params["posterize"] if params["posterize"] > 0 else 0

            print(
                f"Generating preview: points={params['points']}, strength={actual_strength:.2f}, "
                f"blur={params['blur']}, edges={bool(params['edges'])}, "
                f"posterize={actual_posterize}, scale={params['scale']}x, seed={params['seed']}"
            )

            result = generate_voronoi_image(
                display_img,
                points=params["points"],
                strength=actual_strength,
                blur=params["blur"],
                edges=bool(params["edges"]),
                posterize=actual_posterize,
                seed=params["seed"],
            )
            needs_update = False

        # Display
        cv2.imshow(window_name, result)

        # Handle keyboard
        key = cv2.waitKey(100) & 0xFF
        if key == ord("q"):
            print("\n❌ Preview cancelled")
            cv2.destroyAllWindows()
            return None
        elif key == 13:  # Enter key
            cv2.destroyAllWindows()
            # Convert posterize back to power of 2
            actual_posterize = 2 ** params["posterize"] if params["posterize"] > 0 else 0
            # Convert strength using exponential scale
            if params["strength"] == 0:
                actual_strength = 0
            else:
                actual_strength = (np.exp(params["strength"]) / np.exp(10)) * 100

            return {
                "points": params["points"],
                "strength": actual_strength,
                "blur": params["blur"],
                "edges": bool(params["edges"]),
                "posterize": actual_posterize,
                "scale": params["scale"],
                "seed": params["seed"],
            }


def save_voronoi_output(img, output_svg, output_png, points, strength, blur, edges, posterize, scale=1.0, seed=None):
    """Save Voronoi art to SVG and PNG files."""
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Apply posterization if needed
    working_img = posterize_image(img.copy(), posterize) if posterize > 1 else img.copy()

    h, w = working_img.shape[:2]
    gray = cv2.cvtColor(working_img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Variance map
    blur_img = cv2.GaussianBlur(gray, (0, 0), blur)
    var = cv2.GaussianBlur(gray**2, (0, 0), blur) - blur_img**2
    var = cv2.normalize(var, None, 0, 1, cv2.NORM_MINMAX)

    # Weighted random sampling
    weights = var * strength + 1e-3
    weights /= weights.sum()
    ys, xs = np.indices(var.shape)
    coords = np.column_stack((xs.ravel(), ys.ravel()))
    chosen = coords[np.random.choice(len(coords), size=points, p=weights.ravel())]

    # Voronoi
    vor = Voronoi(chosen)

    # SVG
    dwg = svgwrite.Drawing(output_svg, size=(w, h))
    stroke_color = "black" if edges else "none"

    for i, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if not region or -1 in region:
            continue
        polygon = [vor.vertices[j] for j in region if 0 <= j < len(vor.vertices)]
        if len(polygon) < 3:
            continue
        px, py = map(int, chosen[i])
        px, py = np.clip(px, 0, w - 1), np.clip(py, 0, h - 1)
        b, g, r = working_img[int(py), int(px)]
        dwg.add(dwg.polygon(polygon, fill=svgwrite.rgb(r, g, b), stroke=stroke_color, stroke_width=0.3 if edges else 0))

    dwg.save()
    print(f"✅ Saved SVG: {output_svg}")

    # PNG export
    cairosvg.svg2png(url=output_svg, write_to=output_png, output_width=int(w * scale), output_height=int(h * scale))
    print(f"✅ Saved PNG: {output_png}")


def main():
    p = argparse.ArgumentParser(description="Adaptive Voronoi vectorization with optional posterization")
    p.add_argument("--input", required=True, help="Input image path (supports PNG, JPG, JPEG, BMP, TIFF, WebP, etc.)")
    p.add_argument("--output", required=False, default=None, help="Output basename (no extension). Defaults to input filename")
    p.add_argument("--preview", action="store_true", help="Interactive preview mode with real-time parameter adjustment")
    p.add_argument("--points", type=int, default=6000, help="Number of Voronoi sites")
    p.add_argument("--strength", type=float, default=3.0, help="Effect of color variance on density")
    p.add_argument("--blur", type=int, default=2, help="Smoothness of variance map")
    p.add_argument("--edges", action="store_true", help="Show Voronoi borders (debug)")
    p.add_argument("--posterize", type=int, default=0, help="Posterize colors (0=off, e.g., 8,16,32)")
    p.add_argument("--scale", type=float, default=1.0, help="Size multiplier relative to input (e.g. 2.0 = double size)")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducible results (0-100)")
    args = p.parse_args()

    IMG_PATH = args.input

    # Default output name based on input filename
    if args.output is None:
        input_path = Path(IMG_PATH)
        args.output = input_path.stem + "_voronoi"

    OUTPUT_SVG = args.output + ".svg"
    OUTPUT_PNG = args.output + ".png"

    # Preview mode
    if args.preview:
        print("🎨 Starting preview mode...")
        preview_params = preview_mode(IMG_PATH)
        if preview_params is None:
            return  # User cancelled

        # Update args with preview parameters
        args.points = preview_params["points"]
        args.strength = preview_params["strength"]
        args.blur = preview_params["blur"]
        args.edges = preview_params["edges"]
        args.posterize = preview_params["posterize"]
        args.scale = preview_params["scale"]
        args.seed = preview_params["seed"]

        print(
            f"\n📝 Saving with parameters: points={args.points}, strength={args.strength:.2f}, "
            f"blur={args.blur}, edges={args.edges}, posterize={args.posterize}, "
            f"scale={args.scale:.2f}x, seed={args.seed}"
        )

    # Load image - OpenCV supports PNG, JPG, JPEG, BMP, TIFF, WebP, and more
    img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found or unsupported format: {IMG_PATH}")

    # Generate and save final output
    save_voronoi_output(
        img,
        OUTPUT_SVG,
        OUTPUT_PNG,
        points=args.points,
        strength=args.strength,
        blur=args.blur,
        edges=args.edges,
        posterize=args.posterize,
        scale=args.scale,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
