# voronoi_art.py
# Adaptive Voronoi vectorization with optional posterization

import cv2, numpy as np, random, svgwrite, cairosvg, argparse, os, sys
from scipy.spatial import Voronoi
from pathlib import Path


def get_color_options():
    """Get color options list from the parse_color color_map."""
    # Use the same color_map as parse_color for consistency
    color_map = {
        "black": (24, 24, 24),
        "white": (245, 245, 245),
        "red": (220, 50, 47),
        "green": (38, 139, 21),
        "blue": (38, 139, 210),
        "yellow": (203, 153, 50),
        "cyan": (42, 161, 152),
        "magenta": (174, 54, 183),
        "orange": (217, 95, 2),
        "purple": (108, 113, 196),
        "brown": (150, 75, 0),
        "pink": (215, 110, 160),
        "gray": (120, 120, 120),
    }

    # Convert to trackbar format: (display_name, rgb_tuple)
    return [(name.title(), rgb) for name, rgb in color_map.items()]


def parse_color(color_str):
    """Parse color string and return RGB tuple (0-255)."""
    color_str = color_str.lower().strip()

    # Handle hex colors
    if color_str.startswith("#"):
        hex_color = color_str[1:]
        if len(hex_color) == 6:
            return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        elif len(hex_color) == 3:
            return tuple(int(c * 2, 16) for c in hex_color)

    # Handle named colors - use same color definitions as get_color_options
    color_options = get_color_options()
    color_map = {name.lower(): rgb for name, rgb in color_options}
    color_map["grey"] = color_map.get("gray", (120, 120, 120))  # Add grey alias

    if color_str in color_map:
        return color_map[color_str]

    # Default to black if parsing fails
    print(f"Warning: Could not parse color '{color_str}', using black")
    return (24, 24, 24)


def posterize_image(img, n_colors):
    """Posterize image to n_colors."""
    if n_colors <= 1:
        return img
    # Simple uniform quantization
    bins = np.linspace(0, 256, n_colors + 1)
    quantized = np.digitize(img, bins) - 1
    quantized = (bins[quantized] + bins[quantized + 1]) / 2
    return quantized.astype(np.uint8)


def compute_voronoi_data(img, points, strength, blur, posterize, seed=None):
    """Compute Voronoi tessellation data. Returns (vor, chosen, working_img)."""
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Apply posterization if needed
    working_img = posterize_image(img.copy(), posterize) if posterize > 1 else img.copy()

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

    return vor, chosen, working_img


def process_voronoi_regions(vor, chosen, working_img):
    """Process Voronoi regions and yield (polygon, color) for valid regions."""
    h, w = working_img.shape[:2]

    for i, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if not region:
            continue

        # For infinite regions, filter out -1 vertices
        is_infinite = -1 in region
        if is_infinite:
            valid_vertices = [j for j in region if j != -1]
            if len(valid_vertices) < 3:
                continue
            polygon = [vor.vertices[j] for j in valid_vertices]
        else:
            polygon = [vor.vertices[j] for j in region]
            if len(polygon) < 3:
                continue

        # Get color from center point
        px, py = map(int, chosen[i])
        px, py = np.clip(px, 0, w - 1), np.clip(py, 0, h - 1)

        # For infinite cells, sample color from the nearest edge
        if is_infinite:
            # Find which edge is closest
            dist_to_left = px
            dist_to_right = w - 1 - px
            dist_to_top = py
            dist_to_bottom = h - 1 - py

            min_dist = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)

            # Sample from that edge
            if min_dist == dist_to_left:
                color = working_img[py, 0]  # Left edge
            elif min_dist == dist_to_right:
                color = working_img[py, w - 1]  # Right edge
            elif min_dist == dist_to_top:
                color = working_img[0, px]  # Top edge
            else:
                color = working_img[h - 1, px]  # Bottom edge
        else:
            color = working_img[int(py), int(px)]

        yield polygon, color


def generate_voronoi_image(img, points, strength, blur, posterize, edge_color=(0, 0, 0), edge_thickness=1, seed=None):
    """Generate a Voronoi rendering of the image with given parameters."""
    vor, chosen, working_img = compute_voronoi_data(img, points, strength, blur, posterize, seed)

    h, w = working_img.shape[:2]

    # Create output image
    output = np.zeros_like(working_img)

    for polygon, color in process_voronoi_regions(vor, chosen, working_img):
        pts = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(output, [pts], color.tolist())

        # Draw edges if requested
        # Draw edges when thickness > 0 (edge_thickness units are relative; scale for raster preview)
        if edge_thickness > 0.01:  # More strict threshold to avoid hairline edges
            cv_thickness = max(1, int(edge_thickness * 3))  # Scale thickness for raster preview
            cv2.polylines(output, [pts], True, edge_color, cv_thickness)

    return output


def preview_mode(img_path):
    """Interactive preview mode with real-time parameter adjustment."""
    # Load image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found or unsupported format: {img_path}")

    h, w = img.shape[:2]

    # For display only - we'll generate on full image but display scaled
    max_display = 800
    if h > max_display or w > max_display:
        scale_factor = max_display / max(h, w)
        display_h, display_w = int(h * scale_factor), int(w * scale_factor)
    else:
        display_h, display_w = h, w
        scale_factor = 1.0

    # Create window
    window_name = "Voronoi Art Preview - Press 'q' to quit, ENTER to save"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_w, display_h + 200)

    # Parameters (strength: 0-10 exponential, posterize: 0-10 powers of 2, scale: 1-10, seed: 0-20)
    params = {
        "points": 6000,
        "strength": 1,
        "blur": 2,
        "posterize": 0,
        "scale": 1,
        "seed": 0,
        "edge_color": 0,
        "edge_thickness": 3,
    }

    # Use same color palette as parse_color function
    color_options = get_color_options()

    # Create trackbars
    cv2.createTrackbar("Points", window_name, params["points"], 100000, lambda x: None)
    cv2.createTrackbar("Strength", window_name, params["strength"], 10, lambda x: None)
    cv2.createTrackbar("Blur", window_name, params["blur"], 20, lambda x: None)
    cv2.createTrackbar("Edge Color", window_name, params["edge_color"], len(color_options) - 1, lambda x: None)
    cv2.createTrackbar("Edge Thickness", window_name, params["edge_thickness"], 20, lambda x: None)
    cv2.createTrackbar("Posterize", window_name, params["posterize"], 10, lambda x: None)
    cv2.createTrackbar("Scale", window_name, params["scale"], 10, lambda x: None)
    cv2.createTrackbar("Seed", window_name, params["seed"], 20, lambda x: None)

    print("\n" + "=" * 60)
    print("VORONOI ART PREVIEW MODE")
    print("=" * 60)
    print("Adjust the sliders to preview different settings:")
    print("  • Points: Number of Voronoi sites (more = finer detail)")
    print("  • Strength: Adaptive sampling intensity (0-10, exponential scale)")
    print("  • Blur: Variance map smoothness")
    print("  • Edge Color: Color of cell borders")
    print("  • Edge Thickness: Thickness of cell borders (0.1-2.0)")
    print("  • Posterize: Color reduction (0=off, 1=2 colors, 2=4, 3=8, ..., 10=1024)")
    print("  • Scale: Output size multiplier (1-10)")
    print("  • Seed: Random seed for reproducible results (0-20)")
    print("\nPress ENTER to save with current settings")
    print("Press 'q' to quit without saving")
    print("=" * 60 + "\n")

    last_params = params.copy()
    result = None
    needs_update = True

    while True:
        # Get current trackbar values
        params["points"] = max(100, cv2.getTrackbarPos("Points", window_name))
        params["strength"] = cv2.getTrackbarPos("Strength", window_name)
        params["blur"] = max(1, cv2.getTrackbarPos("Blur", window_name))
        params["edge_color"] = cv2.getTrackbarPos("Edge Color", window_name)
        params["edge_thickness"] = cv2.getTrackbarPos("Edge Thickness", window_name)
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
            if params["strength"] == 0:
                actual_strength = 0
            else:
                actual_strength = np.exp(params["strength"]) / np.exp(10) * 100  # Scale to 0-100

            # Convert posterize slider (0-10) to power of 2
            actual_posterize = 2 ** params["posterize"] if params["posterize"] > 0 else 0

            # Get edge color from selection
            edge_color_name, edge_color_rgb = color_options[params["edge_color"]]
            # Convert RGB to BGR for OpenCV
            edge_color_bgr = (edge_color_rgb[2], edge_color_rgb[1], edge_color_rgb[0])

            # Convert edge thickness from slider (0-20) to actual thickness (0.0-2.0)
            actual_edge_thickness = max(0.0, params["edge_thickness"] / 10.0)

            print(
                f"Generating preview: points={params['points']}, strength={actual_strength:.2f}, "
                f"blur={params['blur']}, "
                f"edge_color={edge_color_name}, edge_thickness={actual_edge_thickness:.1f}, "
                f"posterize={actual_posterize}, scale={params['scale']}x, seed={params['seed']}"
            )

            # Generate on FULL resolution image (same as final output)
            result = generate_voronoi_image(
                img,
                points=params["points"],
                strength=actual_strength,
                blur=params["blur"],
                posterize=actual_posterize,
                edge_color=edge_color_bgr,  # Use BGR for OpenCV
                edge_thickness=actual_edge_thickness,
                seed=params["seed"],
            )

            # Resize result for display if needed
            if scale_factor != 1.0:
                result_display = cv2.resize(result, (display_w, display_h))
            else:
                result_display = result

            needs_update = False

        # Display
        if result is not None:
            cv2.imshow(window_name, result_display)

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

            # Get edge color and thickness
            edge_color_name, edge_color_rgb = color_options[params["edge_color"]]
            # Convert RGB to BGR for OpenCV
            edge_color_bgr = (edge_color_rgb[2], edge_color_rgb[1], edge_color_rgb[0])
            actual_edge_thickness = max(0.0, params["edge_thickness"] / 10.0)

            return {
                "points": params["points"],
                "strength": actual_strength,
                "blur": params["blur"],
                "edge_color": edge_color_bgr,  # Return BGR for OpenCV consistency
                "edge_color_name": edge_color_name,
                "edge_thickness": actual_edge_thickness,
                "posterize": actual_posterize,
                "scale": params["scale"],
                "seed": params["seed"],
            }


def save_voronoi_output(
    img, output_svg, output_png, points, strength, blur, posterize, edge_color=(0, 0, 0), edge_thickness=0.3, scale=1.0, seed=None
):
    """Save Voronoi art to SVG and PNG files."""
    vor, chosen, working_img = compute_voronoi_data(img, points, strength, blur, posterize, seed)

    h, w = working_img.shape[:2]

    # SVG
    dwg = svgwrite.Drawing(output_svg, size=(w, h))

    for polygon, color in process_voronoi_regions(vor, chosen, working_img):
        # OpenCV colors are BGR, convert to RGB for SVG
        b, g, r = color

        # Only add stroke attributes when edge_thickness > 0.01 to completely avoid hairlines
        if edge_thickness > 0.01:
            if isinstance(edge_color, tuple) and len(edge_color) == 3:
                # edge_color is in BGR format (from OpenCV), convert to RGB for SVG
                b_edge, g_edge, r_edge = edge_color
                stroke_color = svgwrite.rgb(r_edge, g_edge, b_edge)
            else:
                stroke_color = "black"  # fallback
            # Add polygon with stroke
            dwg.add(dwg.polygon(polygon, fill=svgwrite.rgb(r, g, b), stroke=stroke_color, stroke_width=edge_thickness))
        else:
            # Add polygon without any stroke attributes to avoid hairlines
            dwg.add(dwg.polygon(polygon, fill=svgwrite.rgb(r, g, b)))

    dwg.save()
    print(f"✅ Saved SVG: {output_svg}")

    # PNG export - use direct OpenCV rendering when no edges to avoid transparency gaps
    if edge_thickness <= 0.01:
        # Generate PNG directly from OpenCV to avoid anti-aliasing gaps
        output_img = generate_voronoi_image(img, points, strength, blur, posterize, edge_color, 0, seed)

        # Scale if needed
        if scale != 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            output_img = cv2.resize(output_img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Save as PNG
        cv2.imwrite(output_png, output_img)
        print(f"✅ Saved PNG: {output_png} (direct from OpenCV, no gaps)")
    else:
        # Use SVG->PNG conversion when edges are needed
        cairosvg.svg2png(url=output_svg, write_to=output_png, output_width=int(w * scale), output_height=int(h * scale))
        print(f"✅ Saved PNG: {output_png} (from SVG conversion)")


def main():
    p = argparse.ArgumentParser(description="Adaptive Voronoi vectorization with optional posterization")
    p.add_argument("--input", required=True, help="Input image path (supports PNG, JPG, JPEG, BMP, TIFF, WebP, etc.)")
    p.add_argument("--output", required=False, default=None, help="Output basename (no extension). Defaults to input filename")
    p.add_argument("--preview", action="store_true", help="Interactive preview mode with real-time parameter adjustment")
    p.add_argument("--points", type=int, default=6000, help="Number of Voronoi sites")
    p.add_argument("--strength", type=float, default=3.0, help="Effect of color variance on density")
    p.add_argument("--blur", type=int, default=2, help="Smoothness of variance map")
    # Note: edges flag removed; edge visibility is controlled by --edge-thickness (0 = no edge)
    p.add_argument("--edge-color", type=str, default="black", help="Color of Voronoi edges (e.g., 'black', 'white', 'red', '#FF0000')")
    p.add_argument("--edge-thickness", type=float, default=0.3, help="Thickness of Voronoi edges for SVG output")
    p.add_argument("--posterize", type=int, default=0, help="Posterize colors (0=off, e.g., 8,16,32)")
    p.add_argument("--scale", type=float, default=1.0, help="Size multiplier relative to input (e.g. 2.0 = double size)")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducible results (0-20)")
    args = p.parse_args()

    IMG_PATH = args.input

    # Default output name based on input filename and ensure it's in the same directory as input
    if args.output is None:
        input_path = Path(IMG_PATH)
        # Save in same directory as input, not current working directory
        args.output = str(input_path.parent / (input_path.stem + "_voronoi"))
    else:
        # If user provides custom output, ensure it's absolute or relative to input directory
        if not Path(args.output).is_absolute():
            input_path = Path(IMG_PATH)
            args.output = str(input_path.parent / args.output)

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
        args.edge_color = preview_params["edge_color_name"].lower()
        args.edge_thickness = preview_params["edge_thickness"]
        args.posterize = preview_params["posterize"]
        args.scale = preview_params["scale"]
        args.seed = preview_params["seed"]

        print(
            f"\n📝 Saving with parameters: points={args.points}, strength={args.strength:.2f}, "
            f"blur={args.blur}, edge_color={args.edge_color}, "
            f"edge_thickness={args.edge_thickness:.2f}, posterize={args.posterize}, "
            f"scale={args.scale:.2f}x, seed={args.seed}"
        )

    # Load image - OpenCV supports PNG, JPG, JPEG, BMP, TIFF, WebP, and more
    img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found or unsupported format: {IMG_PATH}")

    # Parse edge color
    edge_color_rgb = parse_color(args.edge_color)
    # Convert RGB to BGR for OpenCV consistency
    edge_color_bgr = (edge_color_rgb[2], edge_color_rgb[1], edge_color_rgb[0])

    # Generate and save final output
    save_voronoi_output(
        img,
        OUTPUT_SVG,
        OUTPUT_PNG,
        points=args.points,
        strength=args.strength,
        blur=args.blur,
        posterize=args.posterize,
        edge_color=edge_color_bgr,  # Use BGR for consistency
        edge_thickness=args.edge_thickness,
        scale=args.scale,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
