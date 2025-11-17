# voronoi_art.py
# Adaptive Voronoi vectorization

import argparse
import cv2
import numpy as np
import svgwrite
import cairosvg
import dearpygui.dearpygui as dpg
from scipy.spatial import Voronoi
from pathlib import Path


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

    # Handle named colors
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
        "grey": (120, 120, 120),  # Alias
    }

    if color_str in color_map:
        return color_map[color_str]

    # Default to black if parsing fails
    print(f"Warning: Could not parse color '{color_str}', using black")
    return (24, 24, 24)


def compute_voronoi_data(img, points, strength, blur, seed=None):
    """Compute Voronoi tessellation data. Returns (vor, chosen, working_img)."""
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    working_img = img.copy()
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

    # Add boundary points to eliminate infinite regions
    # This approach is recommended in scipy.spatial.Voronoi docs for bounded diagrams
    h, w = working_img.shape[:2]

    # Create boundary points around the image perimeter
    # Add points outside the image bounds to ensure all interior regions are finite
    margin = max(w, h)  # 100% margin outside image bounds

    boundary_points = []

    # Top edge points
    for x in np.linspace(-margin, w + margin, num=max(10, w // 50)):
        boundary_points.append([x, -margin])

    # Right edge points
    for y in np.linspace(-margin, h + margin, num=max(10, h // 50)):
        boundary_points.append([w + margin, y])

    # Bottom edge points
    for x in np.linspace(w + margin, -margin, num=max(10, w // 50)):
        boundary_points.append([x, h + margin])

    # Left edge points
    for y in np.linspace(h + margin, -margin, num=max(10, h // 50)):
        boundary_points.append([-margin, y])

    # Combine interior points with boundary points
    all_points = np.vstack([chosen, np.array(boundary_points)])

    # Generate bounded Voronoi diagram
    vor = Voronoi(all_points)

    return vor, chosen, working_img


def process_voronoi_regions(vor, chosen, working_img, debug=False):
    """
    Process Voronoi regions and yield (polygon, color).
    """
    h, w = working_img.shape[:2]

    # Debug counters
    if debug:
        debug_counts = {
            "finite": 0,
            "empty_region": 0,
            "finite_degenerate": 0,
            "total_regions": len(vor.regions),
            "total_points": len(chosen),
            "total_voronoi_points": len(vor.points),
        }

    # Only process regions for the original interior points (not boundary points)
    num_interior_points = len(chosen)

    for i in range(num_interior_points):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]

        # Case 1: Empty region
        if not region:
            if debug:
                debug_counts["empty_region"] += 1
                print(f"  🔍 Point {i}: EMPTY region (region_idx={region_idx})")
            continue

        # With boundary points, all regions should be finite (no -1 vertices)
        if -1 in region:
            if debug:
                print(f"  ⚠️  Point {i}: Unexpected infinite region in bounded Voronoi!")
            continue

        # All regions should be finite now
        polygon = [vor.vertices[j] for j in region]

        # Case 2: Finite region with too few vertices
        if len(polygon) < 3:
            if debug:
                debug_counts["finite_degenerate"] += 1
                print(f"  🔍 Point {i}: FINITE DEGENERATE region (vertices={len(polygon)})")
            continue

        # Case 3: Finite region
        # Get color using center point for all regions (since they're all finite now)
        px, py = map(int, chosen[i])
        px, py = np.clip(px, 0, w - 1), np.clip(py, 0, h - 1)
        color = working_img[int(py), int(px)]

        if debug:
            debug_counts["finite"] += 1
            print(f"  ✅ Point {i}: FINITE region. Using center point color at ({px},{py}) = {color}")
        yield polygon, color

    # Print debug summary
    if debug:
        processed = debug_counts["finite"]
        skipped = debug_counts["empty_region"] + debug_counts["finite_degenerate"]
        print(f"\n📊 BOUNDED VORONOI PROCESSING SUMMARY:")
        print(f"   Interior points: {debug_counts['total_points']}")
        print(f"   Total Voronoi points (including boundary): {debug_counts['total_voronoi_points']}")
        print(f"   Total regions: {debug_counts['total_regions']}")
        print(f"   ✅ Processed: {processed} ({debug_counts['finite']} finite)")
        print(f"   ❌ Skipped: {skipped} ({debug_counts['empty_region']} empty + {debug_counts['finite_degenerate']} degenerate)")
        print(f"   Processing rate: {processed/debug_counts['total_points']*100:.1f}%")


def generate_voronoi_image(img, points, strength, blur, edge_color=(0, 0, 0), edge_thickness=1, seed=None):
    """Generate a Voronoi rendering of the image with given parameters."""
    print(
        f"Generating Voronoi image with {points} points, strength={strength:.2f}, blur={blur}, edge_color={edge_color}, edge_thickness={edge_thickness}, seed={seed}"
    )
    vor, chosen, working_img = compute_voronoi_data(img, points, strength, blur, seed)

    # Create output image
    output = np.zeros_like(working_img)

    for polygon, color in process_voronoi_regions(vor, chosen, working_img, debug=False):
        pts = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(output, [pts], color.tolist())

        # Draw edges if requested
        if edge_thickness > 0:
            cv2.polylines(output, [pts], True, edge_color, edge_thickness)

    return output


def preview_mode(img_path, output_svg=None, output_png=None, debug_enabled=False):
    """Interactive preview mode with real-time parameter adjustment using Dear PyGui."""
    # Load image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found or unsupported format: {img_path}")

    h, w = img.shape[:2]

    # Use reasonable window size for 4K and HD displays
    window_width = 1600
    window_height = 1000

    # Calculate control panel width
    control_panel_width = 500

    # Fixed reasonable display size - no need to resize dynamically
    # Calculate available space for preview at initial window size
    preview_width = window_width - control_panel_width - 60  # margins and spacing
    preview_height = window_height - 120  # title, margins, and spacing

    # Scale image to fit preview area while maintaining aspect ratio
    scale_factor = min(preview_width / w, preview_height / h)
    display_w = int(w * scale_factor)
    display_h = int(h * scale_factor)

    # Initialize parameters
    params = {
        "points_log": 3.0,  # Logarithmic scale: 1=10, 2=100, 3=1000, 4=10000, 5=100000
        "strength": 1.0,
        "blur": 2,
        "scale": 1.0,
        "seed": 0,
        "edge_color": [24 / 255, 24 / 255, 24 / 255],
        "edge_thickness": 1,
        "show_edges": True,
        "needs_update": True,
    }

    # Only add debug_draw if debug is enabled
    if debug_enabled:
        params["debug_draw"] = False

    result = None
    texture_tag = "preview_texture"
    image_tag = "preview_image"

    def update_preview():
        """Generate and update the preview image."""
        nonlocal result
        params["needs_update"] = False

        # Convert points slider (1-5) to actual points using logarithmic scale
        # 1 -> 10, 2 -> 100, 3 -> 1000, 4 -> 10000, 5 -> 100000
        actual_points = int(10 ** params["points_log"])

        # Convert strength (0-10) using exponential scale
        if params["strength"] == 0:
            actual_strength = 0
        else:
            actual_strength = np.exp(params["strength"]) / np.exp(10) * 100

        # Get edge color (convert from RGBA [0-1] to BGR [0-255])
        edge_color_bgr = (
            int(params["edge_color"][2] * 255),  # B
            int(params["edge_color"][1] * 255),  # G
            int(params["edge_color"][0] * 255),  # R
        )

        # Edge thickness - use 0 if edges are disabled
        actual_edge_thickness = params["edge_thickness"] if params["show_edges"] else 0.0

        # Compute Voronoi data once and render here so we can draw debug overlays
        vor, chosen, working_img = compute_voronoi_data(img, actual_points, actual_strength, params["blur"], params["seed"])

        # Create output image and draw filled polygons
        result = np.zeros_like(working_img)
        regions_info = []  # keep polygons for debug drawing

        # Enable debug output when debug_draw is on
        debug_mode = params.get("debug_draw", False)
        if debug_mode:
            print(f"\n🔬 DEBUG MODE - Processing {actual_points} points...")

        for polygon, color in process_voronoi_regions(vor, chosen, working_img, debug=debug_mode):
            pts = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(result, [pts], color.tolist())
            regions_info.append(pts)
            # Draw edges if requested
            if actual_edge_thickness > 0:
                cv2.polylines(result, [pts], True, edge_color_bgr, actual_edge_thickness)

        # Resize for display to match texture size (fixed at initialization)
        result_display = cv2.resize(result, (display_w, display_h))

        # Convert BGR to RGBA for Dear PyGui
        result_rgba = cv2.cvtColor(result_display, cv2.COLOR_BGR2RGBA)
        result_rgba = result_rgba.astype(np.float32) / 255.0

        # If debug draw is enabled, overlay contours and seed points (on a copy)
        if params.get("debug_draw"):
            overlay = result.copy()

            # Contour colors of Voronoi regions
            col = (0, 255, 0)

            # Draw contours
            for pts in regions_info:
                thickness = max(2, int(actual_edge_thickness + 1))  # Slightly thicker for visibility
                cv2.polylines(overlay, [pts], True, col, thickness)

            # Draw seed points (chosen) as small circles (white with black border)
            for i, (px, py) in enumerate(chosen):
                cx = int(np.clip(px, 0, overlay.shape[1] - 1))
                cy = int(np.clip(py, 0, overlay.shape[0] - 1))
                # White circle with black border for better visibility
                cv2.circle(overlay, (cx, cy), radius=3, color=(0, 0, 0), thickness=-1)  # Black fill
                cv2.circle(overlay, (cx, cy), radius=2, color=(255, 255, 255), thickness=-1)  # White center

            result_display = cv2.resize(overlay, (display_w, display_h))
        else:
            # Resize for display to match texture size (fixed at initialization)
            result_display = cv2.resize(result, (display_w, display_h))

        # Update texture data (texture already exists with fixed size)
        result_rgba = cv2.cvtColor(result_display, cv2.COLOR_BGR2RGBA)
        result_rgba = result_rgba.astype(np.float32) / 255.0
        dpg.set_value(texture_tag, result_rgba.flatten())

    def on_param_change(sender, app_data, user_data):
        """Callback when any parameter changes."""
        param_name = user_data

        # Round scale to 0.1 increments
        if param_name == "scale":
            app_data = round(app_data * 10) / 10

        params[param_name] = app_data

        # Update slider display values for logarithmic scales
        if param_name == "points_log":
            actual_points = int(10**app_data)
            dpg.set_value(sender, app_data)  # Ensure value is set
            dpg.configure_item(sender, format=f"{actual_points} points")
        elif param_name == "scale":
            # Calculate output dimensions
            output_w = int(w * app_data)
            output_h = int(h * app_data)
            dpg.configure_item(sender, format=f"{app_data:.1f}x ({output_w}×{output_h})")

        params["needs_update"] = True

    def on_save():
        """Callback when save button is clicked."""
        if output_svg is None or output_png is None:
            print("⚠️ No output paths specified")
            return

        # Convert parameters for CLI equivalent output
        actual_points = int(10 ** params["points_log"])
        if params["strength"] == 0:
            actual_strength = 0
        else:
            actual_strength = (np.exp(params["strength"]) / np.exp(10)) * 100

        edge_color_bgr = (
            int(params["edge_color"][2] * 255),
            int(params["edge_color"][1] * 255),
            int(params["edge_color"][0] * 255),
        )
        actual_edge_thickness = params["edge_thickness"] if params["show_edges"] else 0.0
        actual_scale = params["scale"]

        # Print CLI command equivalent BEFORE computation (escape # in color to avoid shell comment)
        output_basename = output_svg.replace(".svg", "") if output_svg.endswith(".svg") else output_svg
        edge_color_hex = (
            f"#{int(params['edge_color'][0]*255):02x}{int(params['edge_color'][1]*255):02x}{int(params['edge_color'][2]*255):02x}"
        )
        print(f"\n📋 CLI equivalent:")
        print(f"voronoi-art --input {img_path} --output {output_basename} \\")
        print(f"  --points {actual_points} --strength {actual_strength:.2f} --blur {params['blur']} \\")
        print(f"  --edge-color '{edge_color_hex}' --edge-thickness {actual_edge_thickness:.2f} \\")
        print(f"  --scale {actual_scale:.2f} --seed {params['seed']}")
        print(f"\n💾 Generating files...")

        params["needs_update"] = False  # Don't regenerate

        # Generate and save the output
        save_voronoi_output(
            img,
            output_svg,
            output_png,
            points=actual_points,
            strength=actual_strength,
            blur=params["blur"],
            edge_color=edge_color_bgr,
            edge_thickness=actual_edge_thickness,
            scale=actual_scale,
            seed=params["seed"],
        )

        print("✅ Files saved! You can continue adjusting or click Quit to exit.")

    def on_quit():
        """Callback when quit button is clicked."""
        dpg.stop_dearpygui()

    # Initialize Dear PyGui
    dpg.create_context()

    # Set global font scale for better readability
    font_scale = 1
    dpg.set_global_font_scale(font_scale)

    # Create texture registry
    with dpg.texture_registry(tag="texture_registry"):
        # Initialize with a blank image
        blank = np.zeros((display_h, display_w, 4), dtype=np.float32)
        dpg.add_raw_texture(
            width=display_w,
            height=display_h,
            default_value=blank.flatten(),
            format=dpg.mvFormat_Float_rgba,
            tag=texture_tag,
        )

    # Create main window
    with dpg.window(label="Voronoi Art Preview", tag="primary_window"):
        with dpg.group(horizontal=True):
            # Left panel - controls
            with dpg.child_window(width=control_panel_width, height=-1):
                dpg.add_text("Voronoi Parameters", color=(100, 200, 255))
                dpg.add_separator()

                dpg.add_text("Points (logarithmic, 10 to 100,000)")
                initial_points = int(10 ** params["points_log"])
                dpg.add_slider_float(
                    default_value=params["points_log"],
                    min_value=1.0,
                    max_value=5.0,
                    callback=on_param_change,
                    user_data="points_log",
                    width=-1,
                    format=f"{initial_points} points",
                    clamped=True,
                )

                dpg.add_text("Strength (0-10, exponential)")
                dpg.add_slider_float(
                    default_value=params["strength"],
                    min_value=0,
                    max_value=10,
                    callback=on_param_change,
                    user_data="strength",
                    width=-1,
                )

                dpg.add_text("Blur (1-20)")
                dpg.add_slider_int(
                    default_value=params["blur"],
                    min_value=1,
                    max_value=20,
                    callback=on_param_change,
                    user_data="blur",
                    width=-1,
                )

                dpg.add_separator()
                dpg.add_text("Edge Settings", color=(100, 200, 255))
                dpg.add_separator()

                dpg.add_checkbox(
                    label="Show Edges",
                    default_value=params["show_edges"],
                    callback=on_param_change,
                    user_data="show_edges",
                )

                # Only show debug toggle if debug is enabled
                if debug_enabled:
                    dpg.add_checkbox(
                        label="Debug Regions (show contour colors & points)",
                        default_value=params.get("debug_draw", False),
                        callback=on_param_change,
                        user_data="debug_draw",
                    )

                dpg.add_text("Edge Color")
                dpg.add_color_edit(
                    default_value=list(map(lambda x: x * 255, params["edge_color"])),
                    callback=on_param_change,
                    user_data="edge_color",
                    no_alpha=True,
                    width=-1,
                )

                dpg.add_text("Edge Thickness (0-10 pixels)")
                dpg.add_slider_int(
                    default_value=params["edge_thickness"],
                    min_value=0,
                    max_value=10,
                    callback=on_param_change,
                    user_data="edge_thickness",
                    width=-1,
                )

                dpg.add_separator()
                dpg.add_text("Output Settings", color=(100, 200, 255))
                dpg.add_separator()

                dpg.add_text("Scale (0.1x to 10x, 0.1 increments)")
                initial_output_w = int(w * params["scale"])
                initial_output_h = int(h * params["scale"])
                dpg.add_slider_float(
                    default_value=params["scale"],
                    min_value=0.1,
                    max_value=10.0,
                    callback=on_param_change,
                    user_data="scale",
                    width=-1,
                    format=f"{params['scale']:.1f}x ({initial_output_w}×{initial_output_h})",
                    clamped=True,
                )

                dpg.add_text("Random Seed (0-20)")
                dpg.add_slider_int(
                    default_value=params["seed"],
                    min_value=0,
                    max_value=20,
                    callback=on_param_change,
                    user_data="seed",
                    width=-1,
                )

                dpg.add_separator()
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Save", callback=on_save, width=int(control_panel_width * 0.45))
                    dpg.add_button(label="Quit", callback=on_quit, width=int(control_panel_width * 0.45))

            # Right panel - preview (fills remaining space)
            with dpg.child_window(width=-1, height=-1, tag="preview_panel"):
                dpg.add_text("Preview", color=(100, 200, 255))
                dpg.add_separator()
                # Image - will be resized dynamically
                dpg.add_image(texture_tag, tag=image_tag)

    # Setup and show viewport
    dpg.create_viewport(title="Voronoi Art Preview", width=window_width, height=window_height)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("primary_window", True)

    # Initial preview generation
    update_preview()

    # Main loop with resize handling
    last_resize_size = None

    while dpg.is_dearpygui_running():
        if params["needs_update"]:
            update_preview()

        # Check if preview panel size changed and update image size accordingly
        if dpg.does_item_exist("preview_panel") and dpg.does_item_exist(image_tag):
            panel_size = dpg.get_item_rect_size("preview_panel")
            if panel_size and panel_size[0] > 50 and panel_size[1] > 50:
                # Account for title, separator, and padding
                available_width = int(panel_size[0]) - 20
                available_height = int(panel_size[1]) - 60

                # Calculate scaled size maintaining aspect ratio
                scale = min(available_width / display_w, available_height / display_h)
                new_width = int(display_w * scale)
                new_height = int(display_h * scale)

                # Only update if size changed significantly (avoid constant tiny updates)
                current_size = (new_width, new_height)
                if last_resize_size != current_size:
                    dpg.configure_item(image_tag, width=new_width, height=new_height)
                    last_resize_size = current_size

        dpg.render_dearpygui_frame()

    # Cleanup
    dpg.destroy_context()

    return None


def save_voronoi_output(
    img, output_svg, output_png, points, strength, blur, edge_color=(0, 0, 0), edge_thickness=0.3, scale=1.0, seed=None
):
    """Save Voronoi art to SVG and PNG files."""
    vor, chosen, working_img = compute_voronoi_data(img, points, strength, blur, seed)

    h, w = working_img.shape[:2]

    # SVG
    dwg = svgwrite.Drawing(output_svg, size=(w, h))

    for polygon, color in process_voronoi_regions(vor, chosen, working_img, debug=False):
        # OpenCV colors are BGR, convert to RGB for SVG
        b, g, r = color

        # Only add stroke attributes when edge_thickness > 0 to completely avoid hairlines
        if edge_thickness > 0:
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
    if edge_thickness <= 0:
        # Generate PNG directly from OpenCV to avoid anti-aliasing gaps
        output_img = generate_voronoi_image(img, points, strength, blur, edge_color, 0, seed)

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
    p = argparse.ArgumentParser(description="Adaptive Voronoi vectorization")
    p.add_argument("--input", required=True, help="Input image path (supports PNG, JPG, JPEG, BMP, TIFF, WebP, etc.)")
    p.add_argument("--output", required=False, default=None, help="Output basename (no extension). Defaults to input filename")
    p.add_argument("--preview", action="store_true", help="Interactive preview mode with real-time parameter adjustment")
    p.add_argument("--debug", action="store_true", help="Enable debug mode (shows debug controls in preview)")
    p.add_argument("--points", type=int, default=6000, help="Number of Voronoi sites")
    p.add_argument("--strength", type=float, default=3.0, help="Effect of color variance on density")
    p.add_argument("--blur", type=int, default=2, help="Smoothness of variance map")
    p.add_argument("--edge-color", type=str, default="black", help="Color of Voronoi edges (e.g., 'black', 'white', 'red', '#FF0000')")
    p.add_argument("--edge-thickness", type=float, default=0.3, help="Thickness of Voronoi edges for SVG output")
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
        preview_mode(IMG_PATH, OUTPUT_SVG, OUTPUT_PNG, debug_enabled=args.debug)
        # Preview mode handles saving directly via the Save button
        return

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
        edge_color=edge_color_bgr,
        edge_thickness=args.edge_thickness,
        scale=args.scale,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
