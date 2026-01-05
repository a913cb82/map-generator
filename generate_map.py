import os
import sys
import argparse
import requests
import math
import numpy as np
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

MAP_TEMPLATES = {
    "esri": "https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}",
    "osm": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    "osm_hot": "https://a.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
    "google_maps": "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}&key={key}",
    "google_terrain": "https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}&key={key}",
    "google_satellite": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}&key={key}",
    "google_hybrid": "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}&key={key}",
}

def download_tile(url):
    headers = {
        "User-Agent": "MapGenerator/1.0 (Python script for personal use)"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Error downloading tile {url}: {e}")
        return None

def reproject_to_equirectangular(mercator_image):
    """
    Reproject a Mercator image to Equirectangular (Plate Carrée).
    Mercator tiles typically go from -180 to 180 lon and ~-85.05 to ~85.05 lat.
    Equirectangular will be 2:1 aspect ratio covering -180 to 180 and -90 to 90.
    """
    w, h = mercator_image.size
    # Equirectangular aspect ratio is 2:1
    out_w = w
    out_h = w // 2
    
    # Create output array
    input_arr = np.array(mercator_image)
    output_arr = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    
    # Max latitude in Mercator is arctan(sinh(pi)) * 180 / pi approx 85.0511
    max_lat_rad = math.atan(math.sinh(math.pi))
    
    # Precompute y-mapping
    # Equirectangular y goes from -pi/2 to pi/2 (normalized to out_h)
    lats = np.linspace(math.pi / 2, -math.pi / 2, out_h)
    
    # Map lon/lat to Mercator pixels
    valid_mask = np.abs(lats) <= max_lat_rad
    
    # Clip lat to avoid log(tan(0 or pi/2)) errors
    safe_lat = np.clip(lats, -max_lat_rad, max_lat_rad)
    merc_y = np.log(np.tan(math.pi / 4 + safe_lat / 2))
    
    # Map merc_y to input pixel coordinates
    input_y_vals = ((math.pi - merc_y) / (2 * math.pi) * (h - 1)).astype(int)
    # longitude is linear in both
    input_x_vals = np.arange(out_w)
    
    # Fill output
    for y in range(out_h):
        if not valid_mask[y]:
            continue
        output_arr[y, :, :] = input_arr[input_y_vals[y], input_x_vals, :]
        
    return Image.fromarray(output_arr)

def reproject_to_winkel_tripel(mercator_image):
    """
    Reproject a Mercator image to Winkel Tripel.
    Winkel Tripel is the arithmetic mean of Equirectangular and Aitoff.
    """
    w, h = mercator_image.size
    # Aspect ratio of Winkel Tripel is roughly 1.636
    out_w = w
    out_h = int(w / 1.636)
    
    input_arr = np.array(mercator_image)
    output_arr = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    
    phi1 = math.acos(2.0 / math.pi)
    max_lat_rad = math.atan(math.sinh(math.pi))
    
    # Target grid in normalized coordinates
    # Winkel Tripel bounds: x in [-(1+pi/2), (1+pi/2)], y in [-pi/2, pi/2]
    x_max = 1.0 + math.pi / 2.0
    y_max = math.pi / 2.0
    
    x_coords = np.linspace(-x_max, x_max, out_w)
    y_coords = np.linspace(y_max, -y_max, out_h)
    xv, yv = np.meshgrid(x_coords, y_coords)
    
    # Iterative inverse
    # Initial guess: Equirectangular
    lon = xv / math.cos(phi1)
    lat = yv
    
    def forward(lons, lats):
        # Clip internally for stability in arccos/sin
        lons_c = np.clip(lons, -math.pi, math.pi)
        lats_c = np.clip(lats, -math.pi/2, math.pi/2)
        
        alpha = np.arccos(np.clip(np.cos(lats_c) * np.cos(lons_c / 2.0), -1, 1))
        sinc_inv = np.ones_like(alpha)
        mask = np.abs(alpha) > 1e-8
        sinc_inv[mask] = alpha[mask] / np.sin(alpha[mask])
        
        fx = 0.5 * (lons_c * math.cos(phi1) + (2.0 * np.cos(lats_c) * np.sin(lons_c / 2.0)) * sinc_inv)
        fy = 0.5 * (lats_c + np.sin(lats_c) * sinc_inv)
        return fx, fy

    # Newton's method
    for _ in range(8): # Increased iterations slightly for better edge convergence
        cx, cy = forward(lon, lat)
        
        # Numerical Jacobian
        delta = 1e-6
        x_dlon, y_dlon = forward(lon + delta, lat)
        x_dlat, y_dlat = forward(lon, lat + delta)
        
        dx_dlon = (x_dlon - cx) / delta
        dx_dlat = (x_dlat - cx) / delta
        dy_dlon = (y_dlon - cy) / delta
        dy_dlat = (y_dlat - cy) / delta
        
        det = dx_dlon * dy_dlat - dx_dlat * dy_dlon
        # Avoid division by zero
        det[np.abs(det) < 1e-12] = 1e-12
        
        err_x = cx - xv
        err_y = cy - yv
        
        lon -= (err_x * dy_dlat - err_y * dx_dlat) / det
        lat -= (err_y * dx_dlon - err_x * dy_dlon) / det

    # Valid mask: point must be within world bounds and within Mercator range
    # Handle NaNs that might arise from divergent iterations
    lon = np.nan_to_num(lon, nan=1e9)
    lat = np.nan_to_num(lat, nan=1e9)
    
    # Use a small epsilon to avoid edge artifacts
    eps = 1e-4
    valid_mask = (np.abs(lon) <= math.pi + eps) & (np.abs(lat) <= math.pi/2 + eps)
    
    # Further restrict to Mercator latitude range
    valid_mask &= (np.abs(lat) <= max_lat_rad)
    
    # Map lon/lat to Mercator pixels
    safe_lat = np.clip(lat, -max_lat_rad, max_lat_rad)
    merc_y = np.log(np.tan(math.pi / 4.0 + safe_lat / 2.0))
    
    input_y = np.zeros_like(merc_y, dtype=int)
    input_x = np.zeros_like(lon, dtype=int)
    
    # Normalize lon for input_x (Mercator is linear in longitude)
    input_y[valid_mask] = ((math.pi - merc_y[valid_mask]) / (2.0 * math.pi) * (h - 1)).astype(int)
    input_x[valid_mask] = (((lon[valid_mask] + math.pi) % (2*math.pi)) / (2.0 * math.pi) * (w - 1)).astype(int)
    
    # Clip to be safe
    input_y = np.clip(input_y, 0, h - 1)
    input_x = np.clip(input_x, 0, w - 1)
    
    # Vectorized fill
    output_arr[valid_mask] = input_arr[input_y[valid_mask], input_x[valid_mask]]
    
    return Image.fromarray(output_arr)

def generate_map(zoom, map_type, projection, output_file):
    if map_type not in MAP_TEMPLATES:
        print(f"Unknown map type: {map_type}")
        return

    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if map_type.startswith("google_") and not api_key:
        print(f"Error: GOOGLE_MAPS_API_KEY not found in .env, but required for {map_type}")
        return

    num_tiles = 2 ** zoom
    tile_size = 256
    full_size = num_tiles * tile_size

    print(f"Generating {map_type} map at zoom level {zoom} with {projection} projection...")
    print(f"Fetching {num_tiles}x{num_tiles} tiles...")

    canvas = Image.new("RGB", (full_size, full_size))
    template = MAP_TEMPLATES[map_type]

    for x in range(num_tiles):
        for y in range(num_tiles):
            url = template.format(z=zoom, x=x, y=y, key=api_key)
            tile = download_tile(url)
            if tile:
                canvas.paste(tile, (x * tile_size, y * tile_size))
            else:
                print(f"Skipping tile {x},{y}")

    if projection == "equirectangular":
        print("Reprojecting to Equirectangular...")
        canvas = reproject_to_equirectangular(canvas)
    elif projection == "winkel_tripel":
        print("Reprojecting to Winkel Tripel...")
        canvas = reproject_to_winkel_tripel(canvas)

    canvas.save(output_file)
    print(f"Map saved to {output_file}")

if __name__ == "__main__":
    map_choices = list(MAP_TEMPLATES.keys())
    proj_choices = ["mercator", "equirectangular", "winkel_tripel"]
    epilog_examples = "\n".join([f"  python generate_map.py 2 --map {m}" for m in map_choices[:2]])
    
    parser = argparse.ArgumentParser(
        description="Generate a full world map to a .png file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available map types:
  {', '.join(map_choices)}

Available projections:
  {', '.join(proj_choices)}

Examples:
{epilog_examples}
  python generate_map.py 2 --projection equirectangular
        """
    )
    parser.add_argument("zoom", type=int, help="Zoom level (e.g., 0 to 5)")
    parser.add_argument("--map", choices=map_choices, default="esri", 
                        help="Map type to draw (default: esri)")
    parser.add_argument("--projection", choices=proj_choices, default="mercator",
                        help="Map projection to use (default: mercator)")
    parser.add_argument("--output", default="world_map.png", help="Output filename (default: world_map.png)")

    args = parser.parse_args()

    generate_map(args.zoom, args.map, args.projection, args.output)
