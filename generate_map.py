import os
import sys
import argparse
import requests
import math
import numpy as np
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from tqdm import tqdm
from joblib import Memory

# Load environment variables from .env
load_dotenv()

# Setup joblib cache in the .cache directory
memory = Memory(".cache", verbose=0)

MAP_TEMPLATES = {
    "esri": "https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}",
    "osm": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    "osm_hot": "https://a.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
    "google_maps": "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}&key={key}",
    "google_terrain": "https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}&key={key}",
    "google_satellite": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}&key={key}",
    "google_hybrid": "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}&key={key}",
}

@memory.cache
def download_tile_content(url):
    headers = {
        "User-Agent": "MapGenerator/1.0 (Python script for personal use)"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.content
    except requests.exceptions.HTTPError as e:
        # 404 is considered permanent enough to cache as 'None'
        if e.response.status_code == 404:
            return None
        # Rate limits (429), auth (401, 403), or server errors (5xx)
        # should NOT be cached. Raise so joblib doesn't store a result.
        raise e
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        # Transient network issues should NOT be cached.
        raise e

def download_tile(url):
    try:
        content = download_tile_content(url)
        if content:
            return Image.open(BytesIO(content))
    except Exception:
        # Errors that weren't cached (raised above) come through here.
        # We allow the loop to continue.
        pass
    return None

def reproject_to_equirectangular(mercator_image):
    """
    Reproject a Mercator image to Equirectangular (Plate Carrée).
    Vectorized row-by-row with progress bar.
    """
    w, h = mercator_image.size
    out_w = w
    out_h = w // 2
    
    input_arr = np.array(mercator_image)
    output_arr = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    
    max_lat_rad = math.atan(math.sinh(math.pi))
    lats = np.linspace(math.pi / 2, -math.pi / 2, out_h)
    
    # Precompute all y-mappings
    safe_lats = np.clip(lats, -max_lat_rad, max_lat_rad)
    merc_y = np.log(np.tan(math.pi / 4 + safe_lats / 2))
    input_y_vals = ((math.pi - merc_y) / (2 * math.pi) * (h - 1)).astype(int)
    valid_mask = np.abs(lats) <= max_lat_rad
    
    input_x = np.arange(out_w)
    
    for y in tqdm(range(out_h), desc="Reprojecting", leave=False):
        if valid_mask[y]:
            output_arr[y, :, :] = input_arr[input_y_vals[y], input_x, :]
            
    return Image.fromarray(output_arr)

def reproject_to_winkel_tripel(mercator_image):
    """
    Reproject a Mercator image to Winkel Tripel.
    Uses iterative inverse with Newton's method.
    """
    w, h = mercator_image.size
    out_w = w
    out_h = int(w / 1.636)
    
    input_arr = np.array(mercator_image)
    output_arr = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    
    phi1 = math.acos(2.0 / math.pi)
    max_lat_rad = math.atan(math.sinh(math.pi))
    
    # Target grid in normalized coordinates
    x_max = 1.0 + math.pi / 2.0
    y_max = math.pi / 2.0
    
    x_coords = np.linspace(-x_max, x_max, out_w)
    y_coords = np.linspace(y_max, -y_max, out_h)
    xv, yv = np.meshgrid(x_coords, y_coords)
    
    # Initial guess: Equirectangular-ish
    lon = xv / math.cos(phi1)
    lat = yv
    
    def forward(lons, lats):
        # Internal clip for trig stability
        lats_c = np.clip(lats, -math.pi/2, math.pi/2)
        lons_c = np.clip(lons, -math.pi, math.pi)
        
        alpha = np.arccos(np.clip(np.cos(lats_c) * np.cos(lons_c / 2.0), -1.0, 1.0))
        sinc_inv = np.ones_like(alpha)
        mask = np.abs(alpha) > 1e-8
        sinc_inv[mask] = alpha[mask] / np.sin(alpha[mask])
        
        fx = 0.5 * (lons * math.cos(phi1) + (2.0 * np.cos(lats_c) * np.sin(lons / 2.0)) * sinc_inv)
        fy = 0.5 * (lats + np.sin(lats_c) * sinc_inv)
        return fx, fy

    # Newton's method
    for _ in tqdm(range(10), desc="Newton iterations", leave=False):
        cx, cy = forward(lon, lat)
        
        delta = 1e-6
        x_dlon, y_dlon = forward(lon + delta, lat)
        x_dlat, y_dlat = forward(lon, lat + delta)
        
        dx_dlon = (x_dlon - cx) / delta
        dx_dlat = (x_dlat - cx) / delta
        dy_dlon = (y_dlon - cy) / delta
        dy_dlat = (y_dlat - cy) / delta
        
        det = dx_dlon * dy_dlat - dx_dlat * dy_dlon
        det[np.abs(det) < 1e-12] = 1e-12
        
        err_x = cx - xv
        err_y = cy - yv
        
        lon -= (err_x * dy_dlat - err_y * dx_dlat) / det
        lat -= (err_y * dx_dlon - err_x * dy_dlon) / det
        
        # Keep iterations within a reasonable world
        lon = np.clip(lon, -math.pi * 1.1, math.pi * 1.1)
        lat = np.clip(lat, -math.pi/2 * 1.1, math.pi/2 * 1.1)

    # Final mask: points that converged within the world
    eps = 0.01
    valid_mask = (np.abs(lon) <= math.pi + eps) & (np.abs(lat) <= math.pi/2 + eps)
    
    # Further restrict to Mercator range
    valid_mask &= (np.abs(lat) <= max_lat_rad)
    
    # Map to Mercator pixels
    safe_lat = np.clip(lat, -max_lat_rad, max_lat_rad)
    merc_y = np.log(np.tan(math.pi / 4.0 + safe_lat / 2.0))
    
    input_y = np.zeros_like(merc_y, dtype=int)
    input_x = np.zeros_like(lon, dtype=int)
    
    input_y[valid_mask] = ((math.pi - merc_y[valid_mask]) / (2.0 * math.pi) * (h - 1)).astype(int)
    input_x[valid_mask] = (((lon[valid_mask] + math.pi) % (2.0 * math.pi)) / (2.0 * math.pi) * (w - 1)).astype(int)
    
    input_y = np.clip(input_y, 0, h - 1)
    input_x = np.clip(input_x, 0, w - 1)
    
    for y in tqdm(range(out_h), desc="Remapping", leave=False):
        row_mask = valid_mask[y, :]
        if np.any(row_mask):
            output_arr[y, row_mask, :] = input_arr[input_y[y, row_mask], input_x[y, row_mask]]
            
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
    print(f"Resulting image size: {full_size}x{full_size} pixels")

    canvas = Image.new("RGB", (full_size, full_size))
    template = MAP_TEMPLATES[map_type]

    total_tiles = num_tiles * num_tiles
    with tqdm(total=total_tiles, desc="Downloading tiles") as pbar:
        for x in range(num_tiles):
            for y in range(num_tiles):
                url = template.format(z=zoom, x=x, y=y, key=api_key)
                tile = download_tile(url)
                if tile:
                    canvas.paste(tile, (x * tile_size, y * tile_size))
                pbar.update(1)

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