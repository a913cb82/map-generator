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
    # Mercator y goes from -max_lat_rad to max_lat_rad (normalized to h)
    
    lats = np.linspace(math.pi / 2, -math.pi / 2, out_h)
    
    # Mercator formula: y = ln(tan(pi/4 + lat/2))
    # We clip to max_lat_rad because Mercator is undefined at poles
    clipped_lats = np.clip(lats, -max_lat_rad, max_lat_rad)
    merc_y = np.log(np.tan(math.pi / 4 + clipped_lats / 2))
    
    # Map merc_y (from -pi to pi) to input pixel coordinates (from h-1 to 0)
    # Note: merc_y = pi is top (y=0), merc_y = -pi is bottom (y=h-1)
    input_y = ((math.pi - merc_y) / (2 * math.pi) * (h - 1)).astype(int)
    
    # X mapping is linear (both are longitude)
    input_x = np.arange(out_w)
    
    # Fill output
    for y in range(out_h):
        # Fill black if outside Mercator range (poles)
        if abs(lats[y]) > max_lat_rad:
            continue
        output_arr[y, :, :] = input_arr[input_y[y], input_x, :]
        
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

    canvas.save(output_file)
    print(f"Map saved to {output_file}")

if __name__ == "__main__":
    map_choices = list(MAP_TEMPLATES.keys())
    proj_choices = ["mercator", "equirectangular"]
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
