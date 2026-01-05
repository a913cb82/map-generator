import os
import sys
import argparse
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

MAP_TEMPLATES = {
    "esri": "https://server.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}",
    "osm": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    "google_terrain": "https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}&key={key}"
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

def generate_map(zoom, map_type, output_file):
    if map_type not in MAP_TEMPLATES:
        print(f"Unknown map type: {map_type}")
        return

    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if map_type == "google_terrain" and not api_key:
        print("Error: GOOGLE_MAPS_API_KEY not found in .env")
        return

    num_tiles = 2 ** zoom
    tile_size = 256
    full_size = num_tiles * tile_size

    print(f"Generating {map_type} map at zoom level {zoom}...")
    print(f"Resulting image size: {full_size}x{full_size} pixels ({num_tiles}x{num_tiles} tiles)")

    canvas = Image.new("RGB", (full_size, full_size))

    template = MAP_TEMPLATES[map_type]

    for x in range(num_tiles):
        for y in range(num_tiles):
            # Format URL based on template
            # Note: Esri uses {z}/{y}/{x}, OSM uses {z}/{x}/{y}, Google uses params
            url = template.format(z=zoom, x=x, y=y, key=api_key)
            
            tile = download_tile(url)
            if tile:
                canvas.paste(tile, (x * tile_size, y * tile_size))
            else:
                # If tile download fails, we leave it black or could fill with a placeholder
                print(f"Skipping tile {x},{y}")

    canvas.save(output_file)
    print(f"Map saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a full world map to a .png file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_map.py 2 --map esri
  python generate_map.py 3 --map osm
  python generate_map.py 2 --map google_terrain
        """
    )
    parser.add_argument("zoom", type=int, help="Zoom level (e.g., 0 to 5)")
    parser.add_argument("--map", choices=MAP_TEMPLATES.keys(), default="esri", 
                        help="Map type to draw (default: esri)")
    parser.add_argument("--output", default="world_map.png", help="Output filename (default: world_map.png)")

    args = parser.parse_args()

    generate_map(args.zoom, args.map, args.output)
