import os
import sys
import argparse
import requests
import math
import numpy as np
import itertools
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from tqdm import tqdm
from joblib import Memory
from scipy.ndimage import map_coordinates
from pathlib import Path

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
        if e.response.status_code == 404:
            return None
        raise e
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        raise e

def download_tile(url):
    try:
        content = download_tile_content(url)
        if content:
            return Image.open(BytesIO(content))
    except Exception:
        pass
    return None

def high_quality_remap(input_arr, input_y, input_x, valid_mask, out_shape):
    """
    Perform high-quality remapping using bilinear interpolation.
    """
    output_arr = np.zeros(out_shape, dtype=np.uint8)
    coords = np.array([input_y, input_x])
    
    for i in range(3):
        channel = map_coordinates(input_arr[:, :, i], coords, order=1, mode='constant', cval=0)
        output_arr[:, :, i] = channel.reshape(out_shape[:2])
        
    output_arr[~valid_mask.reshape(out_shape[:2])] = 0
    return output_arr

def reproject_to_equirectangular(mercator_image, scale=1.0):
    w, h = mercator_image.size
    out_w = int(w * scale)
    out_h = out_w // 2
    
    input_arr = np.array(mercator_image)
    max_lat_rad = math.atan(math.sinh(math.pi))
    
    output_arr = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    chunk_size = 1024
    
    for y_start in tqdm(range(0, out_h, chunk_size), desc="  Reprojecting Equirectangular", leave=False):
        y_end = min(y_start + chunk_size, out_h)
        curr_chunk_h = y_end - y_start
        
        x_coords = np.linspace(0, out_w - 1, out_w)
        y_coords = np.linspace(y_start, y_end - 1, curr_chunk_h)
        xv, yv = np.meshgrid(x_coords, y_coords)
        
        lats = (0.5 - yv / (out_h - 1)) * math.pi
        safe_lats = np.clip(lats, -max_lat_rad, max_lat_rad)
        merc_y = np.log(np.tan(math.pi / 4 + safe_lats / 2))
        
        input_y = ((math.pi - merc_y) / (2 * math.pi) * (h - 1))
        input_x = (xv / (out_w - 1)) * (w - 1)
        valid_mask = np.abs(lats) <= max_lat_rad
        
        chunk_out = high_quality_remap(input_arr, input_y.ravel(), input_x.ravel(), valid_mask.ravel(), (curr_chunk_h, out_w, 3))
        output_arr[y_start:y_end, :, :] = chunk_out
        
    return Image.fromarray(output_arr)

def reproject_to_winkel_tripel(mercator_image, scale=1.0):
    w, h = mercator_image.size
    out_w = int(w * scale)
    out_h = int(out_w / 1.636)
    
    input_arr = np.array(mercator_image)
    phi1 = math.acos(2.0 / math.pi)
    max_lat_rad = math.atan(math.sinh(math.pi))
    
    output_arr = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    x_max = 1.0 + math.pi / 2.0
    y_max = math.pi / 2.0
    
    chunk_size = 512
    
    def forward(lons, lats):
        lats_c = np.clip(lats, -math.pi/2, math.pi/2)
        lons_c = np.clip(lons, -math.pi, math.pi)
        alpha = np.arccos(np.clip(np.cos(lats_c) * np.cos(lons_c / 2.0), -1.0, 1.0))
        sinc_inv = np.ones_like(alpha)
        mask = np.abs(alpha) > 1e-10
        sinc_inv[mask] = alpha[mask] / np.sin(alpha[mask])
        fx = 0.5 * (lons * math.cos(phi1) + (2.0 * np.cos(lats) * np.sin(lons / 2.0)) * sinc_inv)
        fy = 0.5 * (lats + np.sin(lats) * sinc_inv)
        return fx, fy

    for y_start in tqdm(range(0, out_h, chunk_size), desc="  Reprojecting Winkel Tripel", leave=False):
        y_end = min(y_start + chunk_size, out_h)
        curr_chunk_h = y_end - y_start
        
        x_coords = np.linspace(-x_max, x_max, out_w)
        y_coords = np.linspace(y_max - (y_start / (out_h - 1)) * 2 * y_max, 
                               y_max - ((y_end - 1) / (out_h - 1)) * 2 * y_max, 
                               curr_chunk_h)
        xv, yv = np.meshgrid(x_coords, y_coords)
        
        lon = 2.0 * xv / (1.0 + math.cos(phi1))
        lat = yv
        
        for _ in range(10):
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
            lon -= (cx - xv) * dy_dlat / det - (cy - yv) * dx_dlat / det
            lat -= (cy - yv) * dx_dlon / det - (cx - xv) * dy_dlon / det
            lon = np.clip(lon, -math.pi * 1.1, math.pi * 1.1)
            lat = np.clip(lat, -math.pi/2 * 1.1, math.pi/2 * 1.1)

        eps = 0.05
        valid_mask = (np.abs(lon) <= math.pi + eps) & (np.abs(lat) <= math.pi/2 + eps)
        valid_mask &= (np.abs(lat) <= max_lat_rad)
        
        safe_lat = np.clip(lat, -max_lat_rad, max_lat_rad)
        merc_y = np.log(np.tan(math.pi / 4.0 + safe_lat / 2.0))
        input_y = ((math.pi - merc_y) / (2 * math.pi) * (h - 1))
        input_x = (((lon + math.pi) % (2.0 * math.pi)) / (2.0 * math.pi) * (w - 1))
        
        chunk_out = high_quality_remap(input_arr, input_y.ravel(), input_x.ravel(), valid_mask.ravel(), (curr_chunk_h, out_w, 3))
        output_arr[y_start:y_end, :, :] = chunk_out
            
    return Image.fromarray(output_arr)

def generate_map(zoom, map_type, projection, output_path, scale=1.0):
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if map_type.startswith("google_") and not api_key:
        tqdm.write(f"Error: GOOGLE_MAPS_API_KEY not found in .env, but required for {map_type}")
        return

    num_tiles = 2 ** zoom
    tile_size = 256
    full_size = num_tiles * tile_size

    canvas = Image.new("RGB", (full_size, full_size))
    template = MAP_TEMPLATES[map_type]

    total_tiles = num_tiles * num_tiles
    with tqdm(total=total_tiles, desc=f"  Downloading {map_type} z{zoom}", leave=False) as pbar:
        for x in range(num_tiles):
            for y in range(num_tiles):
                url = template.format(z=zoom, x=x, y=y, key=api_key)
                tile = download_tile(url)
                if tile:
                    canvas.paste(tile, (x * tile_size, y * tile_size))
                pbar.update(1)

    if projection == "equirectangular":
        canvas = reproject_to_equirectangular(canvas, scale=scale)
    elif projection == "winkel_tripel":
        canvas = reproject_to_winkel_tripel(canvas, scale=scale)

    canvas.save(output_path)

if __name__ == "__main__":
    map_choices = list(MAP_TEMPLATES.keys())
    proj_choices = ["mercator", "equirectangular", "winkel_tripel"]
    
    parser = argparse.ArgumentParser(
        description="Generate full world maps to .png files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available map types:\n  {', '.join(map_choices)}\n\nAvailable projections:\n  {', '.join(proj_choices)}"
    )
    parser.add_argument("zooms", type=int, nargs='+', help="Zoom levels (e.g., 0 1 2)")
    parser.add_argument("--maps", choices=map_choices, nargs='+', default=["esri"], help="Map types to draw")
    parser.add_argument("--projections", choices=proj_choices, nargs='+', default=["mercator"], help="Map projections to use")
    parser.add_argument("--scale", type=float, default=1.0, help="Output resolution scale factor (default: 1.0)")
    parser.add_argument("--outdir", default=".", help="Output directory (default: current directory)")

    args = parser.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    tasks = list(itertools.product(args.maps, args.zooms, args.projections))
    
    with tqdm(tasks, desc="Total Progress") as pbar:
        for map_type, zoom, projection in pbar:
            clean_map = map_type.replace('_', '')
            clean_proj = projection.replace('_', '')
            filename = f"{clean_map}_z{zoom}_{clean_proj}_s{args.scale}.png"
            output_path = outdir / filename
            pbar.set_postfix(file=filename)
            generate_map(zoom, map_type, projection, output_path, scale=args.scale)
