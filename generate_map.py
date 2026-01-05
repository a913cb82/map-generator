import os
import sys
import argparse
import requests
import math
import numpy as np
import itertools
import gc
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
        if content: return Image.open(BytesIO(content))
    except Exception: pass
    return None

def high_quality_remap(input_channels, input_y, input_x, valid_mask, out_shape):
    """Perform high-quality remapping using bilinear interpolation."""
    output_arr = np.zeros(out_shape, dtype=np.uint8)
    coords = np.stack([input_y, input_x])
    
    for i in range(3):
        # Process one channel at a time in float32 to avoid large internal float64 copies in map_coordinates
        chan_float = input_channels[i].astype(np.float32)
        channel = map_coordinates(chan_float, coords, order=1, mode='constant', cval=0, prefilter=False)
        output_arr[:, :, i] = channel.reshape(out_shape[:2]).astype(np.uint8)
        del chan_float
        
    output_arr[~valid_mask.reshape(out_shape[:2])] = 0
    return output_arr

def get_equirectangular_coords(out_w, out_h, y_start, y_end, source_w, source_h):
    """Compute and cache Equirectangular remapping coordinates."""
    curr_h = y_end - y_start
    max_lat_rad = math.atan(math.sinh(math.pi))
    
    x_coords = np.linspace(0, out_w - 1, out_w, dtype=np.float32)
    y_coords = np.linspace(y_start, y_end - 1, curr_h, dtype=np.float32)
    xv, yv = np.meshgrid(x_coords, y_coords)
    
    lats = (0.5 - yv / (out_h - 1)) * math.pi
    safe_lats = np.clip(lats, -max_lat_rad, max_lat_rad)
    merc_y = np.log(np.tan(math.pi / 4 + safe_lats / 2))
    
    input_y = ((math.pi - merc_y) / (2 * math.pi) * (source_h - 1))
    input_x = (xv / (out_w - 1)) * (source_w - 1)
    valid_mask = np.abs(lats) <= max_lat_rad
    
    return input_y.astype(np.float32), input_x.astype(np.float32), valid_mask

def get_winkel_tripel_coords(out_w, out_h, y_start, y_end, source_w, source_h):
    """Compute and cache Winkel Tripel remapping coordinates using iterative inverse."""
    curr_h = y_end - y_start
    phi1 = math.acos(2.0 / math.pi)
    max_lat_rad = math.atan(math.sinh(math.pi))
    x_max, y_max = 1.0 + math.pi / 2.0, math.pi / 2.0
    
    x_coords = np.linspace(-x_max, x_max, out_w, dtype=np.float32)
    y_coords = np.linspace(y_max - (y_start / (out_h - 1)) * 2 * y_max, 
                           y_max - ((y_end - 1) / (out_h - 1)) * 2 * y_max, 
                           curr_h, dtype=np.float32)
    xv, yv = np.meshgrid(x_coords, y_coords)
    
    lon = (2.0 * xv / (1.0 + math.cos(phi1))).astype(np.float32)
    lat = yv.copy()
    
    def forward(lons, lats):
        lats_c, lons_c = np.clip(lats, -math.pi/2, math.pi/2), np.clip(lons, -math.pi, math.pi)
        alpha = np.arccos(np.clip(np.cos(lats_c) * np.cos(lons_c / 2.0), -1.0, 1.0))
        sinc_inv = np.ones_like(alpha)
        mask = np.abs(alpha) > 1e-10
        sinc_inv[mask] = alpha[mask] / np.sin(alpha[mask])
        fx = 0.5 * (lons * math.cos(phi1) + (2.0 * np.cos(lats) * np.sin(lons / 2.0)) * sinc_inv)
        fy = 0.5 * (lats + np.sin(lats) * sinc_inv)
        return fx, fy

    for _ in range(10):
        cx, cy = forward(lon, lat)
        delta = 1e-6
        x_dlon, y_dlon = forward(lon + delta, lat)
        x_dlat, y_dlat = forward(lon, lat + delta)
        det = ((x_dlon - cx) / delta) * ((y_dlat - cy) / delta) - ((x_dlat - cx) / delta) * ((y_dlon - cy) / delta)
        det[np.abs(det) < 1e-12] = 1e-12
        err_x, err_y = cx - xv, cy - yv
        lon -= (err_x * ((y_dlat - cy) / delta) - err_y * ((x_dlat - cx) / delta)) / det
        lat -= (err_y * ((x_dlon - cx) / delta) - err_x * ((y_dlon - cy) / delta)) / det
        lon, lat = np.clip(lon, -math.pi * 1.1, math.pi * 1.1), np.clip(lat, -math.pi/2 * 1.1, math.pi/2 * 1.1)

    eps = 0.05
    valid_mask = (np.abs(lon) <= math.pi + eps) & (np.abs(lat) <= math.pi/2 + eps) & (np.abs(lat) <= max_lat_rad)
    safe_lat = np.clip(lat, -max_lat_rad, max_lat_rad)
    merc_y = np.log(np.tan(math.pi / 4.0 + safe_lat / 2.0))
    input_y = ((math.pi - merc_y) / (2 * math.pi) * (source_h - 1))
    input_x = (((lon + math.pi) % (2.0 * math.pi)) / (2.0 * math.pi) * (source_w - 1))
    
    return input_y.astype(np.float32), input_x.astype(np.float32), valid_mask

# Cache these expensive coordinate generations
get_equirectangular_coords = memory.cache(get_equirectangular_coords)
get_winkel_tripel_coords = memory.cache(get_winkel_tripel_coords)

def reproject_and_save(input_image, projection, scale, output_path, chunk_size):
    """Reproject image in chunks and save to output_path."""
    w, h = input_image.size
    
    if projection == "mercator":
        if scale != 1.0:
            out_w, out_h = int(w * scale), int(h * scale)
            tqdm.write(f"  Resizing Mercator to {out_w}x{out_h}...")
            # Resize and save without keeping extra copies in memory
            temp_img = input_image.resize((out_w, out_h), Image.LANCZOS)
            temp_img.save(output_path)
            del temp_img
        else:
            input_image.save(output_path)
        gc.collect()
        return

    # Non-mercator: convert to numpy once and process
    input_arr = np.array(input_image).transpose(2, 0, 1) # Shape (3, H, W)
    out_w = int(w * scale)
    out_h = out_w // 2 if projection == "equirectangular" else int(out_w / 1.636)
    
    output_image = Image.new("RGB", (out_w, out_h))
    
    for y_start in tqdm(range(0, out_h, chunk_size), desc=f"  {projection}", leave=False):
        y_end = min(y_start + chunk_size, out_h)
        if projection == "equirectangular":
            iy, ix, mask = get_equirectangular_coords(out_w, out_h, y_start, y_end, w, h)
        else:
            iy, ix, mask = get_winkel_tripel_coords(out_w, out_h, y_start, y_end, w, h)
            
        chunk_arr = high_quality_remap(input_arr, iy, ix, mask, (y_end - y_start, out_w, 3))
        output_image.paste(Image.fromarray(chunk_arr), (0, y_start))
        del iy, ix, mask, chunk_arr
        # gc.collect() is slow, but we are desperate for memory
        gc.collect()

    output_image.save(output_path)
    del output_image, input_arr
    gc.collect()

def get_mercator_canvas(zoom, map_type):
    """Stitch tiles into a full Mercator canvas."""
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if map_type.startswith("google_") and not api_key:
        tqdm.write(f"Error: GOOGLE_MAPS_API_KEY not found in .env for {map_type}"); return None

    num_tiles = 2 ** zoom
    tile_size = 256
    full_size = num_tiles * tile_size
    canvas = Image.new("RGB", (full_size, full_size))
    template = MAP_TEMPLATES[map_type]

    total = num_tiles * num_tiles
    with tqdm(total=total, desc=f"  Downloading {map_type} z{zoom}", leave=False) as pbar:
        for x in range(num_tiles):
            for y in range(num_tiles):
                url = template.format(z=zoom, x=x, y=y, key=api_key)
                tile = download_tile(url)
                if tile: canvas.paste(tile, (x * tile_size, y * tile_size))
                pbar.update(1)
    return canvas

if __name__ == "__main__":
    map_choices, proj_choices = list(MAP_TEMPLATES.keys()), ["mercator", "equirectangular", "winkel_tripel"]
    parser = argparse.ArgumentParser(description="Generate full world maps to .png files.")
    parser.add_argument("zooms", type=int, nargs='+', help="Zoom levels")
    parser.add_argument("--maps", choices=map_choices, nargs='+', default=["esri"], help="Map types")
    parser.add_argument("--projections", choices=proj_choices, nargs='+', default=["mercator"], help="Projections")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor")
    parser.add_argument("--outdir", default=".", help="Output directory")
    parser.add_argument("--chunksize", type=int, default=256, help="Processing chunk size (rows)")
    args = parser.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Process by zoom then map_type to minimize stitching overhead
    for zoom in args.zooms:
        for map_type in args.maps:
            canvas = get_mercator_canvas(zoom, map_type)
            if not canvas: continue
            
            for projection in args.projections:
                clean_map, clean_proj = map_type.replace('_', ''), projection.replace('_', '')
                filename = f"{clean_map}_z{zoom}_{clean_proj}_s{args.scale}.png"
                output_path = outdir / filename
                tqdm.write(f"Processing: {filename}")
                reproject_and_save(canvas, projection, args.scale, output_path, args.chunksize)
            
            del canvas
            gc.collect()