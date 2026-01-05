# Map Generator

A Python script to generate full world maps by stitching together map tiles from various providers.

## Setup

1. Ensure you have a `.env` file with your `GOOGLE_MAPS_API_KEY` (if using Google Maps layers).
2. Install dependencies:
   ```bash
   pip install requests Pillow python-dotenv
   ```

## Usage

Run the script with the desired zoom level and map type.

```bash
python generate_map.py <zoom_level> --map <map_type> --output <filename>
```

### Arguments

- `zoom`: The zoom level (e.g., 0 for a single tile, 1 for 2x2, 2 for 4x4, etc.)
- `--map`: The map provider to use. Use `--help` to see all available types (esri, osm, google_maps, etc.)
- `--output`: The filename for the resulting `.png` (Default: `world_map.png`)

### Examples

See `python generate_map.py --help` for more examples.

```bash
# Generate a NatGeo map at zoom 2
python generate_map.py 2 --map esri

# Generate an OSM map at zoom 3
python generate_map.py 3 --map osm
```