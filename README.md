# Map Generator

A Python script to generate high-quality world maps by stitching together map tiles from various providers and reprojecting them.

## Setup

1. Ensure you have a `.env` file with your `GOOGLE_MAPS_API_KEY` (if using Google Maps layers).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The script supports batch generation of multiple map configurations. It generates the full cross-product of all provided arguments.

```bash
python generate_map.py <zooms...> --maps <map_types...> --projections <projections...> --outdir <directory> --scale <factor>
```

### Arguments

- `zooms`: One or more zoom levels (e.g., `0 1 2`).
- `--maps`: One or more map providers. Use `--help` to see all available types.
- `--projections`: One or more map projections. Use `--help` to see all available projections (e.g., `mercator`, `equirectangular`, `winkel_tripel`).
- `--outdir`: The directory where generated images will be saved (Default: current directory).
- `--scale`: Output resolution scale factor (Default: `1.0`). Use `2.0` for double resolution.

### Auto-Naming

Generated files are automatically named using the pattern:
`{map}_z{zoom}_{projection}_s{scale}.png`
(Underscores are removed from map and projection names in the final filename).

### Examples

```bash
# Generate Esri and OSM maps at zoom level 2 in two projections
python generate_map.py 2 --maps esri osm --projections mercator winkel_tripel --outdir ./output

# Generate a high-resolution Google Terrain map
python generate_map.py 3 --maps google_terrain --projections winkel_tripel --scale 2.0
```

## Features

- **High Quality**: Uses bilinear interpolation for smooth reprojection.
- **Efficient**: Downloads are cached to disk using `joblib`. Reprojection is memory-efficient and processed in chunks.
- **Progress Tracking**: Real-time progress bars for downloads and processing.

## Testing

Run the unit tests:
```bash
python test_generate_map.py
```
