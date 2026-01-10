# Data Sources & Policy

**IMPORTANT**: This repository contains a frozen, reproducible research system. No raw data is hosted here.
To reproduce results, you must obtain data from the official sources listed below.

## 1. Data Policy
- **NO Raw Data**: We do not distribute raw rainfall, complaint, or terrain files.
- **NO PII**: Complaint data must be anonymized before use.
- **NO API Keys**: Users must supply their own credentials if fetch scripts are used.

## 2. Required Datasets

### A. Rainfall Data
- **Source**: National Weather Service / Local Municipal Sensor Network
- **Format**: NetCDF / CSV
- **Required Fields**: `timestamp`, `latitude`, `longitude`, `precipitation_mm`
- **Resolution**: 15-minute intervals recommended.

### B. Terrain (DEM)
- **Source**: USGS EarthExplorer / Copernicus / Local GIS Portal
- **Format**: GeoTIFF (Cloud Optimized prefered)
- **Resolution**: $\le$ 30m resolution (10m recommended).
- **Processing**: Digital Elevation Models must be hydro-conditioned (sinks filled).

### C. Complaint Data (Ground Truth Proxy)
- **Source**: 311 Service Requests / Municipal Drainage Reports
- **Format**: CSV / GeoJSON
- **Required Fields**:
  - `created_date`: Timestamp of report
  - `latitude`, `longitude`: Location
  - `complaint_type`: Category (e.g., "Street Flooding", "Sewer Backup")

## 3. Directory Structure
Place your data in the following structure (these are ignored by git):

```
data/
├── rainfall/
│   └── 2025_01_ storm_data.nc
├── terrain/
│   └── seattle_dem.tif
├── complaints/
│   └── 311_reports_anonymized.csv
└── shapefiles/
    └── city_boundary.geojson
```

## 4. Preprocessing
Run the ingestion pipeline to normalize data:
```bash
python src/main.py --mode ingest --config config/roi_config.yaml
```
