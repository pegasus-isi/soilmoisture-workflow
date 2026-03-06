# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A [Pegasus WMS](https://pegasus.isi.edu/) scientific workflow that fetches hourly soil moisture data from the Open-Meteo ERA5 API, analyzes it per agricultural field (polygon), trains an LSTM model, and produces ML-driven irrigation recommendations. Designed to run on distributed compute resources (ACCESS/FABRIC/HTCondor) using Singularity containers.

## Running the Workflow

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Local end-to-end test (no HTCondor required):**
```bash
./run_manual.sh        # Uses hardcoded test data
./example_usage.sh     # Lightweight Open-Meteo API walkthrough
```

**Generate and submit the Pegasus DAG:**
```bash
# Generate workflow YAML
./workflow_generator.py \
    --polygons-file polygons.json \
    --polygon-ids field1 \
    --start-date 2024-01-01 \
    --end-date 2024-01-31 \
    --crop-type tomato \
    --soil-type loam \
    --output workflow.yml

# Submit to HTCondor via Pegasus
pegasus-plan --submit -s condorpool -o local workflow.yml
pegasus-status <run_directory>
```

**Build and push Docker container:**
```bash
cd Docker
docker buildx build --platform linux/amd64 \
    -f SoilMoisture_Dockerfile \
    -t kthare10/soilmoisture:latest --push .
```

## Architecture

### DAG Structure (fan-out → fan-in)

For each polygon, jobs execute in this order with parallel execution where possible:

```
fetch_soil_data (per polygon)
    → analyze_moisture (per polygon)  +  train_model (once, on primary polygon)
        → predict_irrigation (per polygon)
            → visualize_moisture (per polygon)
```

### Module Responsibilities

| File | Role |
|------|------|
| `workflow_generator.py` | Pegasus DAG builder — defines site/transformation/replica catalogs and job dependencies |
| `fetch_soil_data.py` | Queries Open-Meteo ERA5 archive API using polygon centroid coordinates; outputs hourly CSV |
| `bin/analyze_moisture.py` | Classifies moisture levels using crop/soil thresholds; computes water deficit and trends |
| `bin/train_model.py` | Trains a 2-layer LSTM (seq_len=24, horizon=24) on the primary polygon's data |
| `bin/predict_irrigation.py` | Hybrid ML + rule-based predictor; outputs urgency score (0–100) and irrigation action |
| `bin/visualize_moisture.py` | Multi-panel matplotlib visualization saved as PNG |

### Key Data Flow

**Input:** GeoJSON-like polygon file with `id`, `name`, `coordinates` fields.

**Output per polygon:**
- `{id}_soil_data.csv` — raw hourly data (moisture, temps)
- `{id}_analysis.json` — moisture classification, water deficit, trends
- `{id}_prediction.json` — urgency score, action, hours-until-critical
- `{id}_visualization.png` — multi-panel dashboard

**Shared ML outputs:** `soil_moisture_model.pt`, `soil_moisture_model_metadata.json`

### ML Model

- Architecture: 2-layer LSTM → FC(64→32→24)
- Input features: soil moisture, soil temperature, hour-of-day, day-of-year
- Falls back to rule-based prediction if < 58 data records (insufficient for 10+ sequences)
- Scaler metadata is saved alongside model weights for inference

### Execution Environments

- `condorpool` (default): Standard HTCondor vanilla universe
- DPU mode (`--enable-dpu`): Splits edge-site (fetch) vs cloud-site (compute) for I/O efficiency
- Container: `docker://kthare10/soilmoisture:latest` via Singularity

### Crop & Soil Configuration

Crop thresholds (`wilting`, `stress`, `optimal_low`, `optimal_high`, `saturated`) are defined in `bin/analyze_moisture.py` in `CROP_THRESHOLDS`. Supported: `tomato`, `corn`, `wheat`, `lettuce`, `potato`, `grape`, `alfalfa`, `cotton`. Soil types in `SOIL_CAPACITY`: `sand`, `sandy_loam`, `loam`, `clay_loam`, `clay`.

### Memory Profiles (Pegasus)

- `fetch_soil_data`: 1 GB
- `analyze_moisture`: 1 GB
- `train_model`: 4 GB
- `predict_irrigation`: 2 GB
- `visualize_moisture`: 2 GB
