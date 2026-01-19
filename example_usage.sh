#!/bin/bash
#
# Soil Moisture Workflow - Example Usage
#
# This script demonstrates how to use the soil moisture workflow
# for irrigation decision-making.

# ==============================================================================
# Prerequisites
# ==============================================================================
#
# Open-Meteo usage does not require an API key.
#
# 3. Install Python dependencies:
#    pip install pandas numpy matplotlib scipy requests pytz Pegasus-wms
#
# 4. Build Docker container (multi-platform for DPU support):
#    cd Docker
#    docker buildx build --platform linux/amd64,linux/arm64 \
#        -f SoilMoisture_Dockerfile \
#        -t kthare10/soilmoisture:latest --push .

# ==============================================================================
# Step 1: Fetch Soil Data from Open-Meteo (ERA5)
# ==============================================================================

cat > polygons.json <<'EOF'
[
  {
    "id": "field1",
    "name": "North Field",
    "coordinates": [[-121.50, 37.50], [-121.48, 37.50], [-121.48, 37.52], [-121.50, 37.52], [-121.50, 37.50]]
  }
]
EOF

./fetch_soil_data.py --fetch \
    --polygons-file polygons.json \
    --polygon-id field1 \
    --start-date 2024-12-01 \
    --end-date 2025-01-18 \
    --output output/soil_data.csv

# ==============================================================================
# Step 4: Run Analysis Scripts Manually (for testing)
# ==============================================================================

echo "=== Running analysis ==="
./bin/analyze_moisture.py \
    --input output/soil_data.csv \
    --output output/analysis.json \
    --crop-type tomato \
    --soil-type loam

echo "=== Predicting irrigation needs ==="
./bin/predict_irrigation.py \
    --analysis output/analysis.json \
    --output output/prediction.json

echo "=== Creating visualization ==="
./bin/visualize_moisture.py \
    --data output/soil_data.csv \
    --analysis output/analysis.json \
    --prediction output/prediction.json \
    --output output/visualization.png

# ==============================================================================
# Step 5: Generate and Submit Pegasus Workflow
# ==============================================================================

echo "=== Generating workflow ==="

# Standard workflow (single execution site)
./workflow_generator.py \
    --polygons-file polygons.json \
    --polygon-ids field1 \
    --start-date 2024-12-01 \
    --end-date 2025-01-18 \
    --crop-type tomato \
    --soil-type loam \
    --output workflow.yml

# Submit to HTCondor
# pegasus-plan --submit -s condorpool -o local workflow.yml

# ==============================================================================
# Edge-to-Cloud Workflow (DPU Mode)
# ==============================================================================

echo "=== Generating edge-to-cloud workflow ==="

./workflow_generator.py \
    --polygons-file polygons.json \
    --polygon-ids field1 \
    --start-date 2024-12-01 \
    --end-date 2025-01-18 \
    --crop-type tomato \
    --soil-type loam \
    --enable-dpu \
    --edge-site edgepool \
    --cloud-site cloudpool \
    --output workflow_dpu.yml

# Submit edge-to-cloud workflow
# pegasus-plan --submit -s edgepool -s cloudpool -o local workflow_dpu.yml

# ==============================================================================
# Multi-Polygon Workflow
# ==============================================================================

echo "=== Generating multi-polygon workflow ==="

./workflow_generator.py \
    --polygons-file polygons.json \
    --polygon-ids POLYGON_ID_1 POLYGON_ID_2 POLYGON_ID_3 \
    --start-date 2024-12-01 \
    --end-date 2025-01-18 \
    --crop-type corn \
    --soil-type sandy_loam \
    --output workflow_multi.yml

# ==============================================================================
# Monitor Workflow
# ==============================================================================

# pegasus-status <run_directory>
# pegasus-analyzer <run_directory>

# ==============================================================================
# Crop Types Available
# ==============================================================================
#
# tomato   - Optimal: 25-35% moisture
# corn     - Optimal: 20-30% moisture
# wheat    - Optimal: 18-28% moisture
# lettuce  - Optimal: 28-38% moisture (high water needs)
# potato   - Optimal: 25-35% moisture
# grape    - Optimal: 18-28% moisture (drought tolerant)
# alfalfa  - Optimal: 22-32% moisture
# cotton   - Optimal: 20-30% moisture
# default  - Optimal: 25-35% moisture

# ==============================================================================
# Soil Types Available
# ==============================================================================
#
# sand       - Field capacity: 15%, low water holding
# sandy_loam - Field capacity: 22%
# loam       - Field capacity: 30%, balanced
# clay_loam  - Field capacity: 35%
# clay       - Field capacity: 40%, high water holding
# default    - Field capacity: 30%

echo "=== Done ==="
