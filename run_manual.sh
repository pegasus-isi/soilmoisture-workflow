#!/bin/bash
set -e

# Simple manual pipeline for Open-Meteo data.

POLYGONS_FILE="polygons.json"
POLYGON_ID="field1"
CROP_TYPE="tomato"
SOIL_TYPE="loam"
START_DATE="2024-12-01"
END_DATE="2025-01-18"

cat > ${POLYGONS_FILE} <<'EOF'
[
  {
    "id": "field1",
    "name": "North Field",
    "coordinates": [[-121.50, 37.50], [-121.48, 37.50], [-121.48, 37.52], [-121.50, 37.52], [-121.50, 37.50]]
  }
]
EOF

echo "=== Step 1: Fetching soil data ==="
./fetch_soil_data.py --fetch \
    --polygons-file ${POLYGONS_FILE} \
    --polygon-id ${POLYGON_ID} \
    --start-date ${START_DATE} \
    --end-date ${END_DATE} \
    --output ${POLYGON_ID}_soil_data.csv

echo "=== Step 2: Analyzing moisture ==="
./bin/analyze_moisture.py \
    --input ${POLYGON_ID}_soil_data.csv \
    --output ${POLYGON_ID}_analysis.json \
    --crop-type ${CROP_TYPE} \
    --soil-type ${SOIL_TYPE}

echo "=== Step 3: Training ML model ==="
./bin/train_model.py \
    --input ${POLYGON_ID}_soil_data.csv \
    --output soil_moisture_model.pt \
    --metadata soil_moisture_model_metadata.json \
    --epochs 50

echo "=== Step 4: Predicting irrigation ==="
./bin/predict_irrigation.py \
    --analysis ${POLYGON_ID}_analysis.json \
    --output ${POLYGON_ID}_prediction.json \
    --model soil_moisture_model.pt \
    --model-metadata soil_moisture_model_metadata.json \
    --soil-data ${POLYGON_ID}_soil_data.csv

echo "=== Step 5: Generating visualization ==="
./bin/visualize_moisture.py \
    --data ${POLYGON_ID}_soil_data.csv \
    --analysis ${POLYGON_ID}_analysis.json \
    --prediction ${POLYGON_ID}_prediction.json \
    --output ${POLYGON_ID}_visualization.png

echo "=== Complete! ==="
echo "Output files:"
ls -la ${POLYGON_ID}_* soil_moisture_model*
