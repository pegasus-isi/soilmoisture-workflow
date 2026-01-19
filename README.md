# Soil Moisture Workflow - Pegasus WMS on FABRIC

A Pegasus workflow system for analyzing soil moisture data from Open-Meteo to answer the question: **"Should I water my crops?"**

## Overview

This workflow fetches soil moisture data for agricultural fields from Open-Meteo, analyzes moisture levels against crop-specific thresholds, and provides irrigation recommendations with urgency scoring.

### Workflow Architecture

```
                                   ┌──────────────────────┐
Open-Meteo API → Fetch Soil Data ──┬── Analyze Moisture ──┬── Predict Irrigation → Visualize
                           │       │         │            │          │                │
                           │       └── Train ML Model ────┘          │                │
                           ↓                 ↓                       ↓                ↓
                    soil_data.csv         model.pt              prediction.json   visualization.png
                                      model_metadata.json
```

### Edge-to-Cloud Architecture (DPU Mode)

With `--enable-dpu`, the workflow distributes processing across edge and cloud:

```
Edge (DPU):   Fetch Soil Data    (I/O intensive, API calls)
                    ↓
Cloud (CPU):  Analyze → Predict → Visualize  (compute intensive)
```

**Benefits:**
- Reduced data transfer to cloud
- Edge preprocessing near data sources
- Better resource utilization

## Features

- **Multi-field support**: Analyze multiple agricultural polygons in parallel
- **Crop-specific thresholds**: Optimized for tomato, corn, wheat, lettuce, potato, grape, alfalfa, cotton
- **Soil type awareness**: Adjusts for sand, loam, clay water holding capacity
- **Irrigation urgency scoring**: 0-100 scale with actionable recommendations
- **Open-Meteo data**: Hourly soil moisture and temperature (ERA5)
- **Visualizations**: Time series plots, urgency gauges, trend indicators
- **Edge-to-Cloud**: Optional DPU-accelerated workflow for FABRIC deployments

### ML-Based Predictions (Required)

- **LSTM Neural Network**: Deep learning model for soil moisture forecasting
  - Sequence-to-sequence prediction (24h history → 24h forecast)
  - Features: soil moisture, temperature, hour of day, day of year
  - Automatic training on fetched historical data
  - **Requires minimum 10 sequences (~58 data records) for training**
- **Predictive Irrigation**: ML-driven irrigation decisions
  - Predicts hours until critical/stress moisture levels
  - Combines ML forecasts with rule-based thresholds
  - **ML model is mandatory** - workflow fails if insufficient data for training

## Prerequisites

### Pegasus/HTCondor Cluster

Before running this workflow, you need a Pegasus/HTCondor cluster. This cluster can be deployed on any infrastructure that supports HTCondor and Pegasus WMS, including cloud providers (AWS, GCP, Azure), on-premises clusters, or research testbeds.

Below we provide an example of how to deploy the cluster on the [FABRIC testbed](https://fabric-testbed.net/):

#### Option 1: FABRIC Artifact (Recommended)

Download and run the pre-configured Pegasus-FABRIC notebook from the FABRIC Artifacts repository:

**Artifact URL**: https://artifacts.fabric-testbed.net/artifacts/53da4088-a175-4f0c-9e25-a4a371032a39

This artifact contains a complete setup for deploying a distributed Pegasus/HTCondor infrastructure across FABRIC sites.

#### Option 2: Jupyter Examples Repository

Use the Pegasus-FABRIC notebook from the official FABRIC Jupyter examples:

**GitHub**: https://github.com/fabric-testbed/jupyter-examples/blob/f7be0c75f22544c72d7b3e3fa42bbdfd9d8bb841/fabric_examples/complex_recipes/pegasus/pegasus-fabric.ipynb

#### Cluster Architecture

The notebook provisions:
- **Submit Node**: Central Manager running HTCondor scheduler and Pegasus WMS
- **Worker Nodes**: Distributed execution points across multiple FABRIC sites
- **FABNetv4 Networking**: Private L3 network connecting all nodes

#### Setup Steps

1. Log into the [FABRIC JupyterHub](https://jupyter.fabric-testbed.net/)
2. Upload/clone the Pegasus-FABRIC notebook
3. Configure your desired sites and node specifications
4. Run the notebook to provision the cluster
5. SSH to the submit node to run workflows

### Software Requirements

- Python 3.9+
- Pegasus WMS v5.0+
- HTCondor v10.2+
- Docker or Singularity

### API Key

**Open-Meteo**: No API key required for standard usage.

### Python Dependencies

```bash
pip install -r requirements.txt
```

## Directory Structure

```
soilmoisture-workflow/
├── workflow_generator.py          # Unified workflow generator (standard + DPU)
├── fetch_soil_data.py             # Open-Meteo data fetcher
├── example_usage.sh               # Example usage script
├── requirements.txt               # Python dependencies
├── bin/
│   ├── analyze_moisture.py        # Moisture analysis with crop thresholds
│   ├── train_model.py             # LSTM model training for moisture prediction
│   ├── predict_irrigation.py      # Irrigation prediction (ML + rules)
│   └── visualize_moisture.py      # Multi-panel visualization
├── Docker/
│   └── SoilMoisture_Dockerfile    # Multi-platform container
├── output/                        # Workflow outputs
└── README.md
```

## Quick Start

### 1. Create a Polygon File

Define your agricultural field as a polygon in a JSON file:

```json
[
  {
    "id": "field1",
    "name": "Evergreen San Jose Field (CA)",
    "coordinates": [[-121.50, 37.50], [-121.48, 37.50], [-121.48, 37.52], [-121.50, 37.52], [-121.50, 37.50]]
  }
]
```

### 3. Generate Workflow

#### Standard Mode (Single Execution Site)

```bash
./workflow_generator.py \
    --polygons-file polygons.json \
    --polygon-ids field1 \
    --start-date 2024-01-01 \
    --end-date 2024-01-31 \
    --crop-type tomato \
    --soil-type loam \
    --output workflow.yml
```

#### Edge-to-Cloud DPU Mode

```bash
./workflow_generator.py \
    --polygons-file polygons.json \
    --polygon-ids field1 \
    --start-date 2024-01-01 \
    --enable-dpu \
    --edge-site edgepool \
    --cloud-site cloudpool \
    --output workflow.yml
```

### 4. Submit to HTCondor

```bash
# Standard mode
pegasus-plan --submit -s condorpool -o local workflow.yml

# DPU mode
pegasus-plan --submit -s edgepool -s cloudpool -o local workflow.yml

# Monitor workflow
pegasus-status <run_directory>
```

### 5. View Results

Results are saved to the `output/` directory:

```
output/
├── <polygon_id>_soil_data.csv           # Raw soil moisture data
├── <polygon_id>_analysis.json           # Moisture analysis results
├── <polygon_id>_prediction.json         # Irrigation recommendations
├── <polygon_id>_visualization.png       # Multi-panel visualization
├── soil_moisture_model.pt               # Trained LSTM model
└── soil_moisture_model_metadata.json    # Model configuration and scalers
```

## Configuration Options

### Workflow Generator Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--polygons-file` | JSON file with polygon definitions | - |
| `--polygon-ids` | Polygon IDs to analyze (optional) | - |
| `--start-date` | Start date YYYY-MM-DD | 30 days ago |
| `--end-date` | End date YYYY-MM-DD | Today |
| `--crop-type` | Crop for threshold selection | default |
| `--soil-type` | Soil for water capacity | loam |
| `-e, --execution-site-name` | HTCondor pool name | condorpool |
| `-o, --output` | Output workflow file | workflow.yml |

### ML Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--ml-epochs` | Number of epochs for LSTM model training | 50 |

### DPU / Edge-to-Cloud Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--enable-dpu` | Enable edge-to-cloud architecture | False |
| `--edge-site` | Edge execution site name | edgepool |
| `--cloud-site` | Cloud execution site name | cloudpool |

### Crop Types

| Crop | Wilting | Stress | Optimal Range | Saturated |
|------|---------|--------|---------------|-----------|
| tomato | 10% | 20% | 25-35% | 45% |
| corn | 8% | 15% | 20-30% | 40% |
| wheat | 8% | 12% | 18-28% | 38% |
| lettuce | 12% | 22% | 28-38% | 45% |
| potato | 10% | 18% | 25-35% | 42% |
| grape | 6% | 12% | 18-28% | 35% |
| alfalfa | 8% | 15% | 22-32% | 40% |
| cotton | 8% | 14% | 20-30% | 38% |

### Soil Types

| Soil | Field Capacity | Wilting Point | Saturation |
|------|----------------|---------------|------------|
| sand | 15% | 5% | 40% |
| sandy_loam | 22% | 8% | 45% |
| loam | 30% | 12% | 48% |
| clay_loam | 35% | 15% | 50% |
| clay | 40% | 20% | 52% |

## Output Files

### 1. Analysis JSON

```json
{
  "metadata": {
    "crop_type": "tomato",
    "soil_type": "loam"
  },
  "polygon_analyses": [
    {
      "polygon_id": "abc123",
      "moisture_classification": "low",
      "moisture_stats": {
        "current": 0.22,
        "mean": 0.25,
        "min": 0.18,
        "max": 0.32
      },
      "water_deficit": {
        "deficit_from_optimal": 0.08,
        "saturation_percent": 55.6
      },
      "trend": {
        "trend": "decreasing",
        "daily_change_rate": -0.005
      }
    }
  ]
}
```

### 2. Prediction JSON

```json
{
  "metadata": {
    "ml_enabled": true
  },
  "predictions": [
    {
      "polygon_id": "abc123",
      "urgency_score": 75,
      "action": "irrigate_soon",
      "recommended_timing": "within 6 hours",
      "irrigation_amount_mm": 15.5,
      "reason": "Soil moisture below optimal range; ML predicts stress in 8h",
      "ml_insights": {
        "ml_prediction_available": true,
        "predicted_min_moisture": 0.15,
        "predicted_trend": "decreasing",
        "hours_until_critical": 18,
        "hours_until_stress": 8
      }
    }
  ],
  "summary": {
    "need_immediate_irrigation": 0,
    "need_scheduled_irrigation": 1,
    "optimal_no_action": 0,
    "ml_predictions_made": 1,
    "overall_action": "SCHEDULE: Some fields need irrigation soon"
  }
}
```

### 3. Visualization

Multi-panel PNG showing:
- **Time series**: Soil moisture over time with threshold bands
- **Urgency gauge**: 0-100 scale with color coding
- **Water status**: Current vs optimal vs field capacity
- **Trend indicator**: Rising/falling/stable
- **Recommendations**: Action, timing, and amount

## Advanced Usage

### Multi-Polygon Analysis

```bash
./workflow_generator.py \
    --polygons-file polygons.json \
    --polygon-ids polygon1 polygon2 polygon3 \
    --crop-type corn \
    --soil-type sandy_loam \
    --output workflow.yml
```

### Custom Docker Container

Build and push multi-platform Docker container:

```bash
cd Docker

# Build for both x86_64
docker buildx build --platform linux/amd64 \
    -f SoilMoisture_Dockerfile \
    -t kthare10/soilmoisture:latest --push .
```

Use custom container:

```bash
./workflow_generator.py \
    --polygons-file polygons.json \
    --polygon-ids field1 \
    --container-image myregistry/soilmoisture:v2 \
    --output workflow.yml
```

## Helper Scripts

For quick local testing, use the provided scripts:

- `example_usage.sh` shows a lightweight Open-Meteo workflow walkthrough.
- `run_manual.sh` runs the full manual pipeline end-to-end.

## Troubleshooting

### Common Issues

**1. No data returned from Open-Meteo**
- Verify polygon ID exists in your polygons file
- Check date range has available ERA5 data

**2. Workflow fails at fetch step**
- Check network connectivity
- Ensure the cluster can reach Open-Meteo endpoints
- Ensure end-date is not in the future (archive API only)

**3. ML training fails with "Insufficient data"**
- The LSTM model requires at least 58 data records (~10 sequences)
- Use a longer date range: `--start-date` should be at least 60 days before `--end-date`
- Try more recent dates - historical data may not be available for older periods
- Example: `--start-date 2024-11-01 --end-date 2025-01-18`

**4. Low irrigation urgency despite dry conditions**
- Verify correct crop type selected
- Check soil type matches actual conditions
- Review threshold settings

**5. DPU jobs not running**
- Verify edge workers have `+has_dpu = True` ClassAd
- Check HTCondor pool configuration
- Review Pegasus site catalog settings

### Debugging

```bash
# View Pegasus logs
pegasus-analyzer <run_directory>

# Check HTCondor job logs
condor_q -analyze <job_id>

# View job stderr/stdout
cat <run_directory>/work/<job_name>/*.err
cat <run_directory>/work/<job_name>/*.out
```

## Irrigation Recommendations

The workflow produces actionable recommendations:

| Urgency Score | Action | Timing | Description |
|---------------|--------|--------|-------------|
| 100 | irrigate_immediately | within 6 hours | Critical - crop damage likely |
| 80 | irrigate_soon | within 6 hours | Plants experiencing stress |
| 50 | schedule_irrigation | within 24 hours | Below optimal, declining |
| 20 | monitor | not needed | Optimal range |
| 0-10 | no_irrigation | not needed | Adequate or saturated |

## Related Resources

- [Open-Meteo API](https://open-meteo.com/)
- [Pegasus WMS Documentation](https://pegasus.isi.edu/documentation/)
- [FABRIC Testbed](https://portal.fabric-testbed.net/)

## Citation

```
@misc{soilmoisture-workflow,
  title={Soil Moisture Analysis and Irrigation Workflow using Pegasus WMS},
  year={2025},
  publisher={GitHub},
  url={https://github.com/pegasus-isi/soilmoisture-workflow}
}
```

## License

This workflow is released under the same license as the parent repository.

## Contributing

Contributions welcome! Please submit issues or pull requests for:
- Additional crop types
- Enhanced ML prediction models
- Alternative data sources
- Performance improvements

---
## Authors
Komal Thareja (kthare10@renci.org)

Built with the assistance of Claude.
