#!/usr/bin/env python3

"""
Soil Moisture Workflow Generator for Pegasus WMS.

Creates a workflow for:
1. Fetching soil moisture data from Open-Meteo
2. Analyzing moisture levels with crop-specific thresholds
3. Predicting irrigation needs using ML/rule-based methods
4. Generating visualizations and recommendations

Supports standard single-site execution and edge-to-cloud DPU architecture.

Usage:
    # Open-Meteo workflow (polygon file)
    ./workflow_generator.py --polygons-file polygons.json \
        --polygon-ids field1 field2 --output workflow.yml

    # Edge-to-cloud with DPU
    ./workflow_generator.py --polygon-ids abc123 --enable-dpu \
        --edge-site edgepool --cloud-site cloudpool --output workflow.yml

    # A Slurm cluster:
    ./workflow_generator.py ... --site-style slurm --queue cpu --project my_lab

Sites: the workflow names no scheduler. Jobs state cores, memory and a
wall-clock runtime, and plan against a site named "compute". A hosted site
catalog (-s unity.yml, ...) defines it; otherwise custom_sites.py writes it to
sites.yml as an HTCondor pool, or for Slurm via --site-style slurm --queue
--project.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from Pegasus.api import *

# Site-catalog handling. Keeps this generator free of scheduler details.
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from custom_sites import (  # noqa: E402
    HOSTED_SITE, STYLES, ensure_sites_yml, hosted_catalog, is_batch_site, parse_profile,
    parse_tag_profile, worker_package_url,
)

# The container's OS, for the Pegasus worker package staged into it. Must
# run on the image's base (Apptainer/SoilMoisture_Container.def:
# python:3.8.12-slim, Debian 11, glibc 2.31), not the submit host. Pegasus
# publishes no Debian 11 package, so use the oldest-glibc one, rhel_8
# (glibc 2.28).
CONTAINER_PLATFORM = "x86_64_rhel_8"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SoilMoistureWorkflow:
    """Generate Pegasus workflow for soil moisture analysis and irrigation prediction."""

    wf = None
    tc = None
    rc = None
    props = None

    dagfile = None
    wf_dir = None
    shared_scratch_dir = None
    local_storage_dir = None
    wf_name = "soilmoisture"
    # Set by main() from the site in use (see the site-catalog block there).
    worker_package_url = None
    bind_workflow_dir = False

    def __init__(self, dagfile="workflow.yml"):
        """Initialize workflow."""
        self.dagfile = dagfile
        self.wf_dir = str(Path(__file__).parent.resolve())
        self.shared_scratch_dir = os.path.join(self.wf_dir, "scratch")
        self.local_storage_dir = os.path.join(self.wf_dir, "output")

    def write(self):
        """Write all catalogs and workflow to files."""
        self.props.write()
        self.rc.write()
        self.tc.write()
        self.wf.write(file=self.dagfile)

    def create_pegasus_properties(self, sites_yml="sites.yml",
                                  bypass_input_staging=False,
                                  hosted_site_catalog=None):
        """Create Pegasus properties configuration."""
        self.props = Properties()
        self.props["pegasus.transfer.threads"] = "16"
        if hosted_site_catalog:
            # A centrally hosted site catalog (pegasushub
            # pegasus-site-catalogs) defines the execution site; pegasus-plan
            # downloads and caches it, and merges sites.yml over it.
            self.props["pegasus.catalog.site.repo.file"] = hosted_site_catalog
        # Symlink rather than copy when an input already sits on the
        # execution site. A no-op otherwise, so always on.
        self.props["pegasus.transfer.links"] = "true"
        if bypass_input_staging:
            # Jobs read inputs (notably the .sif) straight from the submit
            # host's paths. Only valid where workers share a filesystem with
            # the submit host (a Slurm cluster) — never on a condor pool
            # staging over HTCondor file transfer.
            self.props["pegasus.transfer.bypass.input.staging"] = "true"
        if self.worker_package_url:
            # Stage the container-compatible worker package named in the TC
            # (pegasus::worker) and never download one from inside a job:
            # the image may lack curl/wget, workers may lack internet, and the
            # submit host's kickstart may need a newer glibc than the image.
            self.props["pegasus.transfer.worker.package"] = "true"
            self.props["pegasus.transfer.worker.package.strict"] = "false"
            self.props["pegasus.transfer.worker.package.autodownload"] = "false"
        if os.path.isfile(sites_yml):
            # Lets pegasus-plan find sites.yml from any directory.
            self.props["pegasus.catalog.site"] = "YAML"
            self.props["pegasus.catalog.site.file"] = os.path.abspath(sites_yml)

    def create_replica_catalog(self):
        """Create replica catalog for input files."""
        logger.info("Creating replica catalog")
        self.rc = ReplicaCatalog()

    def create_transformation_catalog(
        self,
        container_sif="Apptainer/SoilMoisture_Container.sif",
    ):
        """Create transformation catalog with executables and containers.

        Nothing here names an execution site: the scripts and the .sif live
        on "local" (the submit host) and are shipped to wherever the job
        runs. Each tool states memory and a wall-clock runtime (seconds) —
        batch sites (Slurm) refuse or kill jobs without one."""
        logger.info("Creating transformation catalog")
        self.tc = TransformationCatalog()

        # Container - a local Apptainer .sif built with `apptainer build`.
        # Pegasus stages the file like any other input, so image_site is the
        # site where the .sif physically lives (the submit host = "local").
        sif_path = (
            container_sif
            if os.path.isabs(container_sif)
            else os.path.join(self.wf_dir, container_sif)
        )
        if not os.path.exists(sif_path):
            logger.warning(
                "Apptainer image not found at %s — build it first with: "
                "apptainer build %s Apptainer/SoilMoisture_Container.def",
                sif_path,
                sif_path,
            )
        soilmoisture_container = Container(
            "soilmoisture_container",
            container_type=Container.SINGULARITY,
            image="file://" + sif_path,
            image_site="local",
        )
        if self.bind_workflow_dir:
            # Batch sites stage inputs as symlinks into the workflow
            # directory, and PegasusLite starts containers with --no-home, so
            # without this bind every job fails with kickstart "Unable to
            # execute the specified binary" (exit 127). Never on a condor
            # pool: the directory does not exist on its workers.
            soilmoisture_container.add_pegasus_profile(
                container_arguments=f"--bind {self.wf_dir}")

        # Add transformations
        fetch_soil_data = Transformation(
            "fetch_soil_data",
            site="local",
            pfn=os.path.join(self.wf_dir, "fetch_soil_data.py"),
            is_stageable=True,
            container=soilmoisture_container,
        ).add_pegasus_profile(memory="1 GB", runtime=1800)

        analyze_moisture = Transformation(
            "analyze_moisture",
            site="local",
            pfn=os.path.join(self.wf_dir, "bin/analyze_moisture.py"),
            is_stageable=True,
            container=soilmoisture_container,
        ).add_pegasus_profile(memory="1 GB", runtime=1800)

        train_model = Transformation(
            "train_model",
            site="local",
            pfn=os.path.join(self.wf_dir, "bin/train_model.py"),
            is_stageable=True,
            container=soilmoisture_container,
        ).add_pegasus_profile(memory="4 GB", runtime=3600)

        predict_irrigation = Transformation(
            "predict_irrigation",
            site="local",
            pfn=os.path.join(self.wf_dir, "bin/predict_irrigation.py"),
            is_stageable=True,
            container=soilmoisture_container,
        ).add_pegasus_profile(memory="2 GB", runtime=1800)

        visualize_moisture = Transformation(
            "visualize_moisture",
            site="local",
            pfn=os.path.join(self.wf_dir, "bin/visualize_moisture.py"),
            is_stageable=True,
            container=soilmoisture_container,
        ).add_pegasus_profile(memory="2 GB", runtime=1800)

        transformations = [
            fetch_soil_data,
            analyze_moisture,
            train_model,
            predict_irrigation,
            visualize_moisture,
        ]
        if self.worker_package_url:
            transformations.append(
                Transformation(
                    "worker",
                    namespace="pegasus",
                    site="local",
                    pfn=self.worker_package_url,
                    is_stageable=True,
                    arch=Arch.X86_64,
                    os_type=OS.LINUX,
                )
            )

        self.tc.add_containers(soilmoisture_container)
        self.tc.add_transformations(*transformations)

    def create_workflow(self, args):
        """Create the complete workflow with ML training."""
        logger.info("Creating workflow")
        self.wf = Workflow(self.wf_name)

        polygons_file = File(os.path.basename(args.polygons_file))
        self.rc.add_replica("local", polygons_file, os.path.abspath(args.polygons_file))

        # Shared ML model files
        model_file = File("soil_moisture_model.pt")
        model_metadata = File("soil_moisture_model_metadata.json")

        # Collect all fetch and analyze jobs
        fetch_jobs = []
        analyze_jobs = []
        soil_data_files = []

        # Add fetch and analyze jobs for each polygon
        for polygon_id in args.polygon_ids:
            fetch_job, analyze_job, soil_data_file = self._add_fetch_analyze_jobs(
                polygon_id, args, polygons_file
            )
            fetch_jobs.append(fetch_job)
            analyze_jobs.append(analyze_job)
            soil_data_files.append(soil_data_file)

        # Add ML training job (uses all soil data files)
        train_job = self._add_train_job(soil_data_files, model_file, model_metadata, args)

        # Training depends on having fetched data
        self.wf.add_dependency(fetch_jobs[0], children=[train_job])

        # Add predict and visualize jobs for each polygon (uses trained model)
        for i, polygon_id in enumerate(args.polygon_ids):
            self._add_predict_visualize_jobs(
                polygon_id, args,
                soil_data_files[i], analyze_jobs[i],
                model_file, model_metadata, train_job
            )

    def _add_fetch_analyze_jobs(self, polygon_id: str, args, polygons_file: File):
        """Add fetch and analyze jobs for a polygon. Returns jobs and data file."""
        soil_data_file = File(f"{polygon_id}_soil_data.csv")
        analysis_file = File(f"{polygon_id}_analysis.json")

        # Job 1: Fetch soil data
        fetch_job = Job("fetch_soil_data", _id=f"fetch_{polygon_id}", node_label=f"fetch_{polygon_id}")
        fetch_job.add_args("--fetch", "--polygon-id", polygon_id)
        fetch_job.add_args("--start-date", args.start_date, "--end-date", args.end_date)
        fetch_job.add_args("--output", soil_data_file)
        fetch_job.add_args("--polygons-file", polygons_file)
        fetch_job.add_inputs(polygons_file)
        fetch_job.add_outputs(soil_data_file, stage_out=True, register_replica=False)
        fetch_job.add_pegasus_profile(label=polygon_id)
        # Retry infrastructure transients (the script retries the API itself).
        fetch_job.add_dagman_profile(retry="2")

        # Job 2: Analyze moisture
        analyze_job = Job("analyze_moisture", _id=f"analyze_{polygon_id}", node_label=f"analyze_{polygon_id}")
        analyze_job.add_args(
            "--input", soil_data_file,
            "--output", analysis_file,
            "--crop-type", args.crop_type,
            "--soil-type", args.soil_type,
        )
        analyze_job.add_inputs(soil_data_file)
        analyze_job.add_outputs(analysis_file, stage_out=True, register_replica=False)
        analyze_job.add_pegasus_profile(label=polygon_id)

        # Add jobs to workflow
        self.wf.add_jobs(fetch_job, analyze_job)
        self.wf.add_dependency(fetch_job, children=[analyze_job])

        return fetch_job, analyze_job, soil_data_file

    def _add_train_job(self, soil_data_files, model_file, model_metadata, args):
        """Add ML model training job."""
        # For training, we use the first polygon's data
        # In a more sophisticated version, we could merge all data
        primary_data_file = soil_data_files[0]

        train_job = Job("train_model", _id="train_ml_model", node_label="train_ml_model")
        train_job.add_args(
            "--input", primary_data_file,
            "--output", model_file,
            "--metadata", model_metadata,
            "--sequence-length", "24",
            "--forecast-horizon", "24",
            "--epochs", str(args.ml_epochs),
        )
        train_job.add_inputs(primary_data_file)
        train_job.add_outputs(model_file, stage_out=True, register_replica=False)
        train_job.add_outputs(model_metadata, stage_out=True, register_replica=False)
        train_job.add_pegasus_profile(label="ml_training")

        self.wf.add_jobs(train_job)

        return train_job

    def _add_predict_visualize_jobs(self, polygon_id: str, args,
                                     soil_data_file, analyze_job,
                                     model_file, model_metadata, train_job):
        """Add predict and visualize jobs for a polygon using trained ML model."""
        analysis_file = File(f"{polygon_id}_analysis.json")
        prediction_file = File(f"{polygon_id}_prediction.json")
        visualization_file = File(f"{polygon_id}_visualization.png")

        # Job 3: Predict irrigation (with ML model)
        predict_job = Job("predict_irrigation", _id=f"predict_{polygon_id}", node_label=f"predict_{polygon_id}")
        predict_job.add_args(
            "--analysis", analysis_file,
            "--output", prediction_file,
            "--model", model_file,
            "--model-metadata", model_metadata,
            "--soil-data", soil_data_file,
        )
        predict_job.add_inputs(analysis_file, model_file, model_metadata, soil_data_file)
        predict_job.add_outputs(prediction_file, stage_out=True, register_replica=False)
        predict_job.add_pegasus_profile(label=polygon_id)

        # Job 4: Visualize
        visualize_job = Job("visualize_moisture", _id=f"visualize_{polygon_id}", node_label=f"visualize_{polygon_id}")
        visualize_job.add_args(
            "--data", soil_data_file,
            "--analysis", analysis_file,
            "--prediction", prediction_file,
            "--output", visualization_file,
        )
        visualize_job.add_inputs(soil_data_file, analysis_file, prediction_file)
        visualize_job.add_outputs(visualization_file, stage_out=True, register_replica=False)
        visualize_job.add_pegasus_profile(label=polygon_id)

        # Add jobs to workflow
        self.wf.add_jobs(predict_job, visualize_job)

        # Dependencies
        self.wf.add_dependency(analyze_job, children=[predict_job])
        self.wf.add_dependency(train_job, children=[predict_job])
        self.wf.add_dependency(predict_job, children=[visualize_job])


def main():
    parser = argparse.ArgumentParser(
        description="Generate Pegasus workflow for soil moisture analysis"
    )

    # Required arguments
    parser.add_argument(
        "--polygon-ids", "-p",
        type=str,
        nargs="+",
        required=False,
        help="Polygon IDs to analyze (optional, uses all polygons if omitted)"
    )
    parser.add_argument(
        "--polygons-file",
        type=str,
        default="polygons.json",
        help="JSON file with polygon definitions (default: polygons.json)"
    )

    # Date range
    parser.add_argument(
        "--start-date",
        type=str,
        default=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
        help="Start date for data fetch (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date for data fetch (YYYY-MM-DD)"
    )

    # Crop and soil parameters
    parser.add_argument(
        "--crop-type",
        type=str,
        default="default",
        choices=["tomato", "corn", "wheat", "lettuce", "potato", "grape", "alfalfa", "cotton", "default"],
        help="Crop type for threshold selection"
    )
    parser.add_argument(
        "--soil-type",
        type=str,
        default="loam",
        choices=["sand", "sandy_loam", "loam", "clay_loam", "clay", "default"],
        help="Soil type for water capacity calculation"
    )

    # ML training parameters
    parser.add_argument(
        "--ml-epochs",
        type=int,
        default=50,
        help="Number of epochs for ML model training"
    )

    # Site options. Every one has a default, so a run with no arguments plans
    # on "compute": the hosted catalog's site when one is named (-s or
    # ~/.pegasusrc), else an HTCondor pool written to sites.yml. dest
    # "execution_site" is what Studio matches to keep its "Where it runs"
    # choice in sync.
    parser.add_argument(
        "-e", "--execution-site", "--execution-site-name",
        dest="execution_site",
        type=str,
        default=HOSTED_SITE,
        help=f"Site to plan against (default: {HOSTED_SITE!r}, the name "
             "hosted catalogs give their site)"
    )
    parser.add_argument(
        "-s", "--hosted-site-catalog", metavar="FILE",
        help="Centrally hosted site catalog to plan against, e.g. unity.yml "
             "(github.com/pegasushub/pegasus-site-catalogs); written to "
             "pegasus.properties. Default: the one named in ~/.pegasusrc, "
             "if any."
    )
    parser.add_argument(
        "--site-style", choices=("auto",) + STYLES + ("none",),
        default="auto",
        help="auto (default): keep a sites.yml entry or hosted catalog if one "
             "exists, else add an HTCondor site. condor/slurm: (re)write this "
             "site's entry. none: leave sites.yml alone."
    )
    parser.add_argument("--queue", metavar="PARTITION",
                        help="Batch partition/queue jobs submit to")
    parser.add_argument("--project", metavar="ACCOUNT",
                        help="Allocation/account charged on a batch site")
    parser.add_argument("--site-scratch", metavar="DIR",
                        help="Slurm: shared scratch visible to workers and "
                             "the submit host (default: ./work)")
    parser.add_argument("--site-profile", action="append", default=[],
                        type=parse_profile, metavar="NS:KEY=VALUE",
                        help="Extra profile on the execution site; repeatable")
    parser.add_argument("--tag-profile", action="append", default=[],
                        type=parse_tag_profile, metavar="TAG:NS:KEY=VALUE",
                        help="Profile for jobs carrying a tag; repeatable")
    parser.add_argument("--shared-filesystem", choices=("auto", "yes", "no"),
                        default="auto",
                        help="Let jobs read inputs directly from the submit "
                             "host. auto: on for Slurm sites, off for HTCondor.")
    parser.add_argument("--sites-yml", metavar="FILE", default="sites.yml",
                        help="Local site catalog (default: sites.yml)")
    parser.add_argument("--skip-sites-catalog", action="store_true",
                        help="Deprecated: same as --site-style none")

    # Container
    parser.add_argument(
        "--container-sif",
        type=str,
        default="Apptainer/SoilMoisture_Container.sif",
        help="Path to the Apptainer .sif image, absolute or relative to the "
             "workflow directory (default: Apptainer/SoilMoisture_Container.sif)"
    )

    # Output
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="workflow.yml",
        help="Output workflow file"
    )

    args = parser.parse_args()

    # Validate dates
    try:
        datetime.strptime(args.start_date, "%Y-%m-%d")
        datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        sys.exit(1)

    # Generate workflow
    try:
        if not args.polygon_ids:
            with open(args.polygons_file, "r") as f:
                data = json.load(f)
            polygons = data.get("polygons", data) if isinstance(data, dict) else data
            args.polygon_ids = [p.get("id") for p in polygons if p.get("id")]
            if not args.polygon_ids:
                logger.error("No polygon IDs found in polygons file")
                sys.exit(1)
        workflow = SoilMoistureWorkflow(dagfile=args.output)

        # Site catalog and the settings that depend on the site
        if args.skip_sites_catalog:
            args.site_style = "none"
        action, style = ensure_sites_yml(
            args.sites_yml, args.execution_site, workflow.wf_dir,
            style=args.site_style, hosted=args.hosted_site_catalog,
            queue=args.queue, project=args.project,
            scratch=args.site_scratch, profiles=args.site_profile,
            tag_profiles=args.tag_profile)
        hosted = hosted_catalog(args.hosted_site_catalog)
        logger.info(f"Site catalog: {args.sites_yml}: {action}"
                    + (f" (merged over hosted {hosted})" if hosted else ""))
        if (style is None and hosted and args.execution_site != "local"
                and args.execution_site != HOSTED_SITE):
            # Nothing was written for this site, so planning works only if
            # the hosted catalog happens to define it.
            logger.warning(
                f"{args.execution_site!r} is not defined in {args.sites_yml} "
                f"and hosted catalogs normally define only {HOSTED_SITE!r}: "
                f"pegasus-plan will fail unless {hosted} has it. Use "
                f"-e {HOSTED_SITE}, or --site-style condor/slurm to describe "
                f"{args.execution_site!r}.")
        if args.shared_filesystem == "auto":
            bypass = style is not None and style != "condor"
        else:
            bypass = args.shared_filesystem == "yes"
        workflow.bind_workflow_dir = bypass or is_batch_site(
            style, args.hosted_site_catalog)
        workflow.worker_package_url = worker_package_url(CONTAINER_PLATFORM)
        if not workflow.worker_package_url:
            logger.warning("pegasus-version not found: Pegasus will choose the "
                           "container's worker package itself (needs curl/wget "
                           "in the image and internet on the workers)")

        workflow.create_pegasus_properties(
            sites_yml=args.sites_yml, bypass_input_staging=bypass,
            hosted_site_catalog=args.hosted_site_catalog)
        workflow.create_replica_catalog()
        workflow.create_transformation_catalog(
            container_sif=args.container_sif
        )
        workflow.create_workflow(args)
        workflow.write()

        logger.info("\n" + "=" * 70)
        logger.info("WORKFLOW GENERATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"  Workflow file: {args.output}")
        logger.info(f"  Polygons: {len(args.polygon_ids)}")
        logger.info(f"  Crop type: {args.crop_type}")
        logger.info(f"  Soil type: {args.soil_type}")
        logger.info(f"  ML training epochs: {args.ml_epochs}")
        logger.info(f"  ML enabled: Yes (LSTM soil moisture prediction)")
        logger.info("\nNext steps:")
        logger.info(f"  1. Review workflow: {args.output}")
        logger.info(f"  2. Submit workflow: pegasus-plan --submit -s {args.execution_site} -o local {args.output}")
        logger.info(f"  3. Monitor status:  pegasus-status <submit_dir>")
        logger.info("=" * 70 + "\n")

    except Exception as e:
        logger.error(f"Failed to generate workflow: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
