#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import yaml
from pathlib import Path
import tempfile
import hashlib
import shutil
from typing import List
import re

def convert_arg_name(name: str) -> str:
    """Convert YAML keys to command-line argument format."""
    return f"--{name.replace('-', '-')}"


def get_datasets(config, key) -> List[str]:
    """Extract datasets from the config."""
    if key in config:
        if isinstance(config[key], list):
            return config[key]
        else:
            return [config[key]]
    return []


def validate_datasets(datasets) -> List[str]:
    """Check that datasets exist."""
    for dataset in datasets:
        if not Path(dataset).exists():
            print(f"Error: Dataset {dataset} does not exist")
            sys.exit(1)
    return datasets


def build_command(script_path: str, config, previous_model=None):
    """Build the command to execute based on the config."""
    cmd = [sys.executable, script_path]

    for dataset in validate_datasets(get_datasets(config, "datasets")):
        cmd.append(dataset)

    del config["datasets"]  # Remove datasets from config to avoid conflicts
    
    # Add resume-from-model if we have a previous model
    if previous_model and "resume-from-model" not in config:
        features = config.get("features", None)
        converted_model = prepare_checkpoint_to_pt(previous_model, features)
        config["resume-from-model"] = converted_model

    # Convert config to command-line arguments
    for key, value in config.items():
        if key == "lambda":  # Special case for lambda (reserved keyword)
            cmd.append("--lambda")
            cmd.append(str(value))
        # elif value is True:
        #     cmd.append(convert_arg_name(key))
        # elif value is False:
        #     cmd.append(f"--no-{key.replace('_', '-')}")
        elif value is not None:
            cmd.append(convert_arg_name(key))
            cmd.append(str(value))
        else:
            cmd.append(convert_arg_name(key))

    return cmd


def find_latest_checkpoint(directory):
    """Find the latest checkpoint in a directory based on filename pattern."""
    if not directory.exists():
        return None

    checkpoints = []

    for file in directory.iterdir():
        if file.is_file():
            if file.name.endswith(".ckpt"):
                absolute_path = file.resolve()
                checkpoints.append(absolute_path)

    if not checkpoints:
        return None

    # latest = max(checkpoints, key=lambda x: x[0])
    # return last
    return checkpoints[-1]


def prepare_checkpoint_to_pt(checkpoint_path, features) -> str:
    _, filename = tempfile.mkstemp(suffix=".pt")

    if features is None:
        features = "HalfKAv2_hm^"

    subprocess.run(
        [
            sys.executable,
            "serialize.py",
            checkpoint_path,
            filename,
            f"--features={features}",
        ],
    )

    return filename


def get_output_dir(config) -> Path:
    """Determine the output directory from config or default."""
    if "default_root_dir" in config:
        return Path(config["default_root_dir"]) / Path("lightning_logs")
    return Path("lightning_logs")


def find_latest_model_checkpoint(stage_config) -> str:
    """Find the latest model checkpoint for a given stage configuration."""
    def get_version_number(dir_name):
        """Extract version number from directory name like 'version_1', 'version_123', etc."""
        match = re.search(r'version_(\d+)', dir_name)
        return int(match.group(1)) if match else -1

    output_dir = get_output_dir(stage_config)

    version_dirs = [
        d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("version_")
    ]

    if not version_dirs:
        print(f"Warning: No version directories found in {output_dir}")
        return None

    # Sort version directories by actual version number (highest first)
    version_dirs = sorted(version_dirs, key=lambda d: get_version_number(d.name), reverse=True)

    # append checkpoints to version directories
    version_dirs = [d / Path("checkpoints") for d in version_dirs]

    # iterate through all version directories in reverse order (latest first)
    for version_dir in version_dirs:
        if not version_dir.exists():
            continue

        checkpoints = list(version_dir.glob("*.ckpt"))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            print(f"Found checkpoint: {latest_checkpoint}")
            return str(latest_checkpoint.resolve())


def get_sha256(file_path: Path) -> str:
    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as f:
        # Read the file in chunks to avoid using too much memory
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()


def serialize_cpkt_to_nnue(net_dir: Path, cpkt_path: Path, features: str):
    """Serialize a checkpoint to NNUE format."""
    if not net_dir.exists():
        net_dir.mkdir(parents=True, exist_ok=True)

    if not cpkt_path.exists():
        print(f"Error: Checkpoint file {cpkt_path} does not exist")
        sys.exit(1)

    output_file = Path(tempfile.mktemp(suffix=".nnue"))
    cmd = [
        sys.executable,
        "serialize.py",
        str(cpkt_path),
        str(output_file),
        f"--features={features}",
    ]

    print(f"Serializing checkpoint to NNUE format: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    sha25 = get_sha256(output_file)

    new_output_file = net_dir / f"nn-{sha25[:12]}.nnue"

    shutil.move(str(output_file), str(new_output_file))

    print(f"Serialized NNUE model saved to {new_output_file}")


def run_training(config_file: str, start_stage: int):
    """Run training based on the YAML configuration."""
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    if "training" not in config:
        print("Error: YAML file must contain a 'training' section")
        sys.exit(1)

    script_path = config.get("script_path", "train.py")
    print(f"Using training script: {script_path}")

    stages = config["training"]
    if not stages:
        print("Error: No training stages defined")
        sys.exit(1)

    stage_names = list(stages.keys())

    if start_stage < 0 or start_stage >= len(stage_names):
        print(
            f"Error: Invalid start stage {start_stage}. Must be between 0 and {len(stage_names) - 1}"
        )
        sys.exit(1)

    previous_model = None

    if start_stage > 0:
        previous_stage_name = stage_names[start_stage - 1]
        previous_model = find_latest_model_checkpoint(stages[previous_stage_name])

        if previous_model:
            print(f"Starting from stage {start_stage}: {stage_names[start_stage]}")
            print(f"Using checkpoint from previous stage: {previous_model}")
        else:
            print(f"Starting from stage {start_stage} without a checkpoint")

    # Run each stage in sequence
    for i, stage_name in enumerate(stage_names[start_stage:], start=start_stage):
        stage_config = stages[stage_name]

        print(f"\n{'=' * 30}")
        print(f"Starting training stage: {stage_name}")
        print(f"{'=' * 30}")

        cmd = build_command(script_path, stage_config, previous_model)
        print(f"Executing command: {' '.join(cmd)}")

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(
                f"Error: Stage {stage_name} failed with return code {result.returncode}"
            )
            sys.exit(result.returncode)

        previous_model = find_latest_model_checkpoint(stage_config)
        if previous_model:
            print(f"Found checkpoint for next stage: {previous_model}")
        else:
            print("No checkpoint found for next stage")
            sys.exit(1)

        serialize_cpkt_to_nnue(
            Path(config.get("network_dir", "networks")),
            Path(previous_model),
            stage_config.get("features", "HalfKAv2_hm^"),
        )

    print("\nAll training stages completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run multi-stage training using a YAML configuration"
    )
    parser.add_argument("config", help="YAML configuration file")
    parser.add_argument("--start-stage", type=int, default=0, help="Start stage index")
    args = parser.parse_args()

    run_training(args.config, args.start_stage)
