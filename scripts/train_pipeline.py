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
from typing import List, Dict, Any, Optional, Union
import re


def convert_arg_name(name: str) -> str:
    """Convert YAML keys to command-line argument format."""
    return f"--{name.replace('-', '-')}"


def get_datasets(config: Dict[str, Any], key: str) -> List[str]:
    """Extract datasets from the config."""
    if key in config:
        if isinstance(config[key], list):
            return config[key]
        else:
            return [config[key]]
    return []


def validate_datasets(datasets: List[str]) -> List[str]:
    """Check that datasets exist."""
    for dataset in datasets:
        if not Path(dataset).exists():
            print(f"Error: Dataset {dataset} does not exist")
            sys.exit(1)
    return datasets


def get_cli_args(config: Dict[str, Any]) -> List[str]:
    """Convert config to command-line arguments."""
    args: List[str] = []
    for key, value in config.items():
        if key == "lambda":  # Special case for lambda (reserved keyword)
            args.append("--lambda")
            args.append(str(value))
        elif value is not None:
            args.append(convert_arg_name(key))
            args.append(str(value))
        else:
            args.append(convert_arg_name(key))
    return args


def build_command(script_path: str, feature_set: str, config: Dict[str, Any], prev_ckpt: Optional[str] = None) -> List[str]:
    """Build the command to execute based on the config."""
    cmd: List[str] = [sys.executable, script_path]

    cmd.append(f"--features={feature_set}")

    for dataset in validate_datasets(get_datasets(config, "datasets")):
        cmd.append(dataset)

    del config["datasets"]  # Remove datasets from config to avoid conflicts

    cmd_val_datasets = ""
    for val_dataset in validate_datasets(get_datasets(config, "validation-data")):
        cmd_val_datasets += val_dataset + " "
    
    if cmd_val_datasets:
        cmd.append("--validation-data")
        cmd.append(cmd_val_datasets.strip())

    del config["validation-data"]  # Remove validation-data from config to avoid conflicts

    # Add resume-from-model if we have a previous model
    if prev_ckpt and "resume-from-model" not in config:
        config["resume-from-model"] = prepare_checkpoint_to_pt(prev_ckpt, feature_set)

    # Convert config to command-line arguments
    cmd.extend(get_cli_args(config))

    return cmd


def find_latest_checkpoint(directory: Path) -> Optional[Path]:
    """Find the latest checkpoint in a directory based on filename pattern."""
    if not directory.exists():
        return None

    checkpoints: List[Path] = []

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


def prepare_checkpoint_to_pt(checkpoint_path: str, features: Optional[str]) -> str:
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


def get_output_dir(config: Dict[str, Any]) -> Path:
    """Determine the output directory from config or default."""
    if "default_root_dir" in config:
        return Path(config["default_root_dir"]) / Path("lightning_logs")
    return Path("lightning_logs")


def find_latest_model_checkpoint(stage_config: Dict[str, Any]) -> Optional[str]:
    """Find the latest model checkpoint for a given stage configuration."""

    def get_version_number(dir_name: str) -> int:
        """Extract version number from directory name like 'version_1', 'version_123', etc."""
        match = re.search(r"version_(\d+)", dir_name)
        return int(match.group(1)) if match else -1

    output_dir = get_output_dir(stage_config)

    version_dirs: List[Path] = [
        d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("version_")
    ]

    if not version_dirs:
        print(f"Warning: No version directories found in {output_dir}")
        return None

    # Sort version directories by actual version number (highest first)
    version_dirs = sorted(
        version_dirs, key=lambda d: get_version_number(d.name), reverse=True
    )

    # append checkpoints to version directories
    version_dirs = [d / Path("checkpoints") for d in version_dirs]

    # iterate through all version directories in reverse order (latest first)
    for version_dir in version_dirs:
        if not version_dir.exists():
            continue

        checkpoints: List[Path] = list(version_dir.glob("*.ckpt"))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            print(f"Found checkpoint: {latest_checkpoint}")
            return str(latest_checkpoint.resolve())

    return None


def get_sha256(file_path: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as f:
        # Read the file in chunks to avoid using too much memory
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()


def serialize_cpkt_to_nnue(net_dir: Path, cpkt_path: Path, features: str) -> None:
    """Serialize a checkpoint to NNUE format."""
    if not net_dir.exists():
        net_dir.mkdir(parents=True, exist_ok=True)

    if not cpkt_path.exists():
        print(f"Error: Checkpoint file {cpkt_path} does not exist")
        sys.exit(1)

    output_file = Path(tempfile.mktemp(suffix=".nnue"))
    cmd: List[str] = [
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


def verify_config(config: Dict[str, Any]) -> None:
    """Validate options, check that all other options are available in train.py --help"""
    script_path: str = config.get("script_path", "train.py")
    cmd: List[str] = [sys.executable, script_path, "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: Failed to run {script_path} --help")
        sys.exit(result.returncode)

    # loop over all stages and validate their options
    stages: Dict[str, Dict[str, Any]] = config.get("training", {})
    for stage_name, stage_config in stages.items():
        print(f"Validating stage: {stage_name}")
        
        # Check options exist in the help output
        help_output: str = result.stdout

        cmd_args: List[str] = []

        for key, _ in stage_config.items():
            # validate datasets
            if "datasets" == key:
                datasets = get_datasets(stage_config, "datasets")
                if not datasets:
                    print("Error: No datasets specified in the configuration")
                    sys.exit(1)
                validate_datasets(datasets)
                continue
            if "validation-data" == key:
                val_datasets = get_datasets(stage_config, "validation-data")
                if not val_datasets:
                    print("Error: No validation datasets specified in the configuration")
                    sys.exit(1)
                validate_datasets(val_datasets)
                continue
            if key == "lambda":
                cmd_args.append("--lambda")
            else:
                cmd_args.append(convert_arg_name(key))

        for arg in cmd_args:
            if arg not in help_output:
                print(f"Error: Option '{arg}' not found in {script_path} --help output")
                sys.exit(1)


def run_training(config_file: str, start_stage: int) -> None:
    """Run training based on the YAML configuration."""
    with open(config_file, "r") as file:
        config: Dict[str, Any] = yaml.safe_load(file)

    if "training" not in config:
        print("Error: YAML file must contain a 'training' section")
        sys.exit(1)

    script_path: str = config.get("script_path", "train.py")
    feature_set: str = config.get("features", "HalfKAv2_hm^")
    print(f"Using training script: {script_path}")
    print(f"Using features: {feature_set}")

    stages: Dict[str, Dict[str, Any]] = config["training"]
    if not stages:
        print("Error: No training stages defined")
        sys.exit(1)

    stage_names: List[str] = list(stages.keys())

    if start_stage < 0 or start_stage >= len(stage_names):
        print(
            f"Error: Invalid start stage {start_stage}. Must be between 0 and {len(stage_names) - 1}"
        )
        sys.exit(1)

    prev_ckpt: Optional[str] = None

    # Verify the configuration
    verify_config(config)

    if start_stage > 0:
        previous_stage_name: str = stage_names[start_stage - 1]
        prev_ckpt = find_latest_model_checkpoint(stages[previous_stage_name])

        if prev_ckpt:
            print(f"Starting from stage {start_stage}: {stage_names[start_stage]}")
            print(f"Using checkpoint from previous stage: {prev_ckpt}")
        else:
            print(f"Starting from stage {start_stage} without a checkpoint")

    # Run each stage in sequence
    for i, stage_name in enumerate(stage_names[start_stage:], start=start_stage):
        stage_config: Dict[str, Any] = stages[stage_name]

        print(f"\n{'=' * 30}")
        print(f"Starting training stage: {stage_name}")
        print(f"{'=' * 30}")

        cmd: List[str] = build_command(script_path, feature_set, stage_config, prev_ckpt)
        print(f"Executing command: {' '.join(cmd)}")

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(
                f"Error: Stage {stage_name} failed with return code {result.returncode}"
            )
            sys.exit(result.returncode)

        prev_ckpt = find_latest_model_checkpoint(stage_config)
        if prev_ckpt:
            print(f"Found checkpoint for next stage: {prev_ckpt}")
        else:
            print("No checkpoint found for next stage")
            sys.exit(1)

        serialize_cpkt_to_nnue(
            Path(config.get("network_dir", "networks")),
            Path(prev_ckpt),
            feature_set,
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