#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import yaml
from pathlib import Path
import re


def convert_arg_name(name):
    """Convert YAML keys to command-line argument format."""
    return f"--{name.replace('-', '-')}"


def build_command(script_path, config, previous_model=None):
    """Build the command to execute based on the config."""
    cmd = [sys.executable, script_path]

    if "datasets" in config:
        if isinstance(config["datasets"], list):
            cmd.extend(config["datasets"])
        else:
            cmd.append(config["datasets"])

        # Remove datasets from config to avoid adding them again
        del config["datasets"]

    # Add resume-from-model if we have a previous model
    if previous_model and "resume-from-model" not in config:
        config["resume-from-model"] = previous_model

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


def get_output_dir(config):
    """Determine the output directory from config or default."""
    if "default_root_dir" in config:
        return Path(config["default_root_dir"]) / Path("lightning_logs")
    return Path("lightning_logs")


def run_training(config_file):
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

    previous_model = None

    # Run each stage in sequence
    for stage_name, stage_config in stages.items():
        print(f"\n{'='*30}")
        print(f"Starting training stage: {stage_name}")
        print(f"{'='*30}")

        cmd = build_command(script_path, stage_config, previous_model)
        print(f"Executing command: {' '.join(cmd)}")

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(
                f"Error: Stage {stage_name} failed with return code {result.returncode}"
            )
            sys.exit(result.returncode)

        output_dir = get_output_dir(stage_config)
        version_dirs = [
            d
            for d in output_dir.iterdir()
            if d.is_dir() and d.name.startswith("version_")
        ]
        print(f"{output_dir}")

        if version_dirs:
            latest_version = max(version_dirs, key=lambda x: int(x.name.split("_")[1]))
            checkpoint = find_latest_checkpoint(latest_version / "checkpoints")
            if checkpoint:
                previous_model = str(checkpoint)
                print(f"Found checkpoint for next stage: {previous_model}")
            else:
                print(
                    f"Warning: No checkpoint found in {latest_version / 'checkpoints'}"
                )
        else:
            print(f"Warning: No version directories found in {output_dir}")

    print("\nAll training stages completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run multi-stage training using a YAML configuration"
    )
    parser.add_argument("config", help="YAML configuration file")
    args = parser.parse_args()

    run_training(args.config)
