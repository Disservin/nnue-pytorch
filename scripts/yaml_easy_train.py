from glob import glob
import random
import re
import subprocess
import sys
import yaml

best_engine_arch = subprocess.check_output(
    "sh ./scripts/get_native_properties.sh | awk '{print $1}'",
    shell=True
).decode().strip()

default_args = {
    'seed': random.randint(9999, 9_999_999),
    'tui': False,
    'workspace-path': '/root/easy-train-data',
    'build-engine-arch': best_engine_arch,
    'network-testing-book': 'https://github.com/official-stockfish/books/raw/master/UHO_Lichess_4852_v1.epd.zip',
}

yaml_config_file = sys.argv[1]
with open(yaml_config_file, "r") as f:
    args = yaml.safe_load(f)
    args = {**default_args, **args}

# if config filename contains gpu ids, automatically use them
gpus_from_filename = re.search(r"gpu(\d+)", yaml_config_file)
if gpus_from_filename:
    gpus_str = ",".join(list(gpus_from_filename.group(1)))
    args["gpus"] = gpus_str

# prepare an easy_train.py command for training
command = ["uv run ./scripts/easy_train.py"]
for key,value in sorted(args.items()):
    if key == "training-dataset":
        if isinstance(value, list):
            # list of files with support for wildcard *
            filenames = value
            for dataset_component in filenames:
                if "*" in dataset_component:
                    for glob_match in glob(dataset_component):
                        command.append(f"  --{key} {glob_match}")
                else:
                    command.append(f"  --{key} {dataset_component}")
        elif isinstance(value, dict):
            #   /data/hse-v1/:
            #     leela96-filt-v2.min.high-simple-eval-1k.min-v2.binpack
            for basepath,filenames in value.items():
                for filename in filenames:
                    dataset_component = f"{basepath}{filename}"
                    if "*" in dataset_component:
                        for glob_match in glob(dataset_component):
                            command.append(f"  --{key} {glob_match}")
                    else:
                        command.append(f"  --{key} {dataset_component}")
        else:
            command.append(f"  --{key} {value}")
    else:
        command.append(f"  --{key} {value}")

# pretty print for logging and debugging
if len(sys.argv) > 2 and sys.argv[2] == "print":
    print(" \\\n".join(command))
else:
    print(" ".join(command))