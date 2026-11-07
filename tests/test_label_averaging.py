import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_chain_label_averager(tmp_path):
    exe = tmp_path / "test_label_averaging"
    subprocess.run(
        [
            "g++",
            "-O1",
            "-std=c++20",
            "-march=native",
            f"-I{os.path.join(ROOT, 'data_loader', 'cpp', 'lib')}",
            os.path.join(ROOT, "tests", "test_label_averaging.cpp"),
            "-o",
            str(exe),
        ],
        check=True,
    )
    subprocess.run([str(exe)], check=True)
