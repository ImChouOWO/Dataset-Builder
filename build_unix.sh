#!/usr/bin/env bash
set -euo pipefail
python3 -m pip install -r requirements-build.txt
python3 -m PyInstaller --onefile --windowed --name YOLODatasetBuilder yolo_dataset_builder.py
python3 -m PyInstaller --onefile --name yolo-dataset-builder yolo_dataset_builder.py
