python -m pip install -r requirements-build.txt
python -m PyInstaller --onefile --windowed --name YOLODatasetBuilder yolo_dataset_builder.py
python -m PyInstaller --onefile --name yolo-dataset-builder yolo_dataset_builder.py
