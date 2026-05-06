# YOLO Dataset Builder

一個跨平台的 YOLO 訓練資料集自動化工具，可在 macOS、Windows、Ubuntu 上以 Python 執行，也可以用 PyInstaller 打包成各平台執行檔。

![image](https://github.com/ImChouOWO/Dataset-Builder/blob/main/img/img.png)

## 功能

- 指定父級資料夾後，遞迴掃描所有子資料夾。
- 自動偵測圖像資料夾與標記檔資料夾配對，例如 `images` ↔ `labels`、`img` ↔ `annotations`。
- 允許同一個父級資料夾底下存在多組圖像與標記資料夾。
- 指定 train / val / test 切分比例。
- 指定輸出位置。
- 匯入或手動輸入 label id 與 class name 對應。
- 支援合併類別，並輸出修改後的 YOLO 標記檔。
- 讀取標記檔後輸出各類別分布報告。
- 預設使用複製，不改動來源資料；也支援 move 模式與覆寫輸出資料夾。
- 同時支援 GUI 與 CLI。

## 支援的輸出結構

```text
output_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── classes.txt
├── data.yaml
└── reports/
    ├── class_distribution.json
    ├── class_distribution_original.csv
    ├── class_distribution_mapped.csv
    └── manifest.csv
```

## 執行 GUI

```bash
python yolo_dataset_builder.py --gui
```

或直接執行：

```bash
python yolo_dataset_builder.py
```

## CLI 範例

```bash
python yolo_dataset_builder.py \
  --parent /path/to/raw_dataset_parent \
  --output /path/to/output_dataset \
  --classes classes.txt \
  --merge-rules "2,3 -> 1" \
  --split 0.8 0.1 0.1
```

只分析類別分布，不輸出資料集：

```bash
python yolo_dataset_builder.py \
  --parent /path/to/raw_dataset_parent \
  --output /tmp/not_used \
  --classes classes.txt \
  --analyze-only
```

## Class mapping 格式

支援以下格式：

```text
person
car
bus
```

或：

```text
0 person
1 car
2 bus
```

或：

```text
0: person
1: car
2: bus
```

或 JSON：

```json
{"0": "person", "1": "car", "2": "bus"}
```

## 類別合併規則

以 `source -> target` 表示。可以用 id 或 class name。

```text
2 -> 1
3,4 -> 1
car,bus -> vehicle
```

注意：name-based merge 的 class name 必須存在於 class mapping 中。

## 建議的原始資料夾結構

工具會尋找同一層底下的圖像資料夾與標記資料夾：

```text
raw_parent/
├── dataset_A/
│   ├── images/
│   └── labels/
├── dataset_B/
│   ├── img/
│   └── annotations/
└── dataset_C/
    ├── images/
    └── labels/
```

可在 GUI 或 CLI 中修改候選資料夾名稱，例如：

```bash
--image-dir-names images,img,jpgs
--label-dir-names labels,annotations,txt
```

## 打包成執行檔

安裝 PyInstaller：

```bash
python -m pip install pyinstaller
```

GUI 版：

```bash
pyinstaller --onefile --windowed --name YOLODatasetBuilder yolo_dataset_builder.py
```

CLI 版：

```bash
pyinstaller --onefile --name yolo-dataset-builder yolo_dataset_builder.py
```

輸出會在 `dist/` 目錄。

重要限制：PyInstaller 不是一般意義上的跨平台交叉編譯器。Windows 的 `.exe` 需要在 Windows 上建置；macOS app 需要在 macOS 上建置；Linux binary 需要在 Linux 上建置。

## Ubuntu 的 Tkinter 注意事項

若 Ubuntu 執行 GUI 時出現 `No module named tkinter`，請先安裝：

```bash
sudo apt update
sudo apt install python3-tk
```

## 安全建議

- 預設使用 `copy`，來源資料不會被修改。
- 使用 `move` 前請先備份資料，因為影像會從來源資料夾移出。
- 類別合併會修改「輸出資料夾」中的標記檔，不會原地修改來源標記檔。
