# yolo_top

People tracking and segmentation in TouchDesigner with YOLO26.

Runs on Windows (with or without an NVIDIA GPU) and on macOS.

## Contents

| Folder | What it is |
|---|---|
| [`native/YoloSegTOP`](native/YoloSegTOP/README.md) | Native operator. **YOLO Seg Track** detects, tracks and segments people. One plugin that picks its backend: **CUDA + TensorRT** on NVIDIA (~3-5 ms per frame), **ONNX Runtime** elsewhere (CoreML on Mac, DirectML or CPU on Windows). |

## Quick start

1. Install the operator:

   | System | How |
   |---|---|
   | Windows | Copy `native/YoloSegTOP/build/YoloSegTOP.dll` to `Documents\Derivative\Plugins`. Without NVIDIA, also run `native\YoloSegTOP\scripts\get_directml.bat` to use the GPU (AMD or Intel). |
   | macOS | `bash native/YoloSegTOP/build_mac.sh` (builds and installs). |

2. Export a model:

   ```python
   from ultralytics import YOLO
   YOLO("yolo26n-seg.pt").export(format="onnx", imgsz=(384, 640), half=False, simplify=True)
   ```

3. In TouchDesigner: OP Create, **Custom** tab, **YOLO Seg Track**. Connect a camera and choose the `.onnx` in *Model File*.

Details (parameters, outputs, building, troubleshooting) are in [`native/YoloSegTOP/README.md`](native/YoloSegTOP/README.md).

## Requirements

- TouchDesigner 2025 (the CUDA backend uses the TensorRT 10.9 DLLs that ship with TD 2025.33230).
- Windows 10/11 (fastest with an NVIDIA RTX 20 GPU or newer), or a Mac with macOS 13.4 (Ventura) or newer (required by ONNX Runtime).
