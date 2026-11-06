# YOLO Seg Track TOP

Native TouchDesigner operator that detects, tracks and segments people with YOLO26: it finds each person, gives them a stable ID (ByteTrack) and computes their masks.

## Backends

One plugin, two backends. On Windows the plugin picks one when TouchDesigner loads it; on macOS it always uses ONNX Runtime. The operator and its parameters are the same everywhere, so a `.toe` works on any machine.

| Backend | Used on | Speed (yolo26n-seg 640×640, 1280×720 input) |
|---|---|---|
| **CUDA + TensorRT** | Windows with an NVIDIA RTX 20 GPU or newer | ~3-5 ms per frame on an RTX 5070 Ti Laptop |
| **ONNX Runtime** | macOS (Apple Silicon and Intel) and Windows without NVIDIA | Windows + DirectML (AMD or Intel GPU): ~15 ms network. Windows CPU: 45-70 ms network (8-core laptop). Mac: depends on the chip (CoreML uses the GPU and the Neural Engine) |

With CUDA the image stays on the GPU the whole time. With ONNX Runtime the image goes through CPU memory (the only option TouchDesigner offers plugins on macOS) and the network runs in a separate thread: TouchDesigner never waits for it, at the cost of the output lagging 1-2 frames behind the input.

The info popup (middle-click on the operator) shows which backend is in use. To try the ONNX Runtime backend on a PC with NVIDIA, set the environment variable `YOLOSEG_FORCE_PORTABLE=1` before opening TouchDesigner.

## Install

### Windows (no compiling)

Download `YoloSegTOP.dll` from [Releases](https://github.com/armdz/yolo_top/releases/latest) (or take `build\YoloSegTOP.dll`), copy it to `Documents\Derivative\Plugins` and restart TouchDesigner. The operator shows up in the OP Create dialog, **Custom** tab.

Without NVIDIA, the plugin uses the `onnxruntime.dll` that ships with TouchDesigner, which runs on the CPU. To use the GPU (AMD or Intel), run `scripts\get_directml.bat`. It downloads ONNX Runtime with DirectML to `Plugins\YoloSeg\`, and the operator's info popup then shows `Backend: ONNX Runtime DirectML`.

### macOS

Build it on the Mac (Mac binaries are not in the repo). You need the Xcode Command Line Tools (`xcode-select --install`), CMake (`brew install cmake`) and macOS 13.4 or newer.

```bash
cd native/YoloSegTOP
bash build_mac.sh
```

The script:

1. Downloads ONNX Runtime (universal: Apple Silicon + Intel).
2. Builds `YoloSegTOP.plugin`.
3. Installs it in `~/Library/Application Support/Derivative/TouchDesigner099/Plugins`.

The first time TouchDesigner loads the plugin, it asks for permission.

## Usage

1. Create a **YOLO Seg Track** and connect a camera or video to it.
   - CUDA backend: the input must be **8-bit RGBA**. If it isn't, set *Pixel Format = 8-bit fixed (RGBA)* on the previous TOP.
   - ONNX Runtime backend: any format works.
2. In *Model File*, choose a `.onnx` model (see [Models](#models)).
3. The first time a model loads:
   - CUDA: it builds the TensorRT engine. This takes several minutes (17 min on an RTX 5070 Ti laptop) and is cached next to the `.onnx`, so later loads are instant.
   - Mac: CoreML compiles the model and caches it in a `<model>.coreml` folder next to the `.onnx`.
   - Meanwhile the operator shows a warning; TouchDesigner keeps running.

**Task** (Detection section) chooses what the operator computes:

| Task | Masks | Persistent IDs | Use it for |
|---|---|---|---|
| *Segmentation + Tracking* (default) | Yes | Yes | Following each person and cutting them out. |
| *Segmentation* | Yes | No: IDs are 1, 2, 3… by confidence, every frame | Masks only (e.g. a mask of everyone), when identity doesn't matter. |
| *Tracking* | No: filled boxes | Yes | Positions and IDs only; the fastest. |

**Outputs**

- **Image** (same resolution as the input), set by *Output*:
  - *White Mask*: white mask with alpha.
  - *Colors per ID*: one color per ID, premultiplied.
  - *Debug Overlay*: the input image with masks and boxes on top.
- **Info DAT** (connect an Info DAT to the TOP): one row per person with `id, cx, cy, w, h, x1, y1, x2, y2, conf, area`.
  - Values are normalized 0-1, with y measured from the top.
  - `area` is the fraction of the image covered by the mask (by the box when there is no segmentation).
- **Info CHOP**:
  - `persons`, `infer_ms`, `total_ms`, `model_ready`.
  - `p<i>_id`, `p<i>_cx`, `p<i>_cy`, `p<i>_w`, `p<i>_h` for each person.

**Parameters**

| Section | Parameter | What it does |
|---|---|---|
| Model | Model File | The `.onnx` model. |
| Model | Device | ONNX Runtime backend only (greyed out with CUDA). *Auto*: GPU / Neural Engine when available (CoreML on Mac, DirectML on Windows). *CPU Only*: forces the CPU. |
| Model | FP16 | CUDA: builds the engine in FP16 (faster). Mac: allows reduced precision on the GPU. |
| Detection | Task | *Segmentation*, *Tracking* or *Segmentation + Tracking* (see above). |
| Detection | Confidence | Minimum score to detect a person and start a track. ByteTrack also uses detections down to 0.1 to keep existing tracks. |
| Detection | Output | *White Mask*, *Colors per ID* or *Debug Overlay*. |
| Segmentation | Soft Edges | Anti-aliased mask edges (~1.5 px). Off: hard edges. |
| Tracking | Track Buffer | How many frames a person who is no longer seen keeps their ID. |
| Tracking | Match Threshold | ByteTrack association threshold. |
| Tracking | Reset Tracker | Restarts the IDs. |

All parameters are on a single page, grouped in sections. Parameters that don't apply to the chosen task are greyed out. The info popup (middle-click on the operator) shows which backend is in use.

## Models

A ready-to-use model, `yolo26n-seg.onnx` (YOLO26n-seg, 384×640), is in [Releases](https://github.com/armdz/yolo_top/releases/latest). To use another one, you need a **YOLO26 seg** model exported to ONNX in **FP32** with a fixed size:

```python
from ultralytics import YOLO
YOLO("yolo26n-seg.pt").export(format="onnx", imgsz=(384, 640), half=False, simplify=True)
```

- `imgsz=(384, 640)` matches 16:9 and is faster than the default 640×640. For other aspect ratios, use the matching size (multiples of 32).
- `n` is the fastest; `s` and `m` are more accurate and slower.
- YOLO26 exports in *end2end* mode (no NMS, output `1×300×38`), which is what the operator expects. YOLOv8/11 seg models don't work as-is: the operator reports it with an error.
- If you re-export the `.onnx`, the CUDA backend notices it is newer than the engine and rebuilds it.

## Building

### Windows

Requires Visual Studio 2022 (any edition, or the Build Tools, with *Desktop development with C++*) and CUDA Toolkit 12.x (12.8 or newer to build natively for RTX 50). The DLLs include both backends; the NVIDIA DLLs (driver, TensorRT) are delay-loaded, so they also load on PCs without NVIDIA.

```
build.bat             build and install in Documents\Derivative\Plugins
build.bat noinstall   build only (output in build\)
```

- **It also works with TouchDesigner open:** the loaded DLL is renamed to `.dll.old` and the new one takes its place. TD keeps using the old one until restarted. `.old` files are deleted on the next build.
- TensorRT 10.9 is used straight from TouchDesigner's DLLs (2025.33230).
  - Headers are in `third_party/tensorrt/include`.
  - The `.lib` files are generated from TD's DLLs by `third_party/tensorrt/mklib.bat`.
- If TouchDesigner updates to another TensorRT version, regenerate the `.def` and `.lib` files (see `mklib.bat`) and delete the `.engine` files.

Without the CUDA Toolkit you can build an ONNX Runtime-only version with CMake: `cmake -B build/cmake -A x64`, then `cmake --build build/cmake --config Release`.

### macOS

`bash build_mac.sh` (see [Install](#macos)). `bash build_mac.sh noinstall` only builds, into `build/mac`.

### Testing outside TD

```
rem CUDA backend
set PATH=C:\Program Files\Derivative\TouchDesigner.2025.33230\bin;%PATH%
build\bench.exe model.onnx image.rgba 1280 720 output.bgra 300 2 2

rem ONNX Runtime backend (YOLOSEG_ORT: which onnxruntime.dll to use)
set YOLOSEG_ORT=C:\Program Files\Derivative\TouchDesigner.2025.33230\bin\onnxruntime.dll
build\cpu_bench.exe model.onnx image.rgba 1280 720 output.bgra 50 2 2 0
```

Arguments:

- Raw RGBA image, width and height.
- Output file (raw BGRA).
- Iterations.
- Output mode: 0 White Mask, 1 Colors, 2 Overlay.
- Task: 0 Segmentation, 1 Tracking, 2 Segmentation + Tracking.
- `cpu_bench` only: device, 0 Auto or 1 CPU.

On a Mac, `cpu_bench` is built in `build/mac` with its own `libonnxruntime.dylib` next to it.

## Troubleshooting

| Message | Fix |
|---|---|
| *The input must be 8-bit RGBA* | CUDA backend: set *Pixel Format = 8-bit fixed (RGBA)* on the previous TOP. |
| *Unexpected model format* | Use a YOLO26 seg model exported with `format="onnx"` (see [Models](#models)). |
| *The model has a dynamic shape* | Export with `dynamic=False` and a fixed `imgsz`. |
| *onnxruntime.dll not found* | Use TouchDesigner 2023 or newer, or run `scripts\get_directml.bat`. |
| It is slow | Check the backend in the info popup. If it says *ONNX Runtime CPU* on Windows, install DirectML (or check the NVIDIA driver). |

## Files

| File | What it does |
|---|---|
| `src/YoloSegPlugin.cpp`, `src/Backend.cpp` | Plugin entry point: picks the backend when TouchDesigner loads the plugin. |
| `src/OpCommon.h` | Parameters and Info DAT/CHOP, shared by both backends. |
| `src/SegTypes.h` | Common types (persons, boxes, settings). |
| `src/ByteTracker.cpp` | Port of Ultralytics' ByteTrack, with Kalman filter and Hungarian assignment. Used by both backends. |
| `src/YoloSegTOP.cpp` | CUDA backend operator (CUDA interface with TD). |
| `src/Pipeline.cpp` | CUDA: preprocess → TensorRT → ByteTrack → masks → output. |
| `src/TrtModel.cpp` | CUDA: builds and loads the TensorRT engine, cached on disk. |
| `src/Kernels.cu` | CUDA: CUDA kernels. |
| `src/cpu/YoloSegTOP_CPU.cpp` | ONNX Runtime backend operator. |
| `src/cpu/CpuPipeline.cpp` | ONNX Runtime: worker thread with preprocess → ONNX Runtime → ByteTrack → masks → output. |
| `src/cpu/OrtModel.cpp` | ONNX Runtime: loads ONNX Runtime at runtime and picks the backend (CoreML, DirectML or CPU). |
| `src/cpu/CpuKernels.cpp` | ONNX Runtime: the same math as the CUDA kernels, multithreaded on the CPU. |
| `src/SegShared.h`, `src/cpu/SegSharedCPU.h` | Frame data (boxes, masks, image) exported by the plugin so other plugins can read it. |

## Third-party licenses

- `td/`: Derivative headers, under their *Shared Use License* (in each file's header).
- `third_party/tensorrt/include`: NVIDIA TensorRT headers (Apache 2.0).
- `third_party/onnxruntime`: ONNX Runtime headers (MIT, see `third_party/onnxruntime/LICENSE`).
- `src/ByteTracker.cpp` is a port of `ultralytics/trackers/byte_tracker.py` (Ultralytics, AGPL-3.0). YOLO26 models are also by Ultralytics, under the same license.
