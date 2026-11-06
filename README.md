# yolo_top

People tracking and segmentation in TouchDesigner with YOLO26.

Runs on Windows (with or without an NVIDIA GPU) and on macOS.

[Español (Argentina) más abajo](#español)

## Contents

| Folder | What it is |
|---|---|
| [`native/YoloSegTOP`](native/YoloSegTOP/README.md) | Native operator. **YOLO Seg Track** detects, tracks and segments people. One plugin that picks its backend: **CUDA + TensorRT** on NVIDIA (~3-5 ms per frame), **ONNX Runtime** elsewhere (CoreML on Mac, DirectML or CPU on Windows). |

## Quick start

1. Download from [Releases](https://github.com/armdz/yolo_top/releases/latest):
   - `YoloSegTOP.dll` (Windows).
   - `yolo26n-seg.onnx`: a ready-to-use model (YOLO26n-seg, 384×640).
2. Install the operator:

   | System | How |
   |---|---|
   | Windows | Copy `YoloSegTOP.dll` to `Documents\Derivative\Plugins` and restart TouchDesigner. Without NVIDIA, also run [`get_directml.bat`](native/YoloSegTOP/scripts/get_directml.bat) to use the GPU (AMD or Intel). |
   | macOS | `bash native/YoloSegTOP/build_mac.sh` (builds and installs; Mac binaries are not in the release yet). |

3. In TouchDesigner: OP Create, **Custom** tab, **YOLO Seg Track**. Connect a camera and choose `yolo26n-seg.onnx` in *Model File*.
   - With NVIDIA, the first load builds the TensorRT engine: it takes several minutes (the operator shows a warning meanwhile) and is cached next to the `.onnx`, so later loads are instant.

To export other models (bigger, other aspect ratios), see [Models](native/YoloSegTOP/README.md#models). Details (parameters, outputs, building, troubleshooting) are in [`native/YoloSegTOP/README.md`](native/YoloSegTOP/README.md).

## Requirements

- TouchDesigner 2025 (the CUDA backend uses the TensorRT 10.9 DLLs that ship with TD 2025.33230).
- Windows 10/11 (fastest with an NVIDIA RTX 20 GPU or newer), or a Mac with macOS 13.4 (Ventura) or newer (required by ONNX Runtime).

## License

The YOLO26 model and the ByteTrack port are based on [Ultralytics](https://github.com/ultralytics/ultralytics) code, licensed under AGPL-3.0.

---

## Español

Tracking y segmentación de personas en TouchDesigner con YOLO26, como operador nativo. Anda en Windows (con o sin placa NVIDIA) y en Mac.

### Qué hace

**YOLO Seg Track** detecta a las personas en la imagen, le da a cada una un ID que se mantiene mientras sigue en cuadro (ByteTrack) y saca su máscara.

- **Con NVIDIA** (RTX 20 o más nueva) usa CUDA + TensorRT: unos 3-5 ms por frame.
- **Sin NVIDIA** usa ONNX Runtime: en Windows corre en la placa de video con DirectML o en CPU, y en Mac usa CoreML (GPU y Neural Engine). La red corre en un hilo aparte, así que TouchDesigner no se traba; la salida va 1 o 2 frames atrasada.

Es un solo plugin: elige el motor solo al arrancar TouchDesigner. Un `.toe` hecho en una máquina abre igual en cualquier otra.

### Instalación en Windows

1. Bajá de [Releases](https://github.com/armdz/yolo_top/releases/latest) el `YoloSegTOP.dll` y el modelo `yolo26n-seg.onnx`.
2. Copiá `YoloSegTOP.dll` a `Documents\Derivative\Plugins` (si la carpeta no existe, creala).
3. Si **no** tenés NVIDIA, corré [`get_directml.bat`](native/YoloSegTOP/scripts/get_directml.bat) para que use la placa de video (AMD o Intel). Sin eso anda igual, pero en CPU y más lento.
4. Reiniciá TouchDesigner.

### Instalación en Mac

Por ahora hay que compilarlo en la Mac. Necesitás macOS 13.4 o más nuevo, las herramientas de Xcode y CMake:

```bash
xcode-select --install
brew install cmake
cd native/YoloSegTOP
bash build_mac.sh
```

El script baja ONNX Runtime, compila el plugin y lo instala en `~/Library/Application Support/Derivative/TouchDesigner099/Plugins`. La primera vez que TouchDesigner lo carga, te pide permiso.

### Uso

1. En TouchDesigner: OP Create, pestaña **Custom**, **YOLO Seg Track**.
2. Conectale la cámara o el video.
   - Con NVIDIA, la entrada tiene que ser **8 bit RGBA**. Si no lo es, poné *Pixel Format = 8-bit fixed (RGBA)* en el TOP anterior.
3. En *Model File* elegí `yolo26n-seg.onnx`.
   - Con NVIDIA, la primera vez arma el engine de TensorRT. Tarda varios minutos y mientras tanto el operador muestra un warning. Después lo guarda al lado del `.onnx` y las próximas veces carga al toque.
4. Elegí qué querés en *Task*:
   - *Segmentation + Tracking* (por defecto): máscaras e IDs que se mantienen.
   - *Segmentation*: solo máscaras. Los IDs se renumeran en cada frame (1, 2, 3… por confianza).
   - *Tracking*: solo posiciones e IDs, con cajas en lugar de máscaras. Es lo más rápido.

**Salidas:**

- **Imagen**, según *Output*: máscara blanca, un color por ID, u overlay con la imagen de entrada.
- **Info DAT** (conectale un Info DAT): una fila por persona con `id, cx, cy, w, h, x1, y1, x2, y2, conf, area`, normalizado de 0 a 1 y con y medida desde arriba.
- **Info CHOP**: cantidad de personas, tiempos y `p<i>_id`, `p<i>_cx`, `p<i>_cy`, `p<i>_w`, `p<i>_h` por persona.

El popup de info (clic del medio sobre el operador) te dice qué motor está usando.

### Si algo no anda

| Mensaje | Qué hacer |
|---|---|
| *The input must be 8-bit RGBA* | Poné *Pixel Format = 8-bit fixed (RGBA)* en el TOP anterior. |
| *Unexpected model format* | Usá un YOLO26 seg exportado a ONNX (ver [Models](native/YoloSegTOP/README.md#models)). Los YOLOv8/11 no sirven tal cual. |
| *onnxruntime.dll not found* | Usá TouchDesigner 2023 o más nuevo, o corré `get_directml.bat`. |
| Va lento | Mirá el motor en el popup de info. Si en Windows dice *ONNX Runtime CPU*, instalá DirectML o revisá el driver de NVIDIA. |

### Otros modelos

Para exportar un modelo más grande (más preciso y más lento) o para otra relación de aspecto, con Python y Ultralytics:

```python
from ultralytics import YOLO
YOLO("yolo26s-seg.pt").export(format="onnx", imgsz=(384, 640), half=False, simplify=True)
```

`imgsz=(384, 640)` es para 16:9. Para otra relación de aspecto usá la medida que corresponda, en múltiplos de 32.
