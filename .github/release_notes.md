**YOLO Seg Track** for TouchDesigner: people detection, tracking and segmentation with YOLO26.

### Downloads

- `YoloSegTOP.dll`: Windows. Copy it to `Documents\Derivative\Plugins` and restart TouchDesigner. With an NVIDIA GPU (RTX 20 or newer) it uses CUDA + TensorRT; otherwise ONNX Runtime.
- `yolo26n-seg.onnx`: ready-to-use model (YOLO26n-seg, 384×640, FP32). Choose it in *Model File*. With NVIDIA, the first load builds the TensorRT engine (several minutes, cached next to the `.onnx`).
- `get_directml.bat`: optional, for Windows without NVIDIA. Run it to use the GPU (AMD or Intel) instead of the CPU.

macOS: build it with `bash native/YoloSegTOP/build_mac.sh` (see the README).

The model is by [Ultralytics](https://github.com/ultralytics/ultralytics) and is licensed under AGPL-3.0.

---

**YOLO Seg Track** para TouchDesigner: detección, tracking y segmentación de personas con YOLO26.

### Descargas

- `YoloSegTOP.dll`: Windows. Copialo a `Documents\Derivative\Plugins` y reiniciá TouchDesigner. Con placa NVIDIA (RTX 20 o más nueva) usa CUDA + TensorRT; si no, ONNX Runtime.
- `yolo26n-seg.onnx`: modelo listo para usar (YOLO26n-seg, 384×640, FP32). Elegilo en *Model File*. Con NVIDIA, la primera vez arma el engine de TensorRT (tarda varios minutos y queda guardado al lado del `.onnx`).
- `get_directml.bat`: opcional, para Windows sin NVIDIA. Corrélo para que use la placa de video (AMD o Intel) en lugar de la CPU.

Mac: compilalo con `bash native/YoloSegTOP/build_mac.sh` (ver el README).

El modelo es de [Ultralytics](https://github.com/ultralytics/ultralytics) y tiene licencia AGPL-3.0.
