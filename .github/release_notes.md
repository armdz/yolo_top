**YOLO Seg Track** for TouchDesigner: people detection, tracking and segmentation with YOLO26.

### Downloads

- `YoloSegTOP.dll`: Windows. Copy it to `Documents\Derivative\Plugins` and restart TouchDesigner. With an NVIDIA GPU (RTX 20 or newer) it uses CUDA + TensorRT; otherwise ONNX Runtime.
- `YoloSegTOP-macos.zip`: macOS 13.4 or newer (Apple Silicon and Intel). Unzip it, copy `YoloSegTOP.plugin` to `~/Library/Application Support/Derivative/TouchDesigner099/Plugins` and run once in Terminal: `xattr -dr com.apple.quarantine ~/Library/Application\ Support/Derivative/TouchDesigner099/Plugins/YoloSegTOP.plugin`
- `yolo26n-seg.onnx`: ready-to-use model (YOLO26n-seg, 384×640, FP32). Choose it in *Model File*. With NVIDIA, the first load builds the TensorRT engine (several minutes, cached next to the `.onnx`).
- `get_directml.bat`: optional, for Windows without NVIDIA. Run it to use the GPU (AMD or Intel) instead of the CPU.

The model is by [Ultralytics](https://github.com/ultralytics/ultralytics) and is licensed under AGPL-3.0.

---

**YOLO Seg Track** para TouchDesigner: detección, tracking y segmentación de personas con YOLO26.

### Descargas

- `YoloSegTOP.dll`: Windows. Copialo a `Documents\Derivative\Plugins` y reiniciá TouchDesigner. Con placa NVIDIA (RTX 20 o más nueva) usa CUDA + TensorRT; si no, ONNX Runtime.
- `YoloSegTOP-macos.zip`: macOS 13.4 o más nuevo (Apple Silicon e Intel). Descomprimilo, copiá `YoloSegTOP.plugin` a `~/Library/Application Support/Derivative/TouchDesigner099/Plugins` y corré una vez en la Terminal: `xattr -dr com.apple.quarantine ~/Library/Application\ Support/Derivative/TouchDesigner099/Plugins/YoloSegTOP.plugin`
- `yolo26n-seg.onnx`: modelo listo para usar (YOLO26n-seg, 384×640, FP32). Elegilo en *Model File*. Con NVIDIA, la primera vez arma el engine de TensorRT (tarda varios minutos y queda guardado al lado del `.onnx`).
- `get_directml.bat`: opcional, para Windows sin NVIDIA. Corrélo para que use la placa de video (AMD o Intel) en lugar de la CPU.

El modelo es de [Ultralytics](https://github.com/ultralytics/ultralytics) y tiene licencia AGPL-3.0.
