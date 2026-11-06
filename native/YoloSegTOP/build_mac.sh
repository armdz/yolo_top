#!/bin/bash
# Builds YOLO Seg Track for macOS (ONNX Runtime + CoreML) and installs it in TD's plugin folder:
#   ~/Library/Application Support/Derivative/TouchDesigner099/Plugins/YoloSegTOP.plugin
# Requires the Xcode Command Line Tools (xcode-select --install) and CMake (brew install cmake).
#   ./build_mac.sh            build and install
#   ./build_mac.sh noinstall  build only (output in build/mac)
set -euo pipefail
cd "$(dirname "$0")"

ORT_VERSION=1.23.2
ORT_DIR=third_party/onnxruntime/osx

if [ ! -f "$ORT_DIR/lib/libonnxruntime.dylib" ]; then
	echo "Downloading ONNX Runtime $ORT_VERSION (universal: Apple Silicon + Intel)..."
	tmp="$(mktemp -d)"
	curl -fL "https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VERSION/onnxruntime-osx-universal2-$ORT_VERSION.tgz" \
		| tar -xz -C "$tmp"
	mkdir -p "$ORT_DIR"
	cp -R "$tmp/onnxruntime-osx-universal2-$ORT_VERSION/." "$ORT_DIR/"
	rm -rf "$tmp"
fi

cmake -S . -B build/mac -DCMAKE_BUILD_TYPE=Release -DORT_DIR="$PWD/$ORT_DIR"
cmake --build build/mac --config Release -j "$(sysctl -n hw.ncpu)"

# zip para el release (un bundle es una carpeta; ditto conserva la firma)
rm -f build/YoloSegTOP-macOS.zip
ditto -c -k --norsrc --noextattr --keepParent build/mac/YoloSegTOP.plugin build/YoloSegTOP-macOS.zip

if [ "${1:-}" = "noinstall" ]; then
	echo "OK: build/mac/YoloSegTOP.plugin"
	exit 0
fi

PLUGINS="$HOME/Library/Application Support/Derivative/TouchDesigner099/Plugins"
mkdir -p "$PLUGINS"
for p in build/mac/*.plugin; do
	rm -rf "$PLUGINS/$(basename "$p")"
	cp -R "$p" "$PLUGINS/"
done
echo "OK: installed in $PLUGINS"
echo "Restart TouchDesigner. The first time, it asks for permission to load the plugin."
