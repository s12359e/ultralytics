# Ultralytics YOLO11 Segmentation with OpenCV and ONNX

This example demonstrates how to perform instance segmentation using a [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) model exported to the [ONNX](https://onnx.ai/) format and executed with [OpenCV](https://opencv.org/) 4.8 or newer.

## 🚀 Getting Started

1. **Clone the Repository**

   ```bash
   git clone https://github.com/ultralytics/ultralytics.git
   cd ultralytics/examples/YOLO11-Segmentation-OpenCV-ONNX-Python
   ```

2. **Install Requirements**

   Only OpenCV (for ONNX inference) and NumPy are required:

   ```bash
   pip install opencv-python>=4.8 numpy
   ```

3. **Export Your Model**

   If you have a trained YOLO11 segmentation model (`yolo11n-seg.pt` for example), export it to ONNX using the
   [Ultralytics](https://github.com/ultralytics/ultralytics) CLI (which requires PyTorch) in a separate environment:

   ```bash
   yolo export model=yolo11n-seg.pt format=onnx opset=12
   ```

4. **Run the Segmentation Script**

   ```bash
   python main.py --model yolo11n-seg.onnx --source path/to/image.jpg
   ```

   The script will display the image with predicted masks and bounding boxes.

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request with improvements or bug fixes.
