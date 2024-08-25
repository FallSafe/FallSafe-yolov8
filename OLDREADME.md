# Fall Safe: Real-Time Fall Detection System

## Abstract

**Fall Safe** is designed to address fall-related injuries among vulnerable populations by leveraging computer vision and machine learning. The system detects falls in real-time from CCTV footage, analyzing video streams to identify abnormal movements and postures. Alerts are sent to caregivers or emergency services with details about the incident, aiming to improve response times and safety for at-risk individuals.

## Features

- **Real-Time Fall Detection**: Utilizes YOLOv8 for accurate fall detection.
- **Integration**: Works with existing CCTV setups.
- **Alerts**: Sends notifications with incident details to caregivers or emergency services.

## Getting Started

### Prerequisites

- **Python**: Version 3.10.x
- **Anaconda**: For environment management
- **NVIDIA GPU** (Optional but highly recommended): For accelerated processing

### Setup and Installation

1. **Install Python 3.10.x**
   - Download and install from [Python's official website](https://www.python.org/).

2. **Set Up YOLOv8 Project**
   ```bash
   mkdir YOLO_PROJECT/yolov8-python
   cd YOLO_PROJECT/yolov8-python

3. **Install Anaconda**
   - Download and install from [Anaconda's official website](https://www.anaconda.com/).

4. **Install GPU Drivers and CUDA**
   - Install NVIDIA GPU drivers.
   - Install CUDA (Version 12.1.0) from [NVIDIA](https://developer.nvidia.com/cuda-toolkit).
   - Install CuDNN (Version 9.2.1) from [NVIDIA](https://developer.nvidia.com/cudnn).
   - Verify CUDA installation

5. **Set Up Conda Environment**
   ```bash
   conda create -p yolov8-gpu-env python=3.10
   conda activate yolov8-gpu-env
   pip install -r requirements.txt
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117 --timeout=1000
   pip install ultralytics
   ```

### Running the System

1. **Test YOLOv8 Inference by downloading the models from ultralytics**
   ```bash
   python detection.py --model yolov8n.onnx --source data/images/horses.jpg
   python detection.py --model yolov8n.onnx --source data/videos/road.mp4
   python detection.py --model yolov8n.onnx --source 0
   ```

2. **Get Labelled Dataset from Roboflow**
   - Structure of dataset are as follows:
     ```
     dataset
      ├── train
      │   ├── images
      │   │   ├── image0.jpg
      │   │   ├── image1.jpg
      │   ├── labels
      │   │   ├── image0.txt
      │   │   ├── image1.txt
      ├── val
      │   ├── images
      │   │   ├── image0.jpg
      │   │   ├── image1.jpg
      │   ├── labels
      │   │   ├── image0.txt
      │   │   ├── image1.txt
      ├── test
      │   ├── images
      │   │   ├── image0.jpg
      │   │   ├── image1.jpg
      │   ├── labels
      │   │   ├── image0.txt
      │   │   ├── image1.txt

     ```

3. **Organize the dataset by using Organize.py**
   - Edit Organize.py according to the dataset path
   - Orgnaize the dataset by:
     ```bash
     python Organize.py
     ```

5. **Dataset Structure**
   - Dataset Structure will become as follows:
     ```
     dataset
      ├── train
      │   ├── fall
      │   │   ├── image0.jpg
      │   │   ├── image1.jpg
      │   ├── nofall
      │   │   ├── image0.jpg
      │   │   ├── image1.jpg
      ├── val
      │   ├── fall
      │   │   ├── image0.jpg
      │   │   ├── image1.jpg
      │   ├── nofall
      │   │   ├── image0.jpg
      │   │   ├── image1.jpg
      ├── test
      │   ├── fall
      │   │   ├── image0.jpg
      │   │   ├── image1.jpg
      │   ├── nofall
      │   │   ├── image0.jpg
      │   │   ├── image1.jpg

     ```

3. **Train the Model**
   ```bash
   yolo classify train model=yolov8l-cls.pt data="\path\data\dataset" imgsz=224 device=0 workers=2 batch=16 epochs=100 patience=50 name=yolov8_fallsafe_classification
   ```

4. **Continue Training**
   ```bash
   yolo classify train model=runs/classify/yolov8_fallsafe_classification/weights/last.pt resume=True
   ```

5. **Perform Classification**
   ```bash
   yolo classify predict model=runs/classify/yolov8_fallsafe_classification/weights/best.pt source="inference/classify/image.jpg" save=True
   ```

6. **Real-Time Classification via Camera**
   ```bash
   yolo detect predict model=runs/classify/yolov8_fallsafe_classification/weights/best.pt source="0" save=True conf=0.5 show=True save_txt=True line_thickness=1
   ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have improvements or suggestions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, please contact us at Issues Pages.

## Authors

- [Syed Arbaaz Hussain](https://github.com/SyedArbaazHussain)
- [Adithi N Gatty](https://github.com/AdithiNgatty)
- [Prabuddh Shetty](https://github.com/Prabuddhshetty901)
- [Shreya S Rao](https://github.com/shreyarao515)
---
**Fall Safe** is developed by the above contributors. For more information, visit [our GitHub repository](https://github.com/SyedArbaazHussain/MAJOR_PROJECT).
