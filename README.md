# Fall Safe: Real-Time Fall Detection System

## Abstract

**Fall Safe** is designed to address fall-related injuries among vulnerable populations by leveraging computer vision and machine learning. The system detects falls in real-time from CCTV footage, analyzing video streams to identify abnormal movements and postures. Alerts are sent to caregivers or emergency services with details about the incident, aiming to improve response times and safety for at-risk individuals.

## Features

- **Real-Time Fall Detection**: Utilizes YOLOv8 for accurate fall detection.
- **Integration**: Works with existing CCTV setups.
- **Alerts**: Sends notifications with incident details to caregivers or emergency services.

## Getting Started

### Prerequisites

- **Python**: Latest version
- **NVIDIA GPU** (Optional but highly recommended): For accelerated processing

### Setup and Installation

1. **Install Python**
   - Download and install from [Python's official website](https://www.python.org/downloads/).

2. **Set Up YOLOv8 Project**
   ```bash
   mkdir YOLO_PROJECT/yolov8-python
   cd YOLO_PROJECT/yolov8-python
   ```

3. **Create a Virtual Environment**

   - **Using venv**:
     ```bash
     python -m venv env
     ```
     Activate the virtual environment:
     - On Windows:
       ```bash
       .\env\Scripts\activate
       ```
     - On macOS/Linux:
       ```bash
       source env/bin/activate
       ```

   - **Using conda**: For detailed instructions on creating a conda environment, refer to the [Official Anaconda Documentation](https://docs.anaconda.com/anaconda/install/).

4. **Install GPU Drivers and CUDA**
   - Install NVIDIA GPU drivers.
   - Verify CUDA installation.

5. **Install Required Packages**
   ```bash
   pip install -r requirements.txt
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

### Training

1. **Test YOLOv8 Inference by downloading the models from Ultralytics**
   ```bash
   yolo detect predict model=yolov8l.pt source="path/to/image.png"
   yolo detect predict model=yolov8l.pt source="path/to/road.mp4"
   yolo detect predict model=yolov8l.pt source=0
   ```

2. **Get Labelled Dataset from Roboflow**
   - Upload images to Roboflow and label them as either `fall` or `nofall`.
   - Discard any images that are not relevant by marking them as null.
   - Download the structured dataset from Roboflow and select YOLO for model type when prompted.

3. **Train the Model**

   - Modify the name for the current operation.
   - Adjust the parameters value to properly utilize the GPU.

   ```bash
   yolo detect train model=yolov8l.pt data="path/to/dataset" imgsz=224 device=0 workers=2 batch=16 epochs=100 patience=50 name=yolov8_fallsafe_classification
   ```

4. **Continue Training after Pause OR Further Train model with new/updated Dataset**
   
   ```bash
   yolo detect train model=runs/detect/yolov8_fallsafe_classification/weights/last.pt resume=True
   ```

### Inference
   1. **Perform Classification**
   ```bash
   yolo detect predict model=runs/detect/yolov8_fallsafe_classification/weights/best.pt source="path/to/image.jpg" save=True
   yolo detect predict model=runs/detect/yolov8_fallsafe_classification/weights/best.pt source="path/to/video.mp4" save=True
   ```

   2. **Real-Time Classification via Camera**
      ```bash
      yolo detect predict model=runs/detect/yolov8_fallsafe_classification/weights/best.pt source="0" save=True conf=0.5
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

**Fall Safe** is developed by the above contributors. For more information, visit [our GitHub repository](https://github.com/FallSafe).
