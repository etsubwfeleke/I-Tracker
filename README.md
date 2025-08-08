# iTracker: Gaze Estimation with PyTorch

This project is an implementation of a gaze tracking model using PyTorch, with a focus on training and evaluating on the [GazeCapture dataset](http://gazecapture.csail.mit.edu/). The primary goal of this repository is to provide a clear and well-documented pipeline for gaze estimation, from data preprocessing to model evaluation. This work is heavily guided by the original [GazeCapture PyTorch implementation](https://github.com/CSAILVision/GazeCapture/tree/master/pytorch) by the Computer Science and Artificial Intelligence Laboratory (CSAIL) at MIT. We aim to build upon their foundational work, offering a transparent and accessible resource for researchers and developers interested in gaze tracking.

## Key Features

* **Data Preprocessing:** Scripts to handle and prepare the GazeCapture dataset for training
* **PyTorch Model:** A deep learning model for gaze estimation, adapted from the original iTracker architecture
* **Training Pipeline:** Complete training and evaluation workflow with checkpointing
* **GPU Acceleration:** Support for Apple's MPS backend for faster training
* **Modular Design:** Clear separation of data loading, model architecture, and training components

## Technologies Used

* **Python 3.x**
* **PyTorch:** The core deep learning framework
* **NumPy:** For numerical operations
* **Pandas:** For data manipulation and analysis
* **OpenCV:** For image processing tasks

## Dataset

This project uses the **GazeCapture** dataset, which is one of the largest publicly available datasets for gaze estimation. It contains over 2.4 million frames from more than 1,400 participants.

* **Download the dataset:** You can download the dataset from the official GazeCapture website: [http://gazecapture.csail.mit.edu/](http://gazecapture.csail.mit.edu/)
* **Data Structure:** Once downloaded, place the dataset in a `data/` directory within the project's root folder. The expected structure is:
    ```
    iTracker/
    ├── data/
    │   └── gazecapture/
    │       ├── 00002/
    │       │   ├── appleFace.json
    │       │   ├── appleLeftEye.json
    │       │   └── ...
    │       └── ...
    ├── src/
    └── ...
    ```

## Project Overview

This project implements a deep learning-based gaze tracking system that can predict where a person is looking on a screen using images of their face and eyes. The implementation includes:

- Data preprocessing pipeline
- Neural network model architecture
- Training and evaluation scripts
- Utilities for handling the GazeCapture dataset

## Components

### 1. Data Preparation (`prepareDataset.py`)
- Processes the raw GazeCapture dataset
- Extracts and crops face and eye images
- Generates face grid representations
- Creates metadata.mat file containing:
  - Frame indices
  - Recording numbers
  - Gaze targets (X, Y coordinates)
  - Face grid information
  - Train/validation/test split information

### 2. Data Loading (`DataLoader.py`)
- Custom PyTorch Dataset implementation
- Handles loading and preprocessing of:
  - Left and right eye images
  - Face images
  - Face grid
- Implements data transformations and normalization
- Supports train/validation/test splits

### 3. Model Architecture (`iTrackerModel.py`)
- Implements the iTracker neural network with:
  - Separate CNN branches for left eye, right eye, and face
  - Face grid processing branch
  - Feature fusion and final regression layers
- Uses modern PyTorch conventions and best practices
- Supports GPU acceleration via Apple's MPS backend

### 4. Training Pipeline (`main.py`)
- Handles model training and evaluation
- Features:
  - Configurable hyperparameters
  - Model checkpointing
  - Training/validation/test loops
  - Gaze error calculation
  - Device-agnostic training (CPU/GPU)

## Setup and Usage

1. Download the GazeCapture dataset from [http://gazecapture.csail.mit.edu/](http://gazecapture.csail.mit.edu/)

2. Prepare the dataset:
   ```bash
   python prepareDataset.py --dataset_path /path/to/raw/dataset --output_path /path/to/output
   ```

3. Train the model:
   ```bash
   python main.py
   ```

## Model Architecture Details

The iTracker model consists of:
- Three parallel CNN branches for processing left eye, right eye, and face images
- A separate branch for processing the face grid
- Feature fusion layers that combine information from all branches
- Final regression layers that predict (x,y) gaze coordinates

## Performance Metrics

The model's performance is evaluated using:
- Mean Square Error (MSE) loss
- Average gaze error in degrees
- Training/validation/test split metrics

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{cvpr2016_gazecapture,
Author = {Kyle Krafka and Aditya Khosla and Petr Kellnhofer and Harini Kannan and Suchendra Bhandarkar and Wojciech Matusik and Antonio Torralba},
Title = {Eye Tracking for Everyone},
Year = {2016},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}
```

## Acknowledgments

* This project is heavily inspired by the original **GazeCapture** project and its authors:
    * ***K. Krafka, A. Khosla, P. Kellnhofer, H. Kannan, S. Bhandarkar, W. Matusik, and A. Torralba. "Eye Tracking for Everyone." IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.***
* We are grateful for their contribution to the field and for making their dataset and code publicly available.
This implementation is based on the work of the GazeCapture project by the Computer Science and Artificial Intelligence Laboratory (CSAIL) at MIT. We acknowledge their contributions and provide this code as a means to further research in gaze estimation and tracking.
## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/CSAILVision/GazeCapture/blob/master/LICENSE.md) file for details

## Future Work

Our current implementation lays the groundwork for more advanced eye-tracking applications. Here are our planned next steps:

### Phase 1: Dataset Expansion
- Scale up training to utilize the full GazeCapture dataset
  - Current implementation uses ~2% (~20K samples) of available data
  - Full dataset contains over 2.4M frames from 1,474 participants
  - Expected to significantly improve model robustness and generalization

### Phase 2: Real-time Implementation
- Develop webcam integration for live gaze tracking
- Optimize the inference pipeline for real-time processing
- Create an efficient frame capture and prediction system

### Phase 3: Technical Enhancements
- Integration of attention mechanisms
- Performance optimization strategies
- Mobile deployment solutions
- Cross-platform GPU support

### Phase 4: Architecture Improvements
- Enhanced regularization techniques
- Additional data augmentation methods
- Hyperparameter optimization for larger dataset

The immediate priority is expanding our dataset utilization, which will provide the foundation for robust real-time applications.
