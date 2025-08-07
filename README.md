# iTracker: Gaze Estimation with PyTorch

This project is an implementation of a gaze tracking model using PyTorch, with a focus on training and evaluating on the [GazeCapture dataset](http://gazecapture.csail.mit.edu/). The primary goal of this repository is to provide a clear and well-documented pipeline for gaze estimation, from data preprocessing to model evaluation.

This work is heavily guided by the original [GazeCapture PyTorch implementation](https://github.com/CSAILVision/GazeCapture/tree/master/pytorch) by the Computer Science and Artificial Intelligence Laboratory (CSAIL) at MIT. We aim to build upon their foundational work, offering a transparent and accessible resource for researchers and developers interested in gaze tracking.

## Key Features

* **Data Preprocessing:** Scripts to handle and prepare the GazeCapture dataset for training.
* **PyTorch Model:** A deep learning model for gaze estimation, adapted from the original iTracker architecture.
* **Transparency:** Openly acknowledges the use of the GazeCapture dataset and the original implementation as a reference.

## Demo

*(Coming Soon: A GIF or short video demonstrating the trained model's performance will be added here.)*

## Technologies Used

* **Python 3.x**
* **PyTorch:** The core deep learning framework.
* **NumPy:** For numerical operations.
* **Pandas:** For data manipulation and analysis.
* **OpenCV:** For image processing tasks.
* **Matplotlib:** For data visualization and plotting results.

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

## Project Status

This project is currently in the data preparation phase. The sections below will be updated as the project progresses.

### Training the Model
*(To be continued...)*

### Evaluating the Model
*(To be continued...)*

### Future Work
*(To be continued...)*

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments

* This project is heavily inspired by the original **GazeCapture** project and its authors:
    * *K. Krafka, A. Khosla, P. Kellnhofer, H. Kannan, S. Bhandarkar, W. Matusik, and A. Torralba. "Eye Tracking for Everyone." IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.*
* We are grateful for their contribution to the field and for making their dataset and code publicly available.
