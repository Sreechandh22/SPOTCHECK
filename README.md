# SpotCheck: Early-Stage Skin Cancer Detection

Welcome to the SpotCheck project repository. SpotCheck is a web application designed for early-stage skin cancer detection with high accuracy. Developed using Flask and TensorFlow, this project was a part of the HackMIT competition, where it secured second place.

## Table of Contents

- [Introduction](#introduction)
- [Project Objectives](#project-objectives)
- [Technology and Tools](#technology-and-tools)
- [Setup](#setup)
- [Usage](#usage)
  - [Running the Application](#running-the-application)
  - [Uploading an Image](#uploading-an-image)
- [File Structure](#file-structure)
- [License](#license)
- [Contact](#contact)

## Introduction

SpotCheck is a web application that detects early-stage skin cancer using machine learning algorithms. It leverages TensorFlow for model training and Flask for the web interface, achieving an impressive accuracy rate of 98%.

## Project Objectives

1. Detect early-stage skin cancer with high accuracy.
2. Provide a user-friendly web interface for uploading images and viewing results.
3. Ensure the application is accessible for early-stage skin cancer detection.

## Technology and Tools

- **Framework**: Flask
- **Machine Learning**: TensorFlow
- **Languages**: Python
- **Other Tools**: Numpy, Keras, HTML, CSS

## Setup

1. **Clone the repository**:

    ```sh
    git clone https://github.com/yourusername/SpotCheck.git
    cd SpotCheck
    ```

2. **Create a virtual environment and activate it**:

    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install the required packages**:

    ```sh
    pip install -r requirements.txt
    ```

4. **Ensure the pre-trained model is in the correct location**:

    Place `skin_cancer_model_v2.keras` in the root directory of the project.

## Usage

### Running the Application

1. **Start the Flask server**:

    ```sh
    python app.py
    ```

2. **Open your web browser and navigate to** `http://127.0.0.1:5000/`

### Uploading an Image

1. **Upload an image**: Choose a .jpg or .png image file of a skin lesion.
2. **Submit the image**: The app will process the image and provide a risk percentage for skin cancer.

## File Structure

SpotCheck/
├── Templates/
│ ├── index.html
│ ├── result.html
├── code/
│ ├── Anomaly_Detection.py
│ ├── Code_Generation.py
│ ├── Data_Visualization.py
│ ├── GAN_model.py
│ ├── GPT_Analysis.py
│ ├── User_Interaction.py
│ ├── Visual_Predict.py
│ ├── app.py
│ ├── convert_to_tflite.py
│ ├── hack1.py
│ ├── hack2.py
│ ├── kaggle.json
├── skin_cancer_model_v2.keras
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt


- **Templates/**: HTML templates for rendering web pages.
- **code/**: Contains various Python scripts for different functionalities.
  - `app.py`: Main application file for running the Flask server.
  - `convert_to_tflite.py`: Converts the model to TensorFlow Lite format.
  - Other scripts: Supportive scripts for data analysis and visualization.
- **skin_cancer_model_v2.keras**: The pre-trained model for skin cancer detection.
- **.gitignore**: Specifies files and directories to be ignored by Git.
- **LICENSE**: The project’s license information.
- **README.md**: Project overview and setup guide.
- **requirements.txt**: List of dependencies required to run the project.

## License

This project is licensed under the MIT License.

---

## Contact

For any inquiries or collaboration opportunities, please contact sreechandh2204@gmail.com
