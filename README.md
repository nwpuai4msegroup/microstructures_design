# microstructures_design
Generative design of high-fidelity microstructures using physics-aware machine learning

Title: Source Code for “Generative design of high-fidelity microstructures using physics-aware machine learning”

Author: Weijie Liao, Ruihao Yuan, et al.

### 1. Software Environment

The source code was developed and tested under the following environment:

* Operating System: Linux / Windows 10
* Python Version: 3.8.3

### 2. Required Python Packages

The following packages are required:

numpy==1.22.4
pandas==1.5.3
scikit-learn==0.23.1
matplotlib==3.7.5
torch==1.13.1+cu116
torchvision==0.14.1+cu116
joblib==0.16.0
opencv-python==4.5.5.64
pymoo==0.6.1.1
seaborn==0.10.1

### 3. Directory Structure

├── examples/                  # Input examples
├── model_save/                # Trained models parameters
├── dataset.csv/               # Original material performance data
├── EBSD_generation.ipynb      # Image generation code and examples
├── inverse_design.ipynb       # Reverse optimization code and examples
├── rFunction.py               # Model and related function code
├── z_i.csv                    # Latent space data
└── README.txt

### 4. Model trained parameter acquisition

The trained VAEs model used in this work are available on Google Drive:

https://drive.google.com/file/d/1jS_QCfnW4Kpxf5PUZlSts73PrkoqOAXO/view?usp=drive_link

Place the downloaded files in the “model_save” folder.

### 5. Contact
﻿
For questions regarding the code, please contact:
﻿
Email: rhyuan@nwpu.edu.cn
