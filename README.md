# 🩻 MAMMO_WORKSHOP  

**Reproducibility Workshop for the Undergraduate Thesis:**  
**"System Based on Convolutional Neural Networks for Breast Cancer Detection"**

This repository contains the implementation, adaptation, and reproducibility of a system based on **convolutional neural networks (CNNs)** for the detection of breast cancer in mammography. The work is divided into two main stages, documented in Jupyter Notebooks, and supported by pretrained models, pre-processing scripts, and the necessary resources to replicate the experiments.  

---

## 👩‍🎓 Authors & Advisor
- **Luis Fernando Sánchez Fuentes**  
- **Santiago Enrique Monsalve Durán**  
  Students at **Universidad Industrial de Santander (UIS)**  

**Advisor:** Said Pertuz  
**Research Group:** [Connectivity and Signal Processing Research Group (CPS)](https://uis.edu.co/ffm-gruinv-cps-en/)  

---

## 📂 Project Structure  
```
MAMMO_WORKSHOP/
│
├── Articles/ # Reference articles in PDF
├── Assets/ # Images and figures used in notebooks
├── Installers/ # Python, Conda and ImageMagick installers
├── Models/ # Pretrained models for inference
├── RESULTS/ # Results generated during experiments
│
├── generate_predictions.py # Script to run inference
├── preprocess.py # Script for dataset preprocessing
├── mammo_workshop_p1.ipynb # Notebook - Part 1: Setup & PKL reading
├── mammo_workshop.ipynb # Notebook - Part 2: Full pipeline
│
├── nyu_controls.pkl # Control group annotations
├── nyu_gmic.pkl # Cases annotations
├── requirements.txt # List of dependencies
```
⚠️ **Note:** The dataset is not included in the repository. It must be downloaded from the link provided in the notebooks and placed in the project root directory.  

---

## ⚠️ Important Note on Datasets and Project Structure  

This repository does **not** include the dataset.  
Please download the `.pkl` files from the link provided in the notebooks and place them **directly in the root folder of the project (`MAMMO_WORKSHOP/`)**, at the same level as `preprocess.py` and the Jupyter notebooks.  

The folder structure shown above must be respected, even if some directories are initially empty.  

✅ To ensure everything is in place before running the notebooks, execute the following block in a Jupyter cell:

```
python
import os

required_files = ["nyu_controls.pkl", "nyu_gmic.pkl"]
required_dirs = ["Articles", "Assets", "Installers", "Models", "RESULTS"]

# Check files
for f in required_files:
    if not os.path.isfile(f):
        print(f"❌ Missing file: {f}")
    else:
        print(f"✅ Found: {f}")

# Check directories
for d in required_dirs:
    if not os.path.isdir(d):
        print(f"❌ Missing directory: {d}")
    else:
        print(f"✅ Found directory: {d}")

```        

## 🛠️ Technologies  
- Python **3.7.0**  
- TensorFlow 2.8.1  
- Protobuf 3.20.1  
- OpenCV  
- Wand 0.6.11  
- Matplotlib  
- Pandas  
- Scikit-learn  
- tqdm  

---

## ⚙️ Installation  

We recommend creating a **conda environment** and installing dependencies **one by one** (instead of using `pip install -r requirements.txt`) since errors may occur with bulk installation.  
Also you can follow the jupyter notebook instructions to replicate.

```bash
# Create and activate environment
conda create -n mammo python=3.7.0
conda activate mammo

# Install dependencies manually
pip install tensorflow==2.8.1
pip install protobuf==3.20.1
pip install opencv-python
pip install wand==0.6.11
pip install matplotlib
pip install pandas
pip install scikit-learn
pip install tqdm


🚀 Usage
1. Part 1 – Environment setup & PKL reading

Notebook: mammo_workshop_p1.ipynb

Environment verification.

Installation of dependencies.

Reading .pkl files (nyu_controls.pkl & nyu_gmic.pkl).

Exploration of dataset structure.

2. Part 2 – Full preprocessing & inference

Notebook: mammo_workshop.ipynb

Preprocessing pipeline for mammograms.

Model loading and inference.

Evaluation with clinical labels.


🎯 Objectives

Ensure reproducibility of the thesis experiments.

Provide a clear and structured workflow for researchers and students.

Assess model robustness and transferability across heterogeneous datasets.


📖 References

Shen L, Margolies LR, Rothstein JH, Fluder E, McBride R, Sieh W. Deep Learning to Improve Breast Cancer Detection on Screening Mammography. Scientific Reports. 2019; 9:12495.

Scientific Reports. Automated breast cancer detection in digital mammograms using deep learning. 2023.
