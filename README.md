[![Typing SVG](https://readme-typing-svg.demolab.com?font=Sour+Gummy&weight=900&size=30&pause=1000&color=E6EAFF&background=4C186A&center=true&width=800&height=90&lines=Out+of+Distribution+Plant+Disease+Identification)](https://git.io/typing-svg)

## Dataset Overview and Project Focus

In this repository, you can find the datasets used in our project on **Out-of-Distribution (OOD) and Skewed Data**, 
and the exploration of **Zero-Shot Learning** as a possible technique to address these challenges.

The datasets can be accessed via the following link:  
[Dataset Link - Google Drive](https://drive.google.com/drive/folders/1atwfzhMEn2P4jwJqeQUJjpvfdFT1tS_v)

Our project aims to delve into how OOD and skewed data can impact machine learning models and 
investigate strategies, such as **Zero-Shot Learning**, to mitigate these issues.

---

### Dataset Breakdown

The datasets consist of the following:

| **Dataset**      | **Train Images** | **Test Images** | **Total Images** |
|------------------|------------------|-----------------|------------------|
| **PlantVillage** | 38,994           | 10,495          | 49,489           |
| **PlantDoc**     | -                | 71              | 71               |

---

### Dataset File Contents

The dataset contains several CSV files with metadata related to the images. These files are structured as follows:

| **File/Folder**                | **Content**                                                                                              |
|---------------------------------|----------------------------------------------------------------------------------------------------------|
| `PV train.csv`                  | List of laboratory training images with ground truth from the PlantVillage dataset.                       |
| `PV test seen.csv`              | List of seen testing images with ground truth from the PlantVillage dataset.                             |
| `PV test unseen.csv`            | List of unseen testing images with ground truth from the PlantVillage dataset.                           |
| `PD test unseen.csv`            | List of unseen testing images with ground truth from the PlantDoc dataset.                               |

Each CSV file contains the following columns:

1. **Image File Name**: The name of the image file.
2. **Crop Category (Ground Truth)**: The category of the crop (there are 14 crop categories).
3. **Disease Category (Ground Truth)**: The category of the disease affecting the crop (there are 21 disease categories).

The crop and disease categories are combined to form a single composition. The index for both crop and disease categories starts from 0.

---

# How to run the codes?

Hello readers, in order to use our proposed architecture follow the detailed instructions below
to train the model locally on your device.

## Setting Up a Virtual Environment Using Miniconda

This guide explains how to set up a virtual environment named `mysha` 
using an `environment.yml` file in Miniconda. The environment will include Python, 
CUDA, CuDNN, and a variety of scientific and machine learning libraries.

### Step 1: Install Miniconda
First, you need to install Miniconda, a minimal version of Anaconda, which allows you to create and manage Python environments.

1. Go to the [Miniconda download page](https://docs.conda.io/en/latest/miniconda.html).
2. Download the installer for your operating system:
   - For **Windows**: Download `Miniconda3-latest-Windows-x86_64.exe`.
   - For **macOS**: Download the macOS installer (`.pkg` file).
   - For **Linux**: Download the corresponding `.sh` file.
3. Follow the installation instructions for your operating system.

### Step 2: Create a Virtual Environment Using `environment.yml`
After installing Miniconda, you can create a virtual environment using the provided `environment.yml` file. This file specifies the dependencies and channels required to set up the environment.

1. Save the following `environment.yml` file:

```

   name: mysha 
   channels:
     - conda-forge
     - defaults
   dependencies:
     - python=3.9
     - cudatoolkit=11.2
     - cudnn=8.1.0
     - numpy=1.23.5                 # Downgraded NumPy version
     - pandas=2.2.3                 # Pinning to ensure compatibility
     - matplotlib=3.5.1             # Pinning to ensure compatibility
     - scikit-learn=1.2.2           # Pinning to ensure compatibility
     - pillow=10.4.0                # Updated to match current installation
     - seaborn=0.12.2               # Added Seaborn
     - pip
     - pip:
       - tensorflow<2.11
       - imbalanced-learn==0.10.1    # Updated imbalanced-learn
       - jupyter
       - ipykernel
       # Add other pip-installed packages here
	   
```

2. Open a terminal and navigate to the directory where the `environment.yml` file is saved.
3. Run the following command to create the environment:

```

conda env create -f environment.yml

```

### Step 3: Activate the Environment

Once the environment is successfully created, activate it with the following command:

```

conda activate mysha

```

### Step 4: Start Jupyter Notebook

Run the following command:

```

jupyter notebook

```


### Step 5: getting the codes and running to the training

To get the codes, git clone this directory

```

git clone https://github.com/myndaaa/Out-Of-Distribution-Plant-Disease-Identification--COS300082.git


```

copy the .ipynb codes and open them in your jupyter notebook <br>
Download the required dataset from the drive link at the top of this file <br>
In the code bases change the directories of the datasets to your actual paths. And also the directories
where you want to save your models<br>
and hit run!<br>

----

# Contributors


<table>
  <tr>
    <th colspan="3">Who Did What</th>
  </tr>
  <tr>
    <th>Contributor</th>
    <th>Contribution</th>
    <th>Percent Contribution</th>
  </tr>
  <tr>
    <td rowspan="10">Mysha Nahiyan Shemontee</td>
    <td>Exploratory models - ResNet50</td>
    <td>100%</td>
  </tr>
  <tr>
    <td>Baseline model (crop) - ResNet50v2</td>
    <td>100%</td>
  </tr>
  <tr>
    <td>Baseline model (disease) - ResNet50v2</td>
    <td>100%</td>
  </tr>
  <tr>
    <td>Baseline model (disease) - DenseNet102</td>
    <td>100%</td>
  </tr>
  <tr>
    <td>Baseline model (crop) - DenseNet102</td>
    <td>100%</td>
  </tr>
  <tr>
    <td>Proposed model</td>
    <td>100%</td>
  </tr>
  <tr>
    <td>Ensemble learning (crop baseline model)</td>
    <td>100%</td>
  </tr>
  <tr>
    <td>Ensemble learning (disease baseline model)</td>
    <td>100%</td>
  </tr>
  <tr>
    <td>Jetson Nano embedded system</td>
    <td>100%</td>
  </tr>
  <tr>
    <td>Web interface deployment</td>
    <td>100%</td>
  </tr>
  <tr>
    <td rowspan="2">Voong Zhe Hong</td>
    <td>Exploratory CNN (crop model)</td>
    <td>100%</td>
  </tr>
  <tr>
    <td>Exploratory CNN model pipeline</td>
    <td>100%</td>
  </tr>
  <tr>
    <td>Kelvin Chen Wei Lung</td>
    <td>Exploratory CNN (disease model)</td>
    <td>100%</td>
  </tr>
</table>
