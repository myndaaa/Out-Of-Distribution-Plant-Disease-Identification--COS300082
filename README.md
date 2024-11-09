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
