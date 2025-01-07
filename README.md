
# **DeepProg**

DeepProg is a comprehensive framework for analyzing multi-omics cancer data using machine learning and deep learning methods. This repository contains the essential scripts, sample data, and instructions for reproducing results or generating new analyses.

---

## **Requirements**

### **Essential Packages**
To run DeepProg and generate results, you need to install the essential packages listed in `requirements.txt`. These packages are the minimum dependencies required for running the framework.

```bash
pip install -r requirements.txt
```

### **Full Package Versions**
For full reproducibility purposes, the complete list of package versions used during the development and testing of DeepProg is provided in `requirements_full.txt`. This ensures that you can replicate the exact environment used for our experiments.

```bash
pip install -r requirements_full.txt
```

---

## **Generating Results**


To generate results, first download data from **TCGA** or use [our sample data](#sample-data), and place it in the directory:

/data/<cancer_type>/

Once the data is prepared, use the following command:

```bash
python DeepProgRunner/generate_result.py -D=<cancer_type> -S=<method>
```

- `-D` specifies the **cancer type** (e.g., `ACC`, `HCC`).
- `-S` specifies the **method** (omics of your choice).  
  Detailed descriptions of the available methods can be found inside the `generate_result.py` script.

For example:
We downloaded ACC sample data from the google drive, and put it into 

```bash
/data/ACC/
```

Then, we ran the following command:
```bash
python DeepProgRunner/generate_result.py -D=ACC -S=1
```
This will return ACC cancer type baseline c-index result.

---

## **Sample Data**

Due to the large size of the datasets, sample data is provided externally via Google Drive.  
Currently, we have data for the following cancer types:
- **ACC** (Adrenocortical Carcinoma)
- **HCC** (Hepatocellular Carcinoma)

You can download the sample data from the following link:  
**[Sample Data on Google Drive](https://drive.google.com/drive/folders/13kZgIBd9ehfOVBJ2hylf41ld0bRJ_8n3?usp=drive_link)**

Place the downloaded data in the appropriate folder as specified in `generate_result.py` before running the script.

---

## **Results Overview**

### **C-index Results**
The **C-index results** for each cancer type and method are provided in the file:

```
/result/C_index_result.csv
```

This file contains the **C-index values** for 50 iterations for each cancer type and method combination, allowing users to analyze the robustness and performance of the models.

### **Plots**
All plots generated from the analysis can be found in the folder:

```
/Plots
```

The scripts used to generate these plots are also available in the same folder, enabling users to reproduce or modify the visualizations as needed.

---

## **More Information**

For more details on more advanced use or installation of DeepProg, you can find them in:  **DeepProg GitHub repository**:  
**[DeepProg Repository](https://github.com/lanagarmire/DeepProg/tree/master)**

---

## **Contact**
* Developer: Bowei Li
* contact: bl3571@nyu.edu
For any issues, questions, or suggestions, please contact the repository maintainers or open an issue in this GitHub repository.
