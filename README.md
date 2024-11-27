# Knn_projects
# Breast Cancer Classification

This repository contains a machine learning project for classifying breast cancer cases using the Breast Cancer dataset from Kaggle. The models tested include **K-Nearest Neighbors (KNN)**, **Naive Bayes (GaussianNB)**, and **Support Vector Machines (SVM)**.

## Table of Contents
- [Dataset](#dataset)
- [Installation](#installation)
- [Data Exploration](#data-exploration)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
- [Results](#results)
- [License](#license)

---

## Dataset

The dataset used is available on Kaggle: [Breast Cancer Dataset](https://www.kaggle.com/datasets/erdemtaha/cancer-data).  
It contains information about the mean, standard error, and "worst" (largest) values for various measurements of breast cell nuclei computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.

### Dataset License
The dataset is licensed under **CC-BY-NC-SA-4.0**.

---

## Installation

### Prerequisites
Ensure you have Python and the following libraries installed:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

### Steps to Get Started
1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/breast-cancer-classification.git
    ```
2. Navigate to the project folder:
    ```bash
    cd breast-cancer-classification
    ```
3. Install required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the dataset using Kaggle's API:
    ```bash
    kaggle datasets download -d erdemtaha/cancer-data
    ```
5. Extract the dataset:
    ```python
    import zipfile
    zipfile_path = 'cancer-data.zip'
    with zipfile.ZipFile(zipfile_path, 'r') as file:
        file.extractall()
    ```

---

## Data Exploration

### Data Information
- **Number of Rows**: 569
- **Number of Columns**: 33
- **Target Column**: `diagnosis` (M = Malignant, B = Benign)

### Key Insights:
- The `id` column and an unnamed column were dropped as they do not provide meaningful information.
- A correlation heatmap was generated to understand the relationships between numerical features.

**Correlation Heatmap**  
![Heatmap](assets/heatmap.png) *(Replace with actual heatmap image if available)*

---

## Data Preprocessing

### Steps
1. Dropped unnecessary columns: `Unnamed: 32` and `id`.
2. Split data into features (`X`) and target (`y`).
3. Split the dataset into training and testing sets (80% training, 20% testing).
4. Scaled the features using `StandardScaler` for model compatibility.

---

## Modeling

### Models Used
1. **K-Nearest Neighbors (KNN)**
   - Parameters: `n_neighbors=5`, `weights='uniform'`
   - Cross-validation Accuracy: **96.04%**

2. **Naive Bayes (GaussianNB)**
   - Cross-validation Accuracy: **93.41%**

3. **Support Vector Machine (SVM)**
   - Parameters: `C=0.9`, `kernel='poly'`, `degree=5`, `class_weight='balanced'`
   - Cross-validation Accuracy: **94.51%**

### Code Snippets
- Example of model pipeline creation:
    ```python
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier

    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights='uniform'))
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    print(scores.mean())
    ```

---

## Results

| Model       | Accuracy (%) |
|-------------|--------------|
| KNN         | 96.04        |
| Naive Bayes | 93.41        |
| SVM         | 94.51        |

The **K-Nearest Neighbors (KNN)** model achieved the highest accuracy in this classification task.

---

## License

This project uses the dataset licensed under **CC-BY-NC-SA-4.0**. The code in this repository is released under the MIT License.  

Feel free to fork and use it in your own projects.

---

## Contact

For any questions or suggestions, feel free to reach out:
- **Email**: your.email@example.com
- **GitHub**: [yourusername](https://github.com/yourusername)

