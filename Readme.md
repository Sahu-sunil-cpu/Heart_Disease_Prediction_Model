# ❤️ Heart Disease Prediction using Logistic Regression

A Machine Learning project that predicts the likelihood of **Heart Disease** using **Logistic Regression** trained on the **Heart Disease Dataset**. The project demonstrates an end-to-end machine learning workflow including data preprocessing, exploratory data analysis (EDA), feature scaling, model training, evaluation, model interpretation, and deployment using **Streamlit**.

One of the key objectives of this project is not only to predict whether a patient is at risk of heart disease but also to **identify and explain the most important health indicators driving heart-disease risk**, making the model more interpretable for healthcare screening.

---

# 🚀 Live Demo

The application is deployed using **Streamlit Community Cloud**.

**Live Application:**  
*Add your Streamlit deployment link here.*

The application allows users to enter various clinical measurements such as age, blood pressure, cholesterol level, chest pain type, maximum heart rate, and other health indicators to predict the likelihood of heart disease in real time.

---

# 📖 Project Overview

Heart disease is one of the leading causes of death worldwide. Early identification of high-risk patients can help healthcare professionals provide timely diagnosis, preventive treatment, and lifestyle recommendations.

This project develops a **Logistic Regression classification model** capable of predicting whether a patient is likely to have heart disease based on clinical measurements collected during routine medical examinations.

The project follows a complete machine learning pipeline:

1. Data Loading
2. Data Inspection
3. Data Cleaning
4. Exploratory Data Analysis (EDA)
5. Feature Selection
6. Train-Test Split
7. Feature Scaling
8. Model Training
9. Model Evaluation
10. Health Indicator Analysis
11. Model Serialization
12. Streamlit Web Application Development
13. Deployment

---

# 🎯 Problem Statement

A hospital wants a first-pass screening model that flags patients at a higher risk of heart disease using clinical measurements such as blood pressure, cholesterol level, heart rate, and chest pain characteristics.

The goal is to assist healthcare professionals by providing an initial prediction that can support further medical evaluation.

---

# 💼 Business Objective

Develop a binary classification model capable of predicting the presence of heart disease using patient clinical information.

The model should:

- Identify patients at higher risk of heart disease.
- Support early screening and preventive healthcare.
- Minimize missed positive cases by emphasizing **Recall**, an important evaluation metric in healthcare applications.
- Provide interpretable predictions by identifying the clinical features that contribute most to heart-disease risk.

---

# ❤️ Why This Project Matters

Healthcare classification problems differ from many traditional machine learning tasks because prediction errors can have serious real-world consequences.

In heart disease prediction:

- **False Positives** may result in additional medical examinations.
- **False Negatives** are significantly more critical because patients with heart disease may remain undiagnosed and untreated.

For this reason, this project evaluates the model using multiple classification metrics such as **Accuracy**, **Precision**, **Recall**, and **F1-Score**, with particular emphasis on **Recall**, ensuring that as many high-risk patients as possible are correctly identified.

Additionally, Logistic Regression provides an interpretable model, allowing us to understand which health indicators contribute most to predicting heart disease.

---

# 📂 Dataset

This project uses the **Heart Disease Dataset**, one of the most widely used datasets for binary classification problems in healthcare.

The dataset contains clinical measurements collected from patients during routine cardiovascular examinations. These features are used to predict whether a patient has heart disease.

The dataset used in this project contains the following columns:

| Column | Description |
|---------|-------------|
| age | Age of the patient (years). |
| sex | Gender of the patient (1 = Male, 0 = Female). |
| cp | Chest pain type experienced by the patient. |
| trestbps | Resting blood pressure (mm Hg). |
| chol | Serum cholesterol (mg/dl). |
| fbs | Fasting blood sugar > 120 mg/dl (1 = True, 0 = False). |
| restecg | Resting electrocardiographic results. |
| thalach | Maximum heart rate achieved during exercise. |
| exang | Exercise-induced angina (1 = Yes, 0 = No). |
| oldpeak | ST depression induced by exercise relative to rest. |
| slope | Slope of the peak exercise ST segment. |
| ca | Number of major vessels colored by fluoroscopy. |
| thal | Thalassemia status. |
| target | Presence of heart disease (Target Variable). |

The dataset is stored inside:

```text
Dataset/
└── heart.csv
```

---

# 📊 Understanding the Dataset Columns

## 1. Age

Represents the age of the patient in years.

**Example**

```
29
45
58
67
```

Increasing age is one of the major risk factors associated with cardiovascular disease.

---

## 2. Sex

Represents the biological gender of the patient.

| Value | Meaning |
|------:|---------|
| 0 | Female |
| 1 | Male |

---

## 3. Chest Pain Type (`cp`)

Describes the type of chest pain experienced by the patient.

| Value | Meaning |
|------:|---------|
| 0 | Typical Angina |
| 1 | Atypical Angina |
| 2 | Non-anginal Pain |
| 3 | Asymptomatic |

Chest pain characteristics are among the strongest indicators used in diagnosing heart disease.

---

## 4. Resting Blood Pressure (`trestbps`)

Blood pressure measured while the patient is at rest.

**Unit:** mm Hg

Example:

```
120
130
145
160
```

Higher resting blood pressure may indicate an increased cardiovascular risk.

---

## 5. Serum Cholesterol (`chol`)

Represents cholesterol concentration in the blood.

**Unit:** mg/dl

Example:

```
180
220
260
310
```

Elevated cholesterol levels can contribute to plaque buildup in arteries.

---

## 6. Fasting Blood Sugar (`fbs`)

Indicates whether fasting blood sugar exceeds **120 mg/dl**.

| Value | Meaning |
|------:|---------|
| 0 | No |
| 1 | Yes |

High fasting blood sugar may be associated with diabetes, a known cardiovascular risk factor.

---

## 7. Resting ECG (`restecg`)

Represents the patient's resting electrocardiogram results.

| Value | Meaning |
|------:|---------|
| 0 | Normal |
| 1 | ST-T Wave Abnormality |
| 2 | Left Ventricular Hypertrophy |

---

## 8. Maximum Heart Rate (`thalach`)

Represents the maximum heart rate achieved during exercise testing.

Example:

```
120
145
165
185
```

Lower maximum heart rates during exercise may indicate impaired cardiovascular function.

---

## 9. Exercise-Induced Angina (`exang`)

Indicates whether the patient experienced chest pain during exercise.

| Value | Meaning |
|------:|---------|
| 0 | No |
| 1 | Yes |

Exercise-induced angina is an important clinical indicator of heart disease.

---

## 10. Old Peak (`oldpeak`)

Represents ST depression induced by exercise relative to rest.

Example:

```
0.0
1.5
2.8
4.2
```

Higher values generally indicate a greater likelihood of heart disease.

---

## 11. Slope

Represents the slope of the peak exercise ST segment.

| Value | Meaning |
|------:|---------|
| 0 | Upsloping |
| 1 | Flat |
| 2 | Downsloping |

---

## 12. Number of Major Vessels (`ca`)

Represents the number of major blood vessels colored by fluoroscopy.

Possible values:

```
0
1
2
3
4
```

Higher values generally indicate more severe cardiovascular abnormalities.

---

## 13. Thalassemia (`thal`)

Represents the patient's thalassemia test result.

| Value | Meaning |
|------:|---------|
| 0 | Unknown |
| 1 | Normal |
| 2 | Fixed Defect |
| 3 | Reversible Defect |

Abnormal thalassemia results are strongly associated with heart disease.

---

## 14. Target Variable (`target`)

This is the variable the machine learning model learns to predict.

| Value | Meaning |
|------:|---------|
| 0 | No Heart Disease |
| 1 | Heart Disease |

All remaining features are used to estimate this target variable.

# 🔧 Data Preprocessing

Before training the machine learning model, the dataset was cleaned and prepared to ensure high-quality inputs for classification.

---

## Step 1: Load the Dataset

The dataset is loaded using **Pandas**.

```python
import pandas as pd

df = pd.read_csv("Dataset/heart.csv")
```

---

## Step 2: Inspect the Dataset

The dataset is inspected to understand its structure, data types, and summary statistics.

```python
df.head()

df.info()

df.describe()
```

This helps identify:

- Number of rows and columns
- Data types
- Missing values
- Numerical feature statistics

---

## Step 3: Check Missing Values

The dataset is checked for missing values.

```python
df.isnull().sum()
```

Fortunately, the Heart Disease Dataset contains **no missing values**, so no imputation was required.

---

## Step 4: Check Duplicate Records

Duplicate records can negatively affect model performance.

```python
df.duplicated().sum()
```

If duplicates are found, they are removed.

```python
df = df.drop_duplicates()
```

---

## Step 5: Feature Selection

The target variable is separated from the input features.

```python
X = df.drop("target", axis=1)

y = df["target"]
```

The model uses **13 clinical features** to predict heart disease.

---

# 📋 Features Used for Model Training

The model is trained using the following clinical measurements.

| Feature | Description |
|---------|-------------|
| age | Age of the patient |
| sex | Gender |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol |
| fbs | Fasting blood sugar |
| restecg | Resting ECG results |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels |
| thal | Thalassemia |

---

# 📊 Final Feature Set

| Feature Type | Count |
|--------------|------:|
| Numerical Features | 13 |
| Categorical Features | 0 |
| **Total Features** | **13** |

Since all features are already represented numerically, **no categorical encoding (One-Hot Encoding or Label Encoding)** was required.

---

# 📈 Exploratory Data Analysis (EDA)

Before training the model, exploratory data analysis was performed to better understand the dataset and identify relationships between the features and heart disease.

The following analyses were carried out:

- Dataset overview using `head()`, `info()`, and `describe()`
- Target class distribution
- Age distribution
- Correlation heatmap
- Feature distributions
- Relationship between clinical measurements and heart disease
- Box plots for important health indicators
- Identification of potential outliers

These analyses provided valuable insights into the data and helped validate the preprocessing steps before model training.

---

## Target Distribution

The target variable was visualized to understand the balance between patients with and without heart disease.

```python
sns.countplot(x="target", data=df)
plt.title("Heart Disease Distribution")
plt.show()
```

This visualization helps determine whether the dataset is balanced before training.

---

## Age Distribution

The distribution of patient ages was analyzed.

```python
sns.histplot(df["age"], kde=True)

plt.title("Age Distribution")

plt.show()
```

Understanding age distribution helps identify whether heart disease is more common among certain age groups.

---

## Correlation Heatmap

A correlation heatmap was created to study relationships among the clinical features.

```python
plt.figure(figsize=(12,8))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.show()
```

The heatmap helps identify:

- Strong positive correlations
- Strong negative correlations
- Highly related clinical measurements

---

## Box Plot Analysis

Box plots were used to compare important health indicators between patients with and without heart disease.

Examples include:

- Cholesterol
- Maximum Heart Rate
- ST Depression (Old Peak)

```python
sns.boxplot(x="target", y="chol", data=df)

sns.boxplot(x="target", y="thalach", data=df)

sns.boxplot(x="target", y="oldpeak", data=df)
```

These visualizations help identify differences in feature distributions across the two classes.

---

# 🤖 Model Training

After preprocessing, the dataset was divided into input features (`X`) and target labels (`y`).

```python
X = df.drop("target", axis=1)

y = df["target"]
```

The data was then split into training and testing sets using an **80:20 ratio**.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Using **stratify=y** ensures that both training and testing datasets maintain the same class distribution.

---

## Feature Scaling

Since Logistic Regression performs better when features are on a similar scale, the data was standardized using **StandardScaler**.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
```

Feature scaling improves optimization and helps the model converge more efficiently.

---

## Logistic Regression Model

A **Logistic Regression** classifier was trained using the scaled training data.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)
```

Logistic Regression is widely used for binary classification problems because it is:

- Simple and efficient
- Easy to interpret
- Fast to train
- Suitable for healthcare classification tasks

---

## Saving the Model

The trained model, fitted scaler, and feature names were saved using **Joblib** for deployment in the Streamlit application.

```python
import joblib

joblib.dump(model, "Model/model.pkl")
joblib.dump(scaler, "Model/scaler.pkl")
joblib.dump(list(X.columns), "Model/feature_names.joblib")
```

The Streamlit application loads these files to generate predictions without retraining the model.

---

# ❤️ Identifying the Most Important Health Indicators

One of the additional objectives of this project is to determine which clinical measurements contribute the most to heart disease prediction.

Since Logistic Regression is an interpretable machine learning algorithm, each feature has an associated coefficient that indicates its influence on the prediction.

Feature importance was obtained using the trained model coefficients.

```python
importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

importance["Absolute"] = importance["Coefficient"].abs()

importance = importance.sort_values(
    by="Absolute",
    ascending=False
)
```

The coefficients were then visualized using a horizontal bar chart.

```python
sns.barplot(
    data=importance,
    x="Coefficient",
    y="Feature"
)

plt.title("Feature Importance")

plt.show()
```

The analysis revealed that features such as **Chest Pain Type (`cp`)**, **Number of Major Vessels (`ca`)**, **Thalassemia (`thal`)**, **ST Depression (`oldpeak`)**, and **Maximum Heart Rate (`thalach`)** were among the most influential predictors of heart disease.

This interpretability helps healthcare professionals better understand the model's predictions and supports informed clinical decision-making.

# 📊 Model Evaluation

After training the Logistic Regression model, its performance was evaluated using multiple classification metrics on the test dataset.

Unlike regression problems, classification models are evaluated based on how accurately they classify each observation into the correct category.

Since this project focuses on healthcare screening, **Recall** is one of the most important metrics because missing a patient who actually has heart disease can have serious real-world consequences.

---

# 📈 Classification Report

| Class | Precision | Recall | F1-Score | Support |
|------:|----------:|--------:|----------:|--------:|
| No Heart Disease (0) | **0.89** | **0.70** | **0.78** | **100** |
| Heart Disease (1) | **0.76** | **0.91** | **0.83** | **105** |

---

# 📋 Overall Performance

| Metric | Value |
|---------|------:|
| **Accuracy** | **81%** |
| **Precision (Heart Disease)** | **76%** |
| **Recall (Heart Disease)** | **91%** |
| **F1-Score (Heart Disease)** | **83%** |

---

## Accuracy

Accuracy represents the percentage of total predictions that were classified correctly.

```
Accuracy = 81%
```

This means the model correctly classified approximately **81% of all patients** in the test dataset.

---

## Precision

Precision measures how many patients predicted as having heart disease actually had the disease.

```
Precision = 76%
```

A precision of **76%** indicates that most positive predictions made by the model were correct.

---

## Recall

Recall measures how many actual heart disease patients were successfully identified.

```
Recall = 91%
```

This is the **most important metric** in this project.

A recall of **91%** means the model successfully detected **91 out of every 100 patients** who truly had heart disease.

High recall is especially desirable in healthcare because it minimizes **False Negatives**, reducing the chance of missing patients who require medical attention.

---

## F1-Score

The F1-score combines Precision and Recall into a single metric.

```
F1 Score = 83%
```

An F1-score of **83%** indicates a good balance between correctly identifying heart disease patients while limiting incorrect positive predictions.

---

# ❤️ Interpretation of Results

The Logistic Regression model achieved good overall performance for an initial heart disease screening system.

Some important observations include:

- The model correctly classified **81%** of the patients.
- It achieved an excellent **91% Recall**, meaning most patients with heart disease were successfully detected.
- A Precision of **76%** indicates that most positive predictions were accurate.
- The F1-score of **83%** demonstrates a strong balance between Precision and Recall.

From a healthcare perspective, prioritizing **Recall** is often preferred because failing to identify a patient with heart disease (False Negative) can delay diagnosis and treatment.

---

# 💻 Streamlit Web Application

A modern and interactive web application was developed using **Streamlit** to make the trained machine learning model easily accessible.

The application allows healthcare professionals or users to enter clinical measurements through an intuitive interface and instantly receive a heart disease risk prediction.

---

# ✨ Features

- ❤️ Modern medical-themed interface
- 🩺 Interactive patient information form
- 📊 Real-time heart disease prediction
- 📈 Risk probability gauge
- 📉 Prediction confidence visualization
- 📋 Patient summary table
- 📊 Feature importance chart
- 💡 Health indicator explanation
- ⚡ Fast prediction using the trained model
- 📱 Responsive dashboard layout

---

# 📷 Application Preview

## Home Page

> Replace the image below with a screenshot of your application's home page.

```text
images/home.png
```

```markdown
![Home Page](images/home.png)
```

---

## Prediction Result

> Replace the image below with a screenshot of the prediction result page.

```text
images/prediction.png
```

```markdown
![Prediction Result](images/prediction.png)
```

---

## Feature Importance Visualization

> Replace the image below with a screenshot of the feature importance graph.

```text
images/feature_importance.png
```

```markdown
![Feature Importance](images/feature_importance.png)
```

---

# ❤️ Most Important Health Indicators

One of the major objectives of this project was to identify the clinical features that contribute most to predicting heart disease.

The Logistic Regression coefficients indicate the influence of each feature on the prediction.

The following health indicators were found to have the greatest impact on heart disease risk:

| Health Indicator | Importance |
|------------------|------------|
| Chest Pain Type (`cp`) | Very High |
| Number of Major Vessels (`ca`) | Very High |
| Thalassemia (`thal`) | High |
| ST Depression (`oldpeak`) | High |
| Maximum Heart Rate (`thalach`) | High |

---

## Explanation of Key Indicators

### 🫀 Chest Pain Type (`cp`)

Chest pain is one of the strongest indicators of cardiovascular disease.

Different types of chest pain provide valuable information about blood flow to the heart and significantly influence the prediction.

---

### 🩸 Number of Major Vessels (`ca`)

This feature represents the number of major blood vessels visible during fluoroscopy.

Higher values generally indicate more severe cardiovascular abnormalities, making this one of the strongest predictors.

---

### 🧬 Thalassemia (`thal`)

Abnormal thalassemia test results are highly associated with heart disease.

Patients with fixed or reversible defects tend to have a higher predicted risk.

---

### 📉 ST Depression (`oldpeak`)

ST depression measures changes observed during exercise stress testing.

Higher values usually indicate reduced blood supply to the heart and increase the likelihood of heart disease.

---

### ❤️ Maximum Heart Rate (`thalach`)

Maximum heart rate achieved during exercise reflects cardiovascular performance.

Lower achievable heart rates during exercise may indicate underlying heart-related issues.

---

# 📁 Project Structure

```text
Heart-Disease-Prediction/
│
├── Dataset/
│   └── heart.csv
│
├── Model/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── feature_names.joblib
│
├── notebooks/
│   └── HeartDiseasePrediction.ipynb
│
├── images/
│   ├── home.png
│   ├── prediction.png
│   └── feature_importance.png
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

# 📂 Project Files

## Dataset/

Contains the original Heart Disease dataset used for training and testing the machine learning model.

---

## Model/

Stores the serialized machine learning model, fitted StandardScaler, and feature names.

| File | Description |
|------|-------------|
| model.pkl | Trained Logistic Regression model |
| scaler.pkl | StandardScaler used during training |
| feature_names.joblib | List of feature names used by the trained model |

---

## train.py

Responsible for the complete machine learning pipeline including:

- Loading the dataset
- Data cleaning
- Removing duplicate records
- Exploratory Data Analysis (EDA)
- Feature selection
- Train-test splitting
- Feature scaling
- Logistic Regression model training
- Model evaluation
- Feature importance analysis
- Saving the trained model

---

## app.py

Contains the complete Streamlit web application.

Responsibilities include:

- Loading the trained model
- Loading the fitted scaler
- Loading feature names
- Collecting patient information
- Scaling user inputs
- Predicting heart disease risk
- Displaying prediction probability
- Displaying feature importance
- Providing an interactive dashboard

---

## requirements.txt

Lists all required Python packages needed to run the project.

Example:

```txt
streamlit
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
joblib
```

---

# ⚙️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Pandas | Data Processing |
| NumPy | Numerical Computation |
| Matplotlib | Data Visualization |
| Seaborn | Statistical Visualization |
| Plotly | Interactive Charts |
| Scikit-learn | Machine Learning |
| Streamlit | Web Application |
| Joblib | Model Serialization |
| Git | Version Control |
| GitHub | Project Hosting |

---

# 🚀 Installation

Follow the steps below to run this project locally.

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/Heart-Disease-Prediction.git
```

Navigate to the project directory.

```bash
cd Heart-Disease-Prediction
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate the environment.

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Train the Model

Run the training script.

```bash
python train.py
```

This will generate:

```text
Model/
├── model.pkl
├── scaler.pkl
└── feature_names.joblib
```

---

## 5. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will automatically open in your browser.

By default:

```text
http://localhost:8501
```

---

# 🎯 Using the Application

After launching the Streamlit application:

1. Enter the patient's age.
2. Select the patient's gender.
3. Choose the chest pain type.
4. Enter the resting blood pressure.
5. Enter the cholesterol level.
6. Select fasting blood sugar status.
7. Select resting ECG result.
8. Enter the maximum heart rate achieved.
9. Select whether exercise-induced angina is present.
10. Enter the ST depression value.
11. Select the slope of the ST segment.
12. Select the number of major vessels.
13. Select the thalassemia category.
14. Click **Predict Heart Disease**.

The application will display:

- ❤️ Heart Disease Prediction
- 📊 Risk Probability
- 📋 Patient Summary
- 📈 Probability Gauge
- 📉 Feature Importance Chart
- 💡 Important Health Indicators

---

# 🔄 Machine Learning Workflow

```text
               Heart Disease Dataset
                        │
                        ▼
                Data Cleaning
                        │
                        ▼
          Exploratory Data Analysis
                        │
                        ▼
             Feature Selection
                        │
                        ▼
          Train-Test Split (80:20)
                        │
                        ▼
             StandardScaler
                        │
                        ▼
          Logistic Regression
                        │
                        ▼
            Model Evaluation
                        │
                        ▼
      Feature Importance Analysis
                        │
                        ▼
         Save Model (Joblib)
                        │
                        ▼
        Streamlit Deployment
```

---

# 📈 Model Pipeline

```text
      User Input
           │
           ▼
   Input Validation
           │
           ▼
   Feature Scaling
           │
           ▼
 Logistic Regression
           │
           ▼
 Heart Disease Prediction
           │
           ▼
 Risk Probability
           │
           ▼
 Display Dashboard
```

---

# 🌟 Key Features

- End-to-End Machine Learning Pipeline
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Scaling using StandardScaler
- Logistic Regression Classification
- Feature Importance Analysis
- Health Indicator Interpretation
- Model Serialization using Joblib
- Interactive Streamlit Dashboard
- Real-Time Heart Disease Prediction
- Risk Probability Visualization
- Patient Summary Dashboard
- Modern Medical-Themed UI

---

# 🔮 Future Improvements

Some possible enhancements for future versions include:

- Compare multiple classification algorithms such as:
  - Random Forest Classifier
  - XGBoost Classifier
  - Support Vector Machine (SVM)
  - Decision Tree Classifier
  - Gradient Boosting Classifier
- Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
- Add ROC Curve and Precision-Recall Curve visualizations.
- Integrate SHAP or LIME for advanced model explainability.
- Improve prediction performance using feature engineering.
- Deploy the application using Docker.
- Connect to a cloud database for storing prediction history.
- Add user authentication for healthcare professionals.
- Integrate electronic health record (EHR) systems.

---

# 📚 Learning Outcomes

This project demonstrates practical experience with:

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Classification Problems
- Logistic Regression
- Feature Scaling
- Classification Evaluation Metrics
- Precision, Recall, and F1-Score
- Model Interpretation
- Feature Importance Analysis
- Model Serialization using Joblib
- Streamlit Dashboard Development
- End-to-End Machine Learning Pipeline

---

# 🙋 Author

**Sunil Kumar Sahu**

Computer Science and Engineering Student

Passionate about:

- Machine Learning
- Artificial Intelligence
- Full Stack Development
- Backend Development
- Data Science
- Cloud Computing

### Connect with Me

- **GitHub:** https://github.com/Sahu-sunil-cpu
- **LinkedIn:** https://www.linkedin.com/in/sunil-kumar-sahu
- **Email:** sahusunilcpu@gmail.com

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.

It helps others discover the project and motivates further improvements.

---

# 📄 License

This project is released under the **MIT License**.

You are free to use, modify, and distribute this project for educational and personal purposes.

---

## Thank You!

Thank you for checking out this project.

If you have any suggestions, feedback, or ideas for improvement, feel free to open an issue or submit a pull request.

Happy Coding! 🚀