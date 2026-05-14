# 🏠 House Price Prediction System

> An end-to-end Machine Learning project — from model research and training to a fully deployed production-style web application — built to intelligently predict residential property prices based on key housing features.

---

## 📑 Table of Contents

1. [Project Overview](#-project-overview)
2. [Part I — Machine Learning Model](#-part-i--machine-learning-model)
   - [Problem Statement](#problem-statement)
   - [Dataset](#dataset)
   - [Solution Approach](#solution-approach)
   - [Step-by-Step Training Pipeline](#step-by-step-training-pipeline)
     - [Chapter 1 — Linear Regression](#chapter-1--linear-regression)
     - [Chapter 2 — Gradient Boosting Regressor](#chapter-2--gradient-boosting-regressor)
     - [Chapter 3 — Model Comparison & Selection](#chapter-3--model-comparison--selection)
3. [Part II — Web Application](#-part-ii--web-application)
   - [Technology Stack](#technology-stack)
   - [Application Architecture](#application-architecture)
   - [Project Structure](#project-structure)
   - [Feature Preprocessing & Model Integration](#feature-preprocessing--model-integration)
   - [Application Features](#application-features)
   - [Installation & Setup](#installation--setup)
   - [Running the Application](#running-the-application)
4. [Screenshots](#-screenshots)
5. [Future Improvements](#-future-improvements)
6. [Author](#-author)

---

## 📌 Project Overview

The real estate market generates enormous volumes of housing data, yet accurate price estimation remains a challenge due to the number of influencing variables. This project addresses that challenge by building a supervised Machine Learning pipeline that trains regression models on a structured housing dataset and deploys the best-performing model as an interactive web application.

The project is divided into two distinct parts:

- **Part I** covers the complete ML workflow — data exploration, preprocessing, model training, evaluation, and selection.
- **Part II** covers the full-stack web application that integrates the trained model and exposes it through a clean, responsive user interface.

---

# 🤖 Part I — Machine Learning Model

## Problem Statement

The real estate market contains large volumes of housing data, making it difficult to estimate accurate property prices manually. House prices are influenced by multiple factors including area, number of bedrooms and bathrooms, parking availability, furnishing status, and access to amenities. Traditional estimation methods are often inconsistent, subjective, and time-consuming.

The objective of this project is to develop a **Machine Learning-based House Price Prediction system** that can analyze housing features and accurately predict property prices using regression techniques, while evaluating model performance through visualization and error metrics such as **MAE** and **RMSE**.

---

## Dataset

| Property         | Detail                                      |
|------------------|---------------------------------------------|
| Name             | House Price Prediction Dataset              |
| Source           | Kaggle                                      |
| File             | `Housing.csv`                               |
| Total Records    | 545 rows                                    |
| Total Features   | 13 columns (12 input features + 1 target)   |
| Target Variable  | `price` — house price in PKR                |

### Feature Overview

| Feature           | Type        | Description                          |
|-------------------|-------------|--------------------------------------|
| `price`           | int64       | Target — house price in PKR          |
| `area`            | int64       | Property area in square feet         |
| `bedrooms`        | int64       | Number of bedrooms                   |
| `bathrooms`       | int64       | Number of bathrooms                  |
| `stories`         | int64       | Number of floors                     |
| `mainroad`        | object      | Main road access (yes/no)            |
| `guestroom`       | object      | Guest room availability (yes/no)     |
| `basement`        | object      | Basement availability (yes/no)       |
| `hotwaterheating` | object      | Hot water heating (yes/no)           |
| `airconditioning` | object      | Air conditioning (yes/no)            |
| `parking`         | int64       | Number of parking spaces             |
| `prefarea`        | object      | Located in preferred area (yes/no)   |
| `furnishingstatus`| object      | Furnishing level (furnished/semi/unfurnished) |

### Statistical Summary

| Statistic | price       | area      | bedrooms | bathrooms | stories | parking |
|-----------|-------------|-----------|----------|-----------|---------|---------|
| count     | 545         | 545       | 545      | 545       | 545     | 545     |
| mean      | 4,766,729   | 5,150.54  | 2.97     | 1.29      | 1.81    | 0.69    |
| std       | 1,870,440   | 2,170.14  | 0.74     | 0.50      | 0.87    | 0.86    |
| min       | 1,750,000   | 1,650     | 1        | 1         | 1       | 0       |
| max       | 13,300,000  | 16,200    | 6        | 4         | 4       | 3       |

---

## Solution Approach

A **supervised Machine Learning** approach using regression models was implemented on the housing dataset. The pipeline followed these stages:

1. Load and explore the dataset
2. Encode categorical features into numerical representations
3. Scale numerical features using StandardScaler
4. Split data into training (80%) and testing (20%) sets
5. Train two regression models — Linear Regression and Gradient Boosting Regressor
6. Evaluate both models using MAE and RMSE
7. Visualize and compare results
8. Select the best-performing model for deployment

---

## Step-by-Step Training Pipeline

### Chapter 1 — Linear Regression

---

#### Step 1 — Importing Libraries

All necessary Python libraries were imported at the start of the notebook. `Pandas` and `NumPy` handle data manipulation and numerical operations. `Matplotlib` and `Seaborn` provide data visualization capabilities. `Scikit-learn` supplies the preprocessing tools, model classes, and evaluation metrics required throughout the pipeline.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
```

---

#### Step 2 — Loading & Exploring the Dataset

The housing dataset was loaded into a Pandas DataFrame. An initial exploratory analysis was then performed to understand the structure of the data — including the number of rows and columns, data types, missing value counts, and descriptive statistics. This step is critical before applying any preprocessing, as it reveals the nature and quality of the data.

```python
df = pd.read_csv('/content/drive/MyDrive/Housing.csv')
df.head()
```

```python
print("Dataset Shape:", df.shape)
print("\nColumns in Dataset:\n")
print(df.columns)
print("\nData Types:\n")
print(df.dtypes)
print("\nMissing Values:\n")
print(df.isnull().sum())
print("\nStatistical Summary:\n")
df.describe()
```

**Key Observations:**
- Dataset contains **545 records** and **13 columns**
- **No missing values** were found across any feature
- 7 features are categorical (`object` type) and require encoding
- 6 features are already numerical (`int64` type)

---

#### Step 3 — Categorical Feature Encoding

Machine learning algorithms cannot process raw textual data. All categorical columns were identified and encoded into numerical values using `LabelEncoder`. This transformation converts string labels into integer representations, enabling the regression models to learn from these features effectively.

```python
categorical_columns = df.select_dtypes(include=['object']).columns

print("Categorical Columns:")
print(categorical_columns)
```

```
Categorical Columns:
Index(['mainroad', 'guestroom', 'basement', 'hotwaterheating',
       'airconditioning', 'prefarea', 'furnishingstatus'], dtype='object')
```

```python
le = LabelEncoder()
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])
df.head()
```

---

#### Step 4 — Feature Selection

The dataset was split into input features (`X`) and the target variable (`y`). All 12 property-related features were selected as inputs, and `price` was designated as the prediction target. This separation is a fundamental requirement for supervised learning — the model learns the mapping from input features to the output price.

```python
X = df.drop('price', axis=1)
y = df['price']

print("Input Features:\n")
print(X.columns)
```

```
Input Features:
Index(['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom',
       'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea',
       'furnishingstatus'], dtype='object')
```

---

#### Step 5 — Feature Scaling

`StandardScaler` was applied to normalize the range of all input features. Since different features operate on different scales (e.g., `area` in thousands vs. `bedrooms` in single digits), scaling ensures that no single feature dominates the learning process due to its magnitude. This is particularly important for distance-sensitive and gradient-based algorithms.

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
X_scaled.head()
```

---

#### Step 6 — Train / Test Split

The scaled dataset was divided into training and testing subsets using an **80/20 split ratio**. The training set (436 samples) was used to fit the model, while the testing set (109 samples) was held out to evaluate performance on unseen data. A fixed `random_state=42` was used to ensure reproducibility.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("Training Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)
```

```
Training Features Shape: (436, 12)
Testing Features Shape:  (109, 12)
```

---

#### Step 7 — Training the Linear Regression Model

A `LinearRegression` model was instantiated and trained on the training dataset. Linear Regression is a foundational supervised learning algorithm that models the relationship between input features and the target variable by fitting the best-fit linear equation to the data. It serves as the baseline model in this comparison.

```python
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print("Linear Regression Model Trained Successfully!")
```

---

#### Step 8 — Generating Predictions

After training, the model was used to generate price predictions on the test set. The predicted values were compared against the actual prices to assess how well the model generalizes to unseen data.

```python
y_pred_lr = lr_model.predict(X_test)

predictions_df = pd.DataFrame({
    'Actual Price': y_test.values,
    'Predicted Price': y_pred_lr
})
predictions_df['Predicted Price'] = predictions_df['Predicted Price'].astype(int)
predictions_df.head(10)
```

---

#### Step 9 — Visualizing Actual vs. Predicted Prices

A line graph was plotted to visually compare the actual house prices against the Linear Regression predictions across the first 50 test samples. The blue line represents actual values and the orange line represents predicted values.

The predicted line generally follows the trend of the actual line, indicating that the model successfully captured the key relationships between housing features and prices. However, noticeable deviations exist at certain points, reflecting the model's limitations in capturing non-linear patterns.

```python
plt.figure(figsize=(10, 6))
plt.plot(y_test.values[:50], label='Actual Prices')
plt.plot(y_pred_lr[:50], label='Predicted Prices')
plt.title('Actual vs Predicted House Prices (Linear Regression)')
plt.xlabel('Test Data Index')
plt.ylabel('House Price')
plt.legend()
plt.show()
```

---

#### Step 10 — Model Evaluation (Linear Regression)

The Linear Regression model was evaluated using **Mean Absolute Error (MAE)** and **Root Mean Square Error (RMSE)**. MAE measures the average magnitude of prediction errors, while RMSE penalizes larger errors more heavily due to the squaring operation. Lower values of both metrics indicate better model performance.

```python
mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

print("Linear Regression MAE:", mae_lr)
print("Linear Regression RMSE:", rmse_lr)
```

```
Linear Regression MAE:  979,679.69
Linear Regression RMSE: 1,331,071.42
```

**Interpretation:** The model achieved reasonable accuracy for a baseline regression approach. The RMSE being notably higher than MAE suggests the presence of some larger prediction errors, which is expected in real estate datasets due to price variability and complex non-linear relationships between features.

---

### Chapter 2 — Gradient Boosting Regressor

---

#### Step 11 — Training the Gradient Boosting Model

A `GradientBoostingRegressor` was trained as an advanced ensemble model for performance comparison against Linear Regression. Gradient Boosting is an iterative ensemble technique that builds multiple weak decision trees sequentially, where each tree corrects the errors of the previous one. This approach allows the model to capture complex, non-linear relationships in the data that a simple linear model cannot.

```python
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)

print("Gradient Boosting Model Trained Successfully!")
```

---

#### Step 12 — Predictions & Evaluation (Gradient Boosting)

Predictions were generated using the trained Gradient Boosting model and evaluated using the same MAE and RMSE metrics for a fair comparison.

```python
y_pred_gb = gb_model.predict(X_test)

gb_predictions = pd.DataFrame({
    'Actual Price': y_test.values,
    'Predicted Price': y_pred_gb
})
gb_predictions.head()
```

```python
mae_gb = mean_absolute_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))

print("Gradient Boosting MAE:", mae_gb)
print("Gradient Boosting RMSE:", rmse_gb)
```

```
Gradient Boosting MAE:  964,058.87
Gradient Boosting RMSE: 1,301,871.87
```

---

### Chapter 3 — Model Comparison & Selection

---

#### Step 13 — Comparative Evaluation

After training both models, their performances were compared side-by-side using MAE and RMSE. A bar chart was plotted to visualize the RMSE difference between the two models.

```python
models = ['Linear Regression', 'Gradient Boosting']
rmse_values = [rmse_lr, rmse_gb]

plt.figure(figsize=(8, 5))
plt.bar(models, rmse_values)
plt.title('RMSE Comparison of Models')
plt.ylabel('RMSE Value')
plt.show()
```

#### Performance Summary

| Model                     | MAE            | RMSE           |
|---------------------------|----------------|----------------|
| Linear Regression         | 979,679.69     | 1,331,071.42   |
| **Gradient Boosting**     | **964,058.87** | **1,301,871.87** |

#### Selection Decision

The **Gradient Boosting Regressor** was selected as the final production model based on the following reasoning:

- It achieved a **lower MAE** (~15,000 PKR improvement), meaning its average prediction error is smaller
- It achieved a **lower RMSE** (~29,000 PKR improvement), meaning it handles larger prediction errors more effectively
- Its ensemble nature allows it to model non-linear feature interactions that Linear Regression cannot capture
- It generalizes better to unseen data, making it more suitable for real-world deployment

The trained Gradient Boosting model was serialized using `joblib` and saved as `Gradient Boosting houseprediction.pkl` for integration into the web application.

---

# 🌐 Part II — Web Application

## Technology Stack

| Layer          | Technology                          | Purpose                                              |
|----------------|-------------------------------------|------------------------------------------------------|
| Backend        | Python 3.x                          | Core programming language                            |
| Web Framework  | Flask                               | Routing, request handling, template rendering        |
| ML Integration | Scikit-learn, Joblib                | Model loading and inference                          |
| Data Handling  | Pandas, NumPy                       | Feature construction and preprocessing               |
| Frontend       | HTML5, CSS3                         | Page structure and custom styling                    |
| UI Framework   | Bootstrap 5 (CDN)                   | Responsive layout, components, and grid system       |
| Icons          | Bootstrap Icons (CDN)               | Professional iconography throughout the UI           |
| Templating     | Jinja2 (built into Flask)           | Dynamic HTML rendering with Python variables         |

---

## Application Architecture

The application follows a clean **MVC-inspired architecture**:

```
User Browser
     │
     ▼
Flask Routes (app.py)
     │
     ├── GET  /          → Renders index.html (prediction form)
     └── POST /predict   → Preprocesses input → Loads model → Returns result.html
                                │
                                ▼
                    model/Gradient Boosting houseprediction.pkl
```

- The **model is loaded once at application startup** using `joblib.load()`, avoiding repeated disk reads on every request
- All **preprocessing logic** (encoding + DataFrame construction) is isolated in a dedicated `preprocess()` function
- **Error handling** is implemented at both the client side (JavaScript validation) and server side (Flask try/except blocks)
- **Jinja2 templating** is used to pass the predicted price and property summary dictionary from the backend directly into the HTML result page

---

## Project Structure

```
house prediction web app/
│
├── app.py                                    # Flask backend — routes, preprocessing, prediction
├── retrain_model.py                          # Utility script to retrain model if needed
├── requirements.txt                          # Python package dependencies
├── README.md                                 # Project documentation
│
├── model/
│   └── Gradient Boosting houseprediction.pkl # Serialized trained model (joblib)
│
├── templates/
│   ├── index.html                            # Home page — hero section + prediction form
│   └── result.html                           # Result page — price display + property summary
│
└── static/
    └── style.css                             # Custom CSS — colors, cards, hero, animations
```

---

## Feature Preprocessing & Model Integration

The web application replicates the exact preprocessing pipeline used during model training to ensure prediction consistency.

### Encoding Rules Applied in `app.py`

| Feature           | Input (Form)                        | Encoded Value Sent to Model     |
|-------------------|-------------------------------------|---------------------------------|
| `mainroad`        | yes / no                            | 1 / 0                           |
| `guestroom`       | yes / no                            | 1 / 0                           |
| `basement`        | yes / no                            | 1 / 0                           |
| `hotwaterheating` | yes / no                            | 1 / 0                           |
| `airconditioning` | yes / no                            | 1 / 0                           |
| `prefarea`        | yes / no                            | 1 / 0                           |
| `furnishingstatus`| furnished / semi-furnished / unfurnished | 0 / 1 / 2               |
| `area`            | numeric (sq ft)                     | passed as-is (int)              |
| `bedrooms`        | numeric                             | passed as-is (int)              |
| `bathrooms`       | numeric                             | passed as-is (int)              |
| `stories`         | numeric                             | passed as-is (int)              |
| `parking`         | numeric                             | passed as-is (int)              |

The encoded values are assembled into a **Pandas DataFrame** with columns in the exact same order as the training data before being passed to `model.predict()`.

### Prediction Output

The raw numeric prediction from the model is formatted into a human-readable PKR currency string:

```
PKR 7,250,000
```

---

## Application Features

### Home Page (`index.html`)
- Gradient hero section with project title and description
- Info badges displaying model name, dataset, and prediction speed
- Structured Bootstrap 5 form divided into three sections:
  - **Property Specifications** — area, bedrooms, bathrooms, stories, parking
  - **Amenities & Features** — 6 yes/no dropdown fields with icons
  - **Furnishing** — furnishing status dropdown
- Client-side JavaScript validation that highlights empty fields before form submission
- Server-side error alert rendered via Jinja2 if preprocessing or prediction fails

### Result Page (`result.html`)
- Compact hero section confirming prediction completion
- Large price display card showing the estimated value in PKR format
- Bootstrap success alert confirming the model used
- Full property summary table listing all 12 submitted features
- Action buttons for starting a new prediction or printing the result
- Print-optimized CSS (`@media print`) that hides navigation and buttons

### Error Handling
- `KeyError` — caught when a required form field is missing
- `ValueError` — caught when a numeric field contains invalid input
- General `Exception` — catches any unexpected model or runtime errors
- All errors are rendered back on the home page as a dismissible Bootstrap danger alert

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### 1. Clone or Download the Project
```bash
git clone <repository-url>
cd "house prediction web app"
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### `requirements.txt`
```
flask
pandas
numpy
scikit-learn
joblib
```

### 4. Model Compatibility Note
If you encounter a NumPy version mismatch error when loading the `.pkl` file, run the included retraining script to regenerate a compatible model:

```bash
python retrain_model.py
```

This script retrains the Gradient Boosting model on `Housing.csv` using your current environment and overwrites the `.pkl` file automatically.

---

## Running the Application

```bash
python app.py
```

Open your browser and navigate to:

```
http://127.0.0.1:5000
```

---

## 🖼️ Screenshots

> *(Add screenshots here after running the application)*

| Page        | Description                                        |
|-------------|----------------------------------------------------|
| Home Page   | Hero section, info badges, and prediction form     |
| Result Page | Estimated price card, success alert, summary table |

---

## 🔮 Future Improvements

- [ ] Add model performance metrics dashboard (R², MAE, RMSE) visible in the UI
- [ ] Integrate multiple ML models with a side-by-side comparison view
- [ ] Add SHAP explainability charts to visualize feature importance per prediction
- [ ] Implement user authentication and prediction history logging
- [ ] Support CSV batch prediction upload for multiple properties at once
- [ ] Add interactive price range slider and map-based location pricing
- [ ] Deploy to AWS Elastic Beanstalk, Render, or Railway for public access
- [ ] Add REST API endpoints for third-party integration

---

## 👨‍💻 Author

**Haris Hussain**
Built as an internship-level, production-ready end-to-end Machine Learning web application project.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
