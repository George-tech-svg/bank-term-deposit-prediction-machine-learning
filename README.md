# Bank Term Deposit Prediction - Machine Learning

## End-to-End Machine Learning Project for Bank Marketing Campaign Optimization

---

## Project Overview

This project predicts whether a bank customer will subscribe to a term deposit. The model helps banks optimize their marketing campaigns by identifying customers most likely to subscribe.

**Business Problem:** A Portuguese banking institution wants to improve its direct marketing campaign efficiency by targeting customers who are most likely to subscribe to a term deposit.

**Solution:** A machine learning model that analyzes customer data and predicts subscription probability.

**Key Achievement:** Built a machine learning system that achieves **96.5% recall**, helping banks identify almost all potential subscribers while minimizing missed opportunities.

---

# Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Visualizations](#visualizations)
5. [Key Findings from Analysis](#key-findings-from-analysis)
6. [Models Tested](#models-tested)
7. [Final Model Performance](#final-model-performance)
8. [How to Run This Project](#how-to-run-this-project)
9. [How to Use the Flask API](#how-to-use-the-flask-api)
10. [API Endpoints Summary](#api-endpoints-summary)
11. [Files Explained](#files-explained)
12. [Technologies Used](#technologies-used)
13. [Results Summary](#results-summary)
14. [Future Improvements](#future-improvements)
15. [Author & Certifications](#author--certifications)

---

# Dataset

The dataset contains **45,211 customer records** with **17 features**.

---

## Column Descriptions

| Column | Type | Description |
|--------|------|-------------|
| age | Numeric | Customer age |
| job | Categorical | Job type (management, blue-collar, student, etc.) |
| marital | Categorical | Marital status (married, single, divorced) |
| education | Categorical | Education level (primary, secondary, tertiary) |
| default | Categorical | Has credit default? (yes, no) |
| balance | Numeric | Average yearly balance in euros |
| housing | Categorical | Has housing loan? (yes, no) |
| loan | Categorical | Has personal loan? (yes, no) |
| contact | Categorical | Contact communication type (cellular, telephone) |
| day | Numeric | Last contact day of month |
| month | Categorical | Last contact month of year |
| duration | Numeric | Last contact duration in seconds |
| campaign | Numeric | Number of contacts during this campaign |
| pdays | Numeric | Days since last contact from previous campaign |
| previous | Numeric | Number of contacts before this campaign |
| poutcome | Categorical | Outcome of previous campaign (success, failure, unknown) |
| y | Target | Customer subscribed? (yes, no) |

---

# Project Structure

```text
bank-term-deposit-prediction-machine-learning/
│
├── data/
│   └── bank-full.csv                     # Raw data file
│
├── scripts/
│   ├── 01_load_data.ipynb               # Load and inspect data
│   ├── 02_clean_data.ipynb              # Clean and prepare data
│   ├── 03_analyze_data.ipynb            # Exploratory data analysis
│   ├── 04_visualize_data.ipynb          # Create visualizations
│   ├── 05_build_features.ipynb          # Feature engineering
│   ├── 06_train_model.ipynb             # Train and compare models
│   └── 07_evaluate_model.ipynb          # Evaluate final model
│
├── deployment/
│   ├── best_model.pkl                   # Trained model
│   ├── preprocessor.pkl                 # Data preprocessor
│   ├── requirements.txt                 # API dependencies
│   └── app.py                           # Flask API
│
├── outputs/
│   ├── cleaned_data/                    # Processed datasets
│   ├── figures/                         # All visualizations
│   ├── models/                          # Backup models
│   └── reports/                         # Analysis reports
│
├── src/
│   └── config.py                        # Configuration settings
│
├── requirements.txt                     # Main dependencies
└── README.md                            # Project documentation
```

---

# Visualizations

## Target Variable Distribution

![Target Distribution](outputs/figure/target_distribution.png)

**Insight:** The dataset is imbalanced - only 11.7% of customers subscribed.

---

## Categorical Features vs Subscription Rate

![Categorical Features](outputs/figure/categorical_features_target.png)

### Key Insights

- March campaigns: 52% subscription rate
- Previous success: 64.7% subscription rate
- Students: 28.7% subscription rate
- May campaigns: Only 6.7% subscription rate

---

## Numerical Features Distribution

![Numerical Distributions](outputs/figure/numerical_distributions.png)

### Key Insights

- Age distribution is roughly normal
- Balance is heavily right-skewed
- Most customers receive 1-2 campaign calls

---

## Box Plots: Subscribers vs Non-Subscribers

![Box Plots Comparison](outputs/figure/boxplots_comparison.png)

### Key Insights

- Duration clearly separates subscribers from non-subscribers
- Subscribers have higher median balance
- Too many campaign calls reduces subscription likelihood

---

## Correlation Heatmap

![Correlation Heatmap](outputs/figure/correlation_heatmap.png)

### Key Insights

- Duration has strongest correlation with subscription (0.405)
- Campaign has negative correlation (-0.080)

---

## Model Comparison Chart

![Model Comparison](outputs/figure/model_comparison.png)

### Key Insights

- Random Forest (Tuned) achieved highest recall (96.5%)
- XGBoost showed overfitting

---

# Key Findings from Analysis

| Finding | Insight | Business Action |
|---------|---------|-----------------|
| Best month | March (52%) | Run major campaigns in March |
| Worst month | May (6.7%) | Avoid marketing in May |
| Best job type | Student (28.7%) | Target students |
| Best education | Tertiary (15.0%) | Target university educated |
| Housing loan | No loan = 16.7% | Focus on customers without housing loans |
| Previous outcome | Success = 64.7% | Re-target previous subscribers |

---

# Models Tested

| Model | Test Recall | Test Precision | Overfitting Gap | Verdict |
|-------|-------------|----------------|-----------------|---------|
| Logistic Regression | 12.0% | 11.7% | 0% | Poor |
| Random Forest (Default) | 96.5% | 23.9% | Large | Overfitting |
| Random Forest (Tuned) | 96.5% | 24.4% | 3.2% | BEST |
| XGBoost | 52.7% | 59.7% | 42% | Overfitting |

---

# Final Model Performance

## Selected Model: Random Forest (Tuned)

| Metric | Value | Simple Meaning |
|--------|-------|----------------|
| Recall | 96.5% | Finds 97 out of 100 interested customers |
| Precision | 24.4% | When calling 100 people, 24 actually subscribe |
| Accuracy | 64.7% | Overall correct predictions |

---

## Confusion Matrix Results

- True Negatives: 4,829 (Correctly said NO, did not call)
- False Positives: 3,156 (Wasted calls - said YES but customer said NO)
- False Negatives: 37 (Missed opportunities - said NO but customer would say YES)
- True Positives: 1,021 (Successful calls - said YES and customer said YES)

---

# How to Run This Project

## Prerequisites

- Python 3.8 or higher
- pip package manager

---

## Step 1: Navigate to Project Folder

```bash
cd bank-term-deposit-prediction-machine-learning
```

---

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 3: Run the Notebooks in Order

Open Jupyter Notebook or VS Code and run sequentially:

```text
scripts/01_load_data.ipynb
scripts/02_clean_data.ipynb
scripts/03_analyze_data.ipynb
scripts/04_visualize_data.ipynb
scripts/05_build_features.ipynb
scripts/06_train_model.ipynb
scripts/07_evaluate_model.ipynb
```

---

# How to Use the Flask API

## Step 1: Navigate to Deployment Folder

```bash
cd deployment
```

---

## Step 2: Install API Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 3: Run the API

```bash
python app.py
```

---

## Expected Output

```text
Model and preprocessor loaded successfully

Running on http://127.0.0.1:5000

Running on http://192.168.x.x:5000
```

---

# API Testing Examples

## Test 1: Health Check

Open your browser and go to:

```text
http://localhost:5000/health
```

### Expected Response

```json
{"model_loaded":true,"status":"healthy"}
```

---

## Test 2: Single Prediction Using Python

```python
import requests

url = "http://localhost:5000/predict"

customer = {
    "age": 35,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 2000,
    "housing": "no",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "mar",
    "duration": 300,
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
}

response = requests.post(url, json=customer)
print(response.json())
```

---

## Expected Response

```json
{
    "prediction": "SUBSCRIBE",
    "probability": 0.9181,
    "probability_percent": "91.8%",
    "recommendation": "CALL"
}
```

---

## Test 3: Batch Prediction

```python
import requests

url = "http://localhost:5000/predict_batch"

batch_data = {
    "customers": [
        {
            "customer_id": 1,
            "age": 35,
            "job": "management",
            "marital": "married",
            "education": "tertiary",
            "default": "no",
            "balance": 2000,
            "housing": "no",
            "loan": "no",
            "contact": "cellular",
            "day": 15,
            "month": "mar",
            "duration": 300,
            "campaign": 2,
            "pdays": -1,
            "previous": 0,
            "poutcome": "unknown"
        },
        {
            "customer_id": 2,
            "age": 45,
            "job": "blue-collar",
            "marital": "married",
            "education": "secondary",
            "default": "no",
            "balance": 500,
            "housing": "yes",
            "loan": "no",
            "contact": "cellular",
            "day": 20,
            "month": "may",
            "duration": 150,
            "campaign": 3,
            "pdays": -1,
            "previous": 0,
            "poutcome": "unknown"
        }
    ]
}

response = requests.post(url, json=batch_data)
print(response.json())
```

---

## Test 4: Using PowerShell (Windows)

```powershell
$body = '{"age":35,"job":"management","marital":"married","education":"tertiary","default":"no","balance":2000,"housing":"no","loan":"no","contact":"cellular","day":15,"month":"mar","duration":300,"campaign":2,"pdays":-1,"previous":0,"poutcome":"unknown"}'

Invoke-RestMethod -Uri "http://localhost:5000/predict" -Method POST -Body $body -ContentType "application/json"
```

---

# API Endpoints Summary

| Endpoint | Method | What it does |
|----------|--------|---------------|
| / | GET | Returns API information |
| /health | GET | Returns model status |
| /predict | POST | Predict for one customer |
| /predict_batch | POST | Predict for multiple customers |

---

# Files Explained

| File/Folder | Purpose |
|-------------|---------|
| data/bank-full.csv | Raw dataset |
| scripts/01_load_data.ipynb | Load and inspect data |
| scripts/02_clean_data.ipynb | Handle missing values, fix data types |
| scripts/03_analyze_data.ipynb | Statistical analysis |
| scripts/04_visualize_data.ipynb | Create all charts |
| scripts/05_build_features.ipynb | Feature engineering, preprocessing |
| scripts/06_train_model.ipynb | Train and compare 4 models |
| scripts/07_evaluate_model.ipynb | Test final model performance |
| deployment/best_model.pkl | Saved trained model |
| deployment/preprocessor.pkl | Saved data preprocessor |
| deployment/app.py | Flask API code |
| deployment/requirements.txt | API dependencies |
| outputs/figures/ | All visualization images |
| outputs/reports/ | Analysis and performance reports |
| src/config.py | Configuration settings |

---

# Technologies Used

| Category | Technologies |
|----------|--------------|
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Imbalance Handling | SMOTE (imbalanced-learn) |
| API Development | Flask, Gunicorn |
| Model Serialization | Joblib |
| Environment | Python 3.11 |

---

# Results Summary

## Best Month by Subscription Rate

| Month | Subscription Rate |
|-------|-------------------|
| March | 52.0% |
| September | 27.4% |
| October | 20.3% |
| May | 6.7% |

---

## Best Customer Segments by Subscription Rate

| Segment | Subscription Rate |
|---------|-------------------|
| Previous success | 64.7% |
| Student job | 28.7% |
| No housing loan | 16.7% |
| Single marital status | 15.0% |
| Tertiary education | 15.0% |

---

# Business Recommendations

## Priority 1: Campaign Timing Optimization

| Action | Expected Impact |
|--------|-----------------|
| Run major campaigns in March | Higher subscription rates |
| Reduce campaigns in May | Lower wasted marketing budget |
| Contact customers earlier in week | Better response rates |

---

## Priority 2: Customer Targeting

| Target Group | Reason |
|--------------|--------|
| Students | Highest subscription rate |
| Tertiary educated customers | Higher conversion |
| Previous successful subscribers | Highest probability to subscribe |

---

## Priority 3: Model Deployment

| Improvement | Benefit |
|-------------|---------|
| Real-time prediction API | Faster marketing decisions |
| Automated customer scoring | Efficient targeting |
| Dashboard integration | Better monitoring |

---

# Future Improvements

| Area | Potential Enhancement |
|------|----------------------|
| Data | Collect more recent campaign data |
| Features | Add economic indicators (interest rates, CPI) |
| Models | Test neural networks, ensemble methods |
| Deployment | Deploy to cloud (Render, AWS, GCP) |
| Monitoring | Add model performance monitoring |

---

# License

This project is for educational and portfolio purposes.

---

# Author & Certifications

## Author

**George Onyango Ochieng**

- ICT Graduate
- Data Scientist & Machine Learning Enthusiast
- Passionate about solving business problems using data

---

## Professional Certifications

- Data Science  
  https://savanna.alxafrica.com/certificates/flJSZ2Xs6r

- Machine Learning  
  https://savanna.alxafrica.com/certificates/7zsMrEN5m2

- Data Analytics  
  https://savanna.alxafrica.com/certificates/T95s3SPMxZ

- Python Programming  
  https://savanna.alxafrica.com/certificates/Ee8x6JfGCh

- Professional Foundations  
  https://savanna.alxafrica.com/certificates/RYz9rB28SJ

---

# Contact

- Email: georgebabji1220@gmail.com
- Phone: +254 115 136 359
- WhatsApp: https://wa.me/254111866769
- GitHub: https://github.com/George-techsvg
- LinkedIn: https://www.linkedin.com/in/george-onyango-5a5906360/

---

# Final Note

> "Built because I can't help it."

This project demonstrates:

- End-to-end machine learning workflow
- Business problem solving with AI
- Model evaluation and comparison
- API deployment using Flask
- Real-world marketing optimization
- Production-ready machine learning practices

Thank you for reviewing this project.

---

# Quick Commands Reference

```bash
# Clone repository
git clone https://github.com/George-techsvg/bank-term-deposit-prediction-machine-learning.git

# Navigate into project
cd bank-term-deposit-prediction-machine-learning

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook

# Run Flask API
cd deployment
python app.py
```
