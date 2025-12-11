# 📦 Flipkart Customer Service Satisfaction – Machine Learning Project

**Project Type:** Classification
**Contribution:** Individual
**Author:** *Kush Agra Soni*

---

## 📘 Project Overview

In the highly competitive e-commerce space, customer satisfaction is one of the strongest drivers of brand loyalty and retention. Flipkart handles millions of support interactions through phone, chat, email, and other channels. With such scale, it becomes challenging to maintain consistent customer experience and agent performance.

This project leverages **Machine Learning**, **statistical analysis**, and **NLP-based sentiment extraction** to:

✔ Understand factors driving customer satisfaction
✔ Identify poorly performing agents and support channels
✔ Predict CSAT (Customer Satisfaction Score)
✔ Provide data-backed recommendations for improving service quality

This project ultimately enables Flipkart to identify dissatisfaction early, improve agent training, and deploy better customer-handling strategies.

---

# 📊 Dataset Description

The dataset contains **85,907 customer support interaction records** with **20 variables**, including:

### **Key Columns**

* `channel_name` – Phone/Email/Inbound/Outbound
* `category`, `Sub-category` – Issue types
* `Customer Remarks` – Customer feedback text
* `order_date_time`, `Issue_reported at` – Timestamps
* `Product_category`, `Item_price`
* `Agent_name`, `Supervisor`, `Manager`
* `Tenure Bucket`, `Agent Shift`
* `CSAT Score` (Target variable: 1 to 5)

### **Missing Data**

Several features had high missing values, such as:

* `connected_handling_time` → ~99% missing
* `order_date_time`, `Customer_City`, `Item_price`

These were handled through **dropping**, **aggregation**, or **encoding**.

---

# 🧹 1. Data Wrangling

### ✔ Cleaning Steps

* Removed redundant ID-based columns
* Parsed and cleaned datetime fields
* Extracted sentiment from *Customer Remarks* using **VADER**
* Handled missing values with safe dropping
* Converted tenure buckets to ordinal values
* Created new hierarchical performance features:

  * `agent_avg_csat`
  * `supervisor_avg_csat`
  * `manager_avg_csat`

---

# 📐 2. Statistical Hypothesis Testing

Three statistical analyses were conducted:

### **H₁: Bottom-10 agents have lower CSAT than others**

**Test Used:** Mann–Whitney U Test
✔ Result: **Significant (p < 0.001)** → Bottom agents perform poorly.

### **H₂: CSAT differs across support channels**

**Test Used:** Kruskal–Wallis Test
✔ Result: **Significant difference across channels**

### **H₃: Agent tenure is associated with CSAT**

**Test Used:** Spearman Correlation
✔ Result: Positive monotonic relationship (ρ = 0.04, p < 0.001)
→ More experienced agents perform slightly better.

---

# 💡 3. Feature Engineering

### ✔ Encoding Techniques Used

| Technique                             | Columns                             | Why Used?                                       |
| ------------------------------------- | ----------------------------------- | ----------------------------------------------- |
| **One-Hot Encoding**                  | channel_name, category, Agent Shift | Low cardinality, simple meaningful splits       |
| **Target Encoding**                   | Sub-category, Product_category      | Reduces dimensionality & captures CSAT patterns |
| **Ordinal Encoding**                  | Tenure Bucket                       | Data is ordered & numeric                       |
| **Drop High-Cardinality Identifiers** | Agent/Supervisor/Manager            | Prevent overfitting; replaced with avg CSAT     |

### ✔ New Features Created

* Sentiment score (`customer_response`)
* Historical satisfaction averages for:

  * Agents
  * Supervisors
  * Managers
* Aggregated category-CSAT features
* Date/time components (day/month)

---

# 🔍 4. Feature Selection

### **Methods Used**

✔ **Domain knowledge filtering**
✔ **Correlation Analysis**
✔ **Aggregation to reduce cardinality**

### **Top 10 Most Important Features**

1. customer_response (sentiment)
2. agent_avg_csat
3. Sub-category_mean_csat
4. Product_category_mean_csat
5. Item_price
6. supervisor_avg_csat
7. category_Returns
8. category_Order Related
9. manager_avg_csat
10. tenure_numeric

These were selected based on correlation, interpretability, and business importance.

---

# 🔄 5. Data Transformation & Scaling

### ✔ Transformation Method

**Yeo–Johnson Power Transformation**

* Handles negative & positive values
* Reduces both left & right skewness
* Suitable for linear & tree models

### ✔ Scaling Method

**StandardScaler**

* Standardizes features to mean=0, std=1
* Essential for models like Logistic Regression, XGBoost, NN

---

# 🤖 6. Machine Learning Models

Two classification models were developed:

## **🟦 Model 1: Random Forest Classifier**

* Handles non-linear interactions
* class_weight="balanced"
* Good interpretability

**Accuracy:** ~54%
**Macro F1:** ~0.25
**Strength:** Better handling of minority classes than XGBoost

---

## **🟧 Model 2: XGBoost Classifier**

* Tuned with 500 trees, depth 8
* Best raw accuracy (~71%)
* But **very poor recall** for minority classes (CSAT 2 & 3)

---

# 🏆 Final Model Chosen

## ⭐ **Random Forest Classifier (with hyperparameter optimization)**

Chosen because:
✔ Better overall balance across all CSAT classes
✔ Higher recall for low-CSAT scores (1–3)
✔ Critical for identifying dissatisfied customers
✔ More interpretable for business stakeholders

---

# 🧠 7. Explainability – SHAP Analysis

SHAP values were used to interpret the model:

### **Most Influential Features According to SHAP**

1. Sentiment score (customer_response)
2. Agent’s historical CSAT
3. Sub-category_mean_csat
4. Product_category_mean_csat
5. Item_price
6. Supervisor/manager influence
7. Tenure

📌 *Sentiment from customer remarks is the strongest predictor of satisfaction.*

---

# 🛠️ Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Seaborn, Matplotlib
* NLTK VADER Sentiment
* SHAP Explainability
* 
