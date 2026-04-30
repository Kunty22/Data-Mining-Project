# Customer Segmentation using K-Means Clustering

## 📖 Overview

This project was developed by me as part of a university Data Mining course. It demonstrates a complete unsupervised machine learning workflow applied to mall customer data, with the goal of identifying distinct customer segments based on behavioural features.

The objective was to go beyond simply running a clustering algorithm by understanding the full pipeline: how data is explored, how the optimal number of clusters is determined, and how results are interpreted in a business context.

---

## ⚙️ Project Workflow

The project is structured into multiple stages:

1. **Exploratory Data Analysis (EDA)**
   - Understanding feature distributions (Age, Income, Spending Score)
   - Analysing correlations between variables
   - Identifying data characteristics before modelling

2. **Data Preprocessing**
   - Feature selection (removing non-behavioural variables)
   - Feature scaling using StandardScaler
   - Ensuring distance-based algorithms are not dominated by high-range features

3. **Determining Optimal Clusters**
   - Elbow Method (inertia across k = 1–10)
   - Silhouette Score analysis (k = 2–10)
   - Both methods converged on **k = 5**

4. **Clustering**
   - K-Means applied with k = 5
   - Each customer assigned a cluster label (0–4)
   - Results saved to output CSV

5. **Dimensionality Reduction & Visualisation**
   - PCA applied to reduce 3D feature space to 2 principal components
   - Cluster separation visualised in 2D scatter plot

6. **Interpretation**
   - Each cluster profiled by Age, Income, and Spending Score
   - Business insights derived for targeted marketing strategies

---

## 🧠 Algorithm Used

- **K-Means Clustering** (unsupervised learning)
- Validation: Elbow Method + Silhouette Score
- Visualisation: PCA (2 components)

---

## 📊 Results

### 🏆 Final Model

- **Algorithm:** K-Means
- **Optimal k:** 5
- **Silhouette Score:** 0.4166
- **Final Inertia:** 168.25

### 📈 Cluster Validation

| k | Silhouette Score | Inertia  |
|---|-----------------|----------|
| 2 | 0.3355          | 389.39   |
| 3 | 0.3578          | 295.21   |
| 4 | 0.4040          | 205.23   |
| **5** | **0.4166**  | **168.25** |
| 6 | 0.4284          | 133.87   |
| 7 | 0.4172          | 117.01   |

> k = 5 was selected as the optimal value — both the Elbow Method and Silhouette Score plateau beyond this point, and the marginal gain at k = 6 does not justify the added complexity.

---

## 🔍 Customer Segments

| Cluster | Profile | Description |
|---------|---------|-------------|
| 0 | 💎 Premium | High income – High spending |
| 1 | 🧊 Conservative | High income – Low spending |
| 2 | 🔥 Enthusiastic | Low income – High spending |
| 3 | 💰 Budget | Low income – Low spending |
| 4 | ⚖️ Average | Moderate income – Moderate spending |

### Key Insights

- **Spending Score** and **Annual Income** were the most discriminating features for cluster separation
- Clusters appear well-separated in PCA projection, confirming effective segmentation
- Age, Income, and Spending Score show weak inter-correlations — no multicollinearity, all features retained
- Segmentation results can directly inform targeted marketing, personalised offers, and resource allocation

---

## 📁 Project Structure

```
DataMining-541036/
│
├── customer_segmentation_clustering.py   # Main clustering pipeline
│
├── Mall_Customers.csv                    # Original dataset (200 customers)
├── Mall_Customers.csv.xls                # Excel version of dataset
├── mall_customers_with_clusters.csv      # Dataset with assigned cluster labels
│
└── Presentation and Report/
    ├── Report-541036.pdf                 # Full written project report
    └── Report-541036.docx                # Editable report
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib scikit-learn
```

### Run the Project

```bash
git clone https://github.com/<your-username>/DataMining-541036.git
cd DataMining-541036
python customer_segmentation_clustering.py
```

### Dataset

The Mall Customers dataset contains **200 customers** with 5 variables:

| Column | Description |
|--------|-------------|
| CustomerID | Unique identifier (excluded from clustering) |
| Genre | Gender (excluded to focus on behavioural features) |
| Age | Customer age |
| Annual Income (k$) | Annual income in thousands |
| Spending Score (1–100) | Mall-assigned spending behaviour score |

---

## 👥 Author

| Name |
|------|
| Kunt James Washima |

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Libraries:** scikit-learn, pandas, numpy, matplotlib
- **IDE:** PyCharm

---

## 📜 License

This project was developed for academic purposes. Please contact the author before reusing or redistributing any part of this work.
