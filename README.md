# 🛍️ Smart Customer Segmentation Using K-Means Clustering

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Unlock hidden patterns in customer behavior! This project provides a comprehensive solution for customer segmentation using K-Means clustering, featuring an interactive Streamlit dashboard and detailed analysis notebook. Perfect for data scientists, marketers, and business analysts looking to drive data-driven marketing strategies.

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Data Requirements](#-data-requirements)
- [Methodology](#-methodology)
- [Results & Insights](#-results--insights)
- [Technical Details](#-technical-details)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

### 🎯 **Interactive Dashboard**
- **Real-time Analysis**: Upload your data and get instant clustering results
- **Dynamic Visualization**: Interactive plots with Plotly for better insights
- **Feature Selection**: Choose which customer attributes to include in clustering
- **Optimal K Detection**: Automatic detection of optimal number of clusters using Elbow Method and Silhouette Analysis
- **Business Insights**: AI-powered recommendations for each customer segment

### 📊 **Advanced Analytics**
- **Multiple Validation Methods**: Elbow Method, Silhouette Score, and KneeLocator
- **Comprehensive Profiling**: Detailed analysis of each customer segment
- **Statistical Validation**: Model performance metrics and validation
- **Export Capabilities**: Download results in CSV format for further analysis

### 🎨 **Visualization Suite**
- **Cluster Scatter Plots**: 2D/3D visualization of customer segments
- **Distribution Charts**: Age, income, and spending patterns by cluster
- **Silhouette Analysis**: Detailed cluster quality assessment
- **Interactive Dashboards**: Real-time exploration of customer data

## 📁 Project Structure

```
Smart-Customer-Segmentation-Using-K-Means-Clustering/
├── 📁 app/
│   └── application.py              # Streamlit dashboard application
├── 📁 data/
│   ├── Mall_Customers.csv         # Original customer dataset
│   └── customer_segments.csv      # Processed data with cluster labels
├── 📁 notebook/
│   └── customersegmentationusingkmean.ipynb  # Jupyter notebook analysis
├── 📄 README.md                   # Project documentation
└── 🖼️ SystemUI.png               # Dashboard screenshot
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/ArunPandeyLaudari/Smart-Customer-Segmentation-Using-K-Means-Clustering.git
cd Smart-Customer-Segmentation-Using-K-Means-Clustering
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install streamlit pandas numpy matplotlib seaborn plotly scikit-learn kneed openpyxl
```

### Step 3: Run the Application
```bash
streamlit run app/application.py
```

## 🎯 Quick Start

1. **Launch the Dashboard**: Run `streamlit run app/application.py`
2. **Upload Data**: Use the sidebar to upload your customer data (CSV format)
3. **Configure Settings**: Select features for clustering and adjust parameters
4. **Analyze Results**: Review optimal cluster recommendations and visualizations
5. **Export Insights**: Download segmented data and business recommendations

## 📖 Usage

### Interactive Dashboard

The Streamlit dashboard provides an intuitive interface for customer segmentation:

1. **Data Upload**: Upload your customer data in CSV format
2. **Feature Selection**: Choose relevant features (Income, Spending Score, Age)
3. **Cluster Configuration**: Adjust clustering parameters
4. **Visualization**: Explore interactive plots and insights
5. **Export Results**: Download segmented data and summaries

### Jupyter Notebook Analysis

For detailed analysis and customization:

1. Open `notebook/customersegmentationusingkmean.ipynb`
2. Run cells sequentially to understand the methodology
3. Modify parameters and features as needed
4. Export results for further analysis

## 📊 Data Requirements

### Expected Data Format
Your customer data should include:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| CustomerID | Integer | Unique customer identifier | 1, 2, 3... |
| Gender | String | Customer gender | Male, Female |
| Age | Integer | Customer age in years | 19, 25, 35... |
| Annual Income (k$) | Integer | Annual income in thousands | 15, 25, 50... |
| Spending Score (1-100) | Integer | Spending behavior score | 39, 81, 6... |

### Supported File Formats
- CSV files (.csv)
- Excel files (.xlsx, .xls)
- Multiple Excel sheets supported

## 🔬 Methodology

### 1. Data Preprocessing
- **Data Cleaning**: Handle missing values and duplicates
- **Feature Selection**: Choose relevant numerical features
- **Scaling**: Optional standardization using StandardScaler

### 2. Optimal Cluster Detection
- **Elbow Method**: Find the "elbow" point in WCSS curve
- **Silhouette Analysis**: Measure cluster quality and separation
- **KneeLocator**: Automated elbow detection algorithm

### 3. K-Means Clustering
- **Algorithm**: K-Means++ initialization for better convergence
- **Validation**: Multiple random initializations (n_init=10)
- **Optimization**: Minimize within-cluster sum of squares (WCSS)

### 4. Model Validation
- **Silhouette Score**: Measure cluster cohesion and separation
- **Visual Inspection**: Analyze cluster plots and distributions
- **Business Logic**: Validate segments make business sense

## 📈 Results & Insights

### Customer Segments Identified

Based on the analysis of 200 customers, we identified **5 distinct customer segments**:

| Cluster | Size | Profile | Strategy |
|---------|------|---------|----------|
| **Cluster 0** | 81 customers | Balanced Customers | Standard marketing approach |
| **Cluster 1** | 39 customers | Premium Customers | Focus on luxury items and personalized service |
| **Cluster 2** | 22 customers | Enthusiastic Spenders | Great for promotions and new product launches |
| **Cluster 3** | 35 customers | High Earners, Low Spenders | Target with premium products and exclusive offers |
| **Cluster 4** | 23 customers | Budget-Conscious Customers | Focus on value propositions and discounts |

### Key Metrics
- **Silhouette Score**: 0.554 (Good cluster separation)
- **Optimal Clusters**: 5 (confirmed by both Elbow and Silhouette methods)
- **Data Quality**: No missing values, 200 clean records

## 🛠️ Technical Details

### Dependencies
```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0
scikit-learn>=1.3.0
kneed>=0.7.0
openpyxl>=3.0.0
```

### Performance
- **Processing Time**: < 5 seconds for 200 customers
- **Memory Usage**: < 100MB for typical datasets
- **Scalability**: Tested up to 10,000 customers

### Browser Compatibility
- Chrome (recommended)
- Firefox
- Safari
- Edge

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup
```bash
git clone https://github.com/ArunPandeyLaudari/Smart-Customer-Segmentation-Using-K-Means-Clustering.git
cd Smart-Customer-Segmentation-Using-K-Means-Clustering
pip install -r requirements.txt
streamlit run app/application.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/ArunPandeyLaudari/Smart-Customer-Segmentation-Using-K-Means-Clustering/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ArunPandeyLaudari/Smart-Customer-Segmentation-Using-K-Means-Clustering/discussions)

## 🙏 Acknowledgments

- Dataset: [Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- Libraries: Streamlit, Scikit-learn, Plotly, Pandas
- Inspiration: Customer segmentation best practices in retail analytics

---

**Built with ❤️ for the data science community**

*Unlock the power of customer segmentation and drive data-driven business decisions!*
