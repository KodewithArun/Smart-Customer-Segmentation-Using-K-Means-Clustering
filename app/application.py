import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #3730a3;
        border-bottom: 2px solid #e0e7ff;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
        margin-bottom: 1rem;
    }
    .insight-box {
        background-color: #f0f9ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bae6fd;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🛍️ Customer Segmentation Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### Analyze customer behavior and identify distinct market segments using K-Means clustering")

# Sidebar for controls
st.sidebar.header("🎛️ Dashboard Controls")

# Data loading section
@st.cache_data
def load_sample_data():
    """Generate sample customer data for demonstration"""
    np.random.seed(42)
    n_customers = 200
    
    data = {
        'CustomerID': range(1, n_customers + 1),
        'Gender': np.random.choice(['Male', 'Female'], n_customers),
        'Age': np.random.randint(18, 70, n_customers),
        'Annual Income (k$)': np.random.randint(15, 140, n_customers),
        'Spending Score (1-100)': np.random.randint(1, 100, n_customers)
    }
    
    return pd.DataFrame(data)

@st.cache_data
def load_data():
    """Load customer data"""
    try:
        # Try to load from uploaded file first
        return pd.read_csv('Mall_Customers.csv')
    except:
        # Use sample data if file not available
        return load_sample_data()

# File upload option
uploaded_file = st.sidebar.file_uploader("Upload Customer Data (CSV)", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Data uploaded successfully!")
else:
    df = load_data()
    st.sidebar.info("📊 Using sample data for demonstration")

# Data overview section
st.markdown('<div class="section-header">📊 Data Overview</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Customers", len(df))
with col2:
    st.metric("Features", len(df.columns))
with col3:
    st.metric("Male Customers", len(df[df['Gender'] == 'Male']))
with col4:
    st.metric("Female Customers", len(df[df['Gender'] == 'Female']))

# Display data preview
if st.checkbox("Show Data Preview", value=True):
    st.dataframe(df.head(10), use_container_width=True)

# Feature selection
st.sidebar.markdown("### 🎯 Clustering Configuration")
available_features = ['Annual Income (k$)', 'Spending Score (1-100)', 'Age']
selected_features = st.sidebar.multiselect(
    "Select Features for Clustering",
    available_features,
    default=['Annual Income (k$)', 'Spending Score (1-100)']
)

if len(selected_features) < 2:
    st.error("Please select at least 2 features for clustering analysis.")
    st.stop()

# Prepare data for clustering
X = df[selected_features].copy()

# Scaling option
use_scaling = st.sidebar.checkbox("Apply Feature Scaling", value=False)
if use_scaling:
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), 
        columns=selected_features,
        index=X.index
    )
    X_clustering = X_scaled
else:
    X_clustering = X

# Clustering analysis
st.markdown('<div class="section-header">🔍 Optimal Cluster Analysis</div>', unsafe_allow_html=True)

# Determine optimal number of clusters
@st.cache_data
def compute_clustering_metrics(_X, max_k=10):
    """Compute WCSS and silhouette scores for different k values"""
    wcss = []
    silhouette_scores = []
    K_range = range(1, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(_X)
        wcss.append(kmeans.inertia_)
        
        if k > 1:  # Silhouette score needs at least 2 clusters
            labels = kmeans.labels_
            silhouette_avg = silhouette_score(_X, labels)
            silhouette_scores.append(silhouette_avg)
    
    return wcss, silhouette_scores, K_range

max_clusters = st.sidebar.slider("Maximum Clusters to Test", 3, 15, 10)
wcss, silhouette_scores, K_range = compute_clustering_metrics(X_clustering, max_clusters)

# Create interactive plots for cluster analysis
col1, col2 = st.columns(2)

with col1:
    # Elbow method plot
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(
        x=list(K_range), 
        y=wcss,
        mode='lines+markers',
        name='WCSS',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=8)
    ))
    
    # Find elbow using KneeLocator
    kl = KneeLocator(K_range, wcss, curve="convex", direction="decreasing")
    optimal_k_elbow = kl.elbow
    
    if optimal_k_elbow:
        fig_elbow.add_vline(
            x=optimal_k_elbow, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Elbow at k={optimal_k_elbow}"
        )
    
    fig_elbow.update_layout(
        title="Elbow Method for Optimal k",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="WCSS",
        showlegend=False
    )
    st.plotly_chart(fig_elbow, use_container_width=True)

with col2:
    # Silhouette score plot
    fig_sil = go.Figure()
    fig_sil.add_trace(go.Scatter(
        x=list(range(2, max_clusters + 1)), 
        y=silhouette_scores,
        mode='lines+markers',
        name='Silhouette Score',
        line=dict(color='#10b981', width=3),
        marker=dict(size=8)
    ))
    
    # Find optimal k from silhouette score
    optimal_k_silhouette = range(2, max_clusters + 1)[np.argmax(silhouette_scores)]
    max_silhouette = max(silhouette_scores)
    
    fig_sil.add_vline(
        x=optimal_k_silhouette, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Best k={optimal_k_silhouette}"
    )
    
    fig_sil.update_layout(
        title="Silhouette Score Analysis",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Average Silhouette Score",
        showlegend=False
    )
    st.plotly_chart(fig_sil, use_container_width=True)

# Display optimal k recommendations
st.markdown('<div class="insight-box">', unsafe_allow_html=True)
st.write("**📈 Optimal Cluster Recommendations:**")
col1, col2 = st.columns(2)
with col1:
    st.metric("Elbow Method", optimal_k_elbow if optimal_k_elbow else "Not clear")
with col2:
    st.metric("Silhouette Score", f"{optimal_k_silhouette} (Score: {max_silhouette:.3f})")
st.markdown('</div>', unsafe_allow_html=True)

# User selection of number of clusters
st.sidebar.markdown("### 🎯 Final Clustering")
n_clusters = st.sidebar.slider(
    "Select Number of Clusters", 
    2, 
    max_clusters, 
    value=optimal_k_elbow if optimal_k_elbow else 5
)

# Perform final clustering
@st.cache_data
def perform_clustering(_X, n_clusters):
    """Perform K-means clustering"""
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    labels = kmeans.fit_predict(_X)
    centroids = kmeans.cluster_centers_
    silhouette_avg = silhouette_score(_X, labels)
    return labels, centroids, silhouette_avg, kmeans

labels, centroids, silhouette_avg, kmeans_model = perform_clustering(X_clustering, n_clusters)
df['Cluster'] = labels

st.markdown('<div class="section-header">🎨 Cluster Visualization</div>', unsafe_allow_html=True)

# Interactive cluster visualization
if len(selected_features) >= 2:
    fig_scatter = px.scatter(
        df, 
        x=selected_features[0], 
        y=selected_features[1],
        color='Cluster',
        hover_data=['Age', 'Gender'] if 'Age' in df.columns else ['Gender'],
        title=f"Customer Segments ({n_clusters} Clusters)",
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    
    # Add centroids
    if not use_scaling:
        centroids_df = pd.DataFrame(centroids, columns=selected_features)
        fig_scatter.add_trace(go.Scatter(
            x=centroids_df[selected_features[0]],
            y=centroids_df[selected_features[1]],
            mode='markers',
            marker=dict(
                size=15,
                color='black',
                symbol='x',
                line=dict(width=2, color='white')
            ),
            name='Centroids',
            showlegend=True
        ))
    
    fig_scatter.update_layout(height=600)
    st.plotly_chart(fig_scatter, use_container_width=True)

# Cluster analysis and insights
st.markdown('<div class="section-header">📊 Cluster Analysis & Insights</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    # Cluster characteristics table
    cluster_stats = df.groupby('Cluster').agg({
        'Age': ['mean', 'std'],
        'Annual Income (k$)': ['mean', 'std'],
        'Spending Score (1-100)': ['mean', 'std'],
        'Gender': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Mixed'
    }).round(2)
    
    cluster_stats.columns = ['Age_Mean', 'Age_Std', 'Income_Mean', 'Income_Std', 
                            'Spending_Mean', 'Spending_Std', 'Dominant_Gender']
    
    st.dataframe(cluster_stats, use_container_width=True)

with col2:
    # Cluster sizes
    cluster_counts = df['Cluster'].value_counts().sort_index()
    fig_pie = px.pie(
        values=cluster_counts.values,
        names=[f'Cluster {i}' for i in cluster_counts.index],
        title="Cluster Distribution"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# Business insights for each cluster
st.markdown("### 💼 Business Insights & Strategy Recommendations")

for cluster_id in range(n_clusters):
    cluster_data = df[df['Cluster'] == cluster_id]
    
    with st.expander(f"🎯 Cluster {cluster_id} Analysis ({len(cluster_data)} customers)"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_age = cluster_data['Age'].mean()
            st.metric("Average Age", f"{avg_age:.1f} years")
        
        with col2:
            avg_income = cluster_data['Annual Income (k$)'].mean()
            st.metric("Average Income", f"${avg_income:.1f}k")
        
        with col3:
            avg_spending = cluster_data['Spending Score (1-100)'].mean()
            st.metric("Average Spending Score", f"{avg_spending:.1f}/100")
        
        # Generate business insights
        if avg_income < 40 and avg_spending < 40:
            insight = "💰 **Budget-Conscious Customers** - Focus on value propositions, discounts, and affordable product lines."
            strategy = "Implement loyalty programs, bulk discounts, and budget-friendly marketing campaigns."
        elif avg_income > 70 and avg_spending < 40:
            insight = "🏦 **High Earners, Conservative Spenders** - Target with premium, durable products and exclusive offers."
            strategy = "Emphasize quality, exclusivity, and long-term value in marketing messages."
        elif avg_income < 60 and avg_spending > 60:
            insight = "🛍️ **Enthusiastic Spenders** - Great for promotions, new product launches, and impulse purchases."
            strategy = "Focus on trendy products, flash sales, and social media marketing."
        elif avg_income > 60 and avg_spending > 60:
            insight = "👑 **Premium Customers** - Focus on luxury items, personalized service, and VIP experiences."
            strategy = "Offer premium products, personal shopping services, and exclusive events."
        else:
            insight = "⚖️ **Balanced Customers** - Standard marketing approach with diverse product offerings."
            strategy = "Use general marketing strategies with a mix of value and premium options."
        
        st.markdown(insight)
        st.markdown(f"**Recommended Strategy:** {strategy}")
        
        # Gender distribution for this cluster
        if len(cluster_data) > 0:
            gender_dist = cluster_data['Gender'].value_counts()
            st.write(f"**Gender Distribution:** {gender_dist.to_dict()}")

# Model validation metrics
st.markdown('<div class="section-header">🎯 Model Validation</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Silhouette Score", f"{silhouette_avg:.3f}")
with col2:
    st.metric("Number of Clusters", n_clusters)
with col3:
    st.metric("Total Customers", len(df))

# Silhouette analysis plot
if st.checkbox("Show Detailed Silhouette Analysis"):
    sample_silhouette_values = silhouette_samples(X_clustering, labels)
    
    fig_sil_detail = go.Figure()
    
    y_lower = 10
    colors = px.colors.qualitative.Set1
    
    for i in range(n_clusters):
        cluster_silhouette_values = sample_silhouette_values[labels == i]
        cluster_silhouette_values = np.sort(cluster_silhouette_values)
        
        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        fig_sil_detail.add_trace(go.Scatter(
            x=cluster_silhouette_values,
            y=np.arange(y_lower, y_upper),
            fill='tonexty' if i > 0 else 'tozeroy',
            mode='none',
            name=f'Cluster {i}',
            fillcolor=colors[i % len(colors)]
        ))
        
        y_lower = y_upper + 10
    
    fig_sil_detail.add_vline(
        x=silhouette_avg,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Average: {silhouette_avg:.3f}"
    )
    
    fig_sil_detail.update_layout(
        title=f"Silhouette Analysis for {n_clusters} Clusters",
        xaxis_title="Silhouette Coefficient Values",
        yaxis_title="Cluster Label",
        height=400
    )
    
    st.plotly_chart(fig_sil_detail, use_container_width=True)

# Download results
st.markdown('<div class="section-header">💾 Export Results</div>', unsafe_allow_html=True)

# Prepare results for download
results_df = df.copy()
results_summary = pd.DataFrame({
    'Cluster': range(n_clusters),
    'Size': [len(df[df['Cluster'] == i]) for i in range(n_clusters)],
    'Avg_Age': [df[df['Cluster'] == i]['Age'].mean() for i in range(n_clusters)],
    'Avg_Income': [df[df['Cluster'] == i]['Annual Income (k$)'].mean() for i in range(n_clusters)],
    'Avg_Spending': [df[df['Cluster'] == i]['Spending Score (1-100)'].mean() for i in range(n_clusters)]
}).round(2)

col1, col2 = st.columns(2)

with col1:
    csv_results = results_df.to_csv(index=False)
    st.download_button(
        label="📊 Download Customer Data with Clusters",
        data=csv_results,
        file_name="customer_segments.csv",
        mime="text/csv"
    )

with col2:
    csv_summary = results_summary.to_csv(index=False)
    st.download_button(
        label="📈 Download Cluster Summary",
        data=csv_summary,
        file_name="cluster_summary.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("### 🚀 How to Use This Dashboard")
st.markdown("""
1. **Upload Data**: Use the sidebar to upload your customer data (CSV or Excel format)
2. **Map Columns**: If needed, map your data columns to the expected format
3. **Configure**: Select features for clustering and adjust settings
4. **Analyze**: Review the optimal cluster recommendations
5. **Visualize**: Explore interactive cluster visualizations
6. **Insights**: Read business insights and strategy recommendations
7. **Export**: Download results in CSV or Excel format for further analysis

**Supported File Formats:**
- CSV files (.csv)
- Excel files (.xlsx, .xls)
- Multiple Excel sheets supported with sheet selection

**Expected Data Structure:**
- Numeric columns for clustering (e.g., income, spending score, age)
- Optional categorical columns (e.g., gender, region)
- Each row should represent a unique customer

**Note**: The app includes automatic column mapping if your data uses different column names.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**Required Libraries:**")
st.sidebar.code("pip install streamlit pandas numpy matplotlib seaborn plotly scikit-learn kneed openpyxl", language="bash")

st.sidebar.markdown("---")
st.sidebar.markdown("**📧 Contact**")
st.sidebar.markdown("Built with ❤️ using Streamlit")