"""
SHOPPER SPECTRUM: Customer Segmentation and Product Recommendations
Clean Working Version for VS Code
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

# Configuration
st.set_page_config(page_title="Shopper Spectrum", page_icon="🛒", layout="wide")

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #2E86AB;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🛒 Shopper Spectrum</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem;">Customer Segmentation & Product Recommendation System</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("📊 Navigation")
    app_mode = st.selectbox(
        "Choose Module",
        ["Dashboard", "Product Recommendations", "Customer Segmentation", "About"]
    )
    
    st.markdown("---")
    st.info("""
    **Features:**
    - Customer segmentation using RFM analysis
    - Product recommendations
    - Business insights
    """)

# Generate sample data
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    
    # Generate sample transactions
    n_customers = 500
    n_products = 50
    n_transactions = 2000
    
    customer_ids = [f'C{i:05d}' for i in range(1, n_customers + 1)]
    products = [f'Product {i}' for i in range(1, n_products + 1)]
    
    data = []
    for i in range(n_transactions):
        data.append({
            'CustomerID': np.random.choice(customer_ids),
            'Product': np.random.choice(products),
            'Quantity': np.random.randint(1, 10),
            'UnitPrice': round(np.random.uniform(10, 200), 2),
            'Date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(0, 365))
        })
    
    df = pd.DataFrame(data)
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    return df

# Calculate RFM
@st.cache_data
def calculate_rfm(df):
    reference_date = df['Date'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('CustomerID').agg({
        'Date': lambda x: (reference_date - x.max()).days,
        'CustomerID': 'count',
        'TotalPrice': 'sum'
    })
    
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    return rfm

# Perform clustering
@st.cache_data
def perform_clustering(rfm):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Assign labels
    def assign_segment(row):
        if row['Recency'] < 50 and row['Frequency'] > 10 and row['Monetary'] > 1000:
            return 'High-Value'
        elif row['Frequency'] > 5 and row['Monetary'] > 500:
            return 'Regular'
        elif row['Frequency'] < 3 and row['Monetary'] < 200:
            return 'At-Risk'
        else:
            return 'Occasional'
    
    rfm['Segment'] = rfm.apply(assign_segment, axis=1)
    return rfm

# Build recommendation system
@st.cache_data
def build_recommendations(df):
    user_item = df.pivot_table(
        index='CustomerID',
        columns='Product',
        values='Quantity',
        aggfunc='sum',
        fill_value=0
    )
    
    item_similarity = cosine_similarity(user_item.T)
    
    return pd.DataFrame(
        item_similarity,
        index=user_item.columns,
        columns=user_item.columns
    )

# Load data
df = generate_sample_data()
rfm = calculate_rfm(df)
rfm = perform_clustering(rfm)
similarity_df = build_recommendations(df)

# DASHBOARD
if app_mode == "Dashboard":
    st.header("📊 Business Dashboard")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", len(rfm))
    
    with col2:
        st.metric("Total Products", df['Product'].nunique())
    
    with col3:
        st.metric("Total Revenue", f"${df['TotalPrice'].sum():,.0f}")
    
    with col4:
        st.metric("Avg Transaction", f"${df['TotalPrice'].mean():.2f}")
    
    # Customer segments
    st.subheader("📈 Customer Segments")
    
    col1, col2 = st.columns(2)
    
    with col1:
        segment_counts = rfm['Segment'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        ax.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', colors=colors)
        ax.set_title('Customer Segments Distribution')
        st.pyplot(fig)
    
    with col2:
        st.write("**Segment Breakdown:**")
        for segment, count in segment_counts.items():
            percentage = (count / len(rfm)) * 100
            st.write(f"**{segment}:** {count} customers ({percentage:.1f}%)")
    
    # Top products
    st.subheader("🔥 Top 10 Products")
    top_products = df.groupby('Product')['Quantity'].sum().sort_values(ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    top_products.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_xlabel('Total Quantity Sold')
    ax.set_title('Top 10 Best Selling Products')
    plt.gca().invert_yaxis()
    st.pyplot(fig)

# PRODUCT RECOMMENDATIONS
elif app_mode == "Product Recommendations":
    st.header("🔍 Product Recommendation System")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_product = st.selectbox(
            "Select a product:",
            similarity_df.index.tolist()
        )
    
    with col2:
        num_recommendations = st.slider("Number of recommendations", 3, 10, 5)
    
    if st.button("Get Recommendations", type="primary"):
        st.success(f"**Selected Product:** {selected_product}")
        
        # Get recommendations
        similarities = similarity_df[selected_product].sort_values(ascending=False)
        recommendations = similarities.iloc[1:num_recommendations+1]
        
        st.subheader("Recommended Products:")
        
        for i, (product, similarity) in enumerate(recommendations.items(), 1):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{i}. {product}**")
            with col2:
                st.write(f"{similarity*100:.1f}% match")
            st.progress(float(similarity))
            st.markdown("---")

# CUSTOMER SEGMENTATION
elif app_mode == "Customer Segmentation":
    st.header("👥 Customer Segmentation Predictor")
    
    st.info("""
    Enter customer metrics to predict their segment:
    - **Recency:** Days since last purchase
    - **Frequency:** Number of purchases
    - **Monetary:** Total amount spent
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        recency = st.number_input("Recency (days)", 0, 365, 30)
    
    with col2:
        frequency = st.number_input("Frequency", 1, 100, 5)
    
    with col3:
        monetary = st.number_input("Monetary ($)", 1.0, 10000.0, 500.0)
    
    if st.button("Predict Segment", type="primary"):
        # Predict segment
        if recency < 50 and frequency > 10 and monetary > 1000:
            segment = "High-Value"
            color = "#2E86AB"
            desc = "💰 Premium customer - recent, frequent, high spending"
            action = "Offer exclusive deals and loyalty rewards"
        elif frequency > 5 and monetary > 500:
            segment = "Regular"
            color = "#A23B72"
            desc = "🛒 Steady customer - regular purchases"
            action = "Cross-sell complementary products"
        elif frequency < 3 and monetary < 200:
            segment = "At-Risk"
            color = "#C73E1D"
            desc = "⚠️ Risk of churn - inactive, low spending"
            action = "Win-back campaigns with special offers"
        else:
            segment = "Occasional"
            color = "#F18F01"
            desc = "🎯 Occasional buyer - needs engagement"
            action = "Send personalized offers"
        
        st.markdown(f"""
        <div style="background-color: {color}20; padding: 1.5rem; border-radius: 0.5rem; border-left: 5px solid {color};">
            <h3 style="color: {color};">Predicted Segment: {segment}</h3>
            <p>{desc}</p>
            <p><strong>Recommended Action:</strong> {action}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Segment descriptions
    st.subheader("📋 Segment Guide")
    
    segments = {
        "High-Value": "Most valuable customers with recent, frequent purchases",
        "Regular": "Steady customers with good frequency and spending",
        "Occasional": "Infrequent buyers who need more engagement",
        "At-Risk": "Inactive customers who may churn soon"
    }
    
    for seg, desc in segments.items():
        with st.expander(f"**{seg}**"):
            st.write(desc)

# ABOUT
else:
    st.header("ℹ️ About Shopper Spectrum")
    
    st.write("""
    **Shopper Spectrum** is a comprehensive e-commerce analytics platform that provides:
    
    ### Features:
    - 📊 **Customer Segmentation**: Using RFM analysis and K-Means clustering
    - 🔍 **Product Recommendations**: Collaborative filtering based on purchase patterns
    - 📈 **Business Insights**: Real-time analytics and KPIs
    - 🎯 **Predictive Analytics**: Customer behavior predictions
    
    ### Technology Stack:
    - Python 3.x
    - Streamlit
    - Scikit-learn
    - Pandas & NumPy
    - Matplotlib & Seaborn
    
    ### Sample Data:
    This demo uses randomly generated sample data to demonstrate the system's capabilities.
    
    ### How to Use:
    1. **Dashboard**: View overall business metrics
    2. **Product Recommendations**: Get AI-powered product suggestions
    3. **Customer Segmentation**: Predict customer segments
    
    ---
    
    Built with ❤️ using Python and Streamlit
    """)

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #666;">🛒 Shopper Spectrum - E-commerce Analytics Platform</p>',
    unsafe_allow_html=True
)