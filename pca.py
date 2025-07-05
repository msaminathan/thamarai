import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(page_title="PCA Theory & Implementation", layout="wide")

# Custom CSS for better math rendering
st.markdown("""
<style>
.math-container {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
}
.highlight-box {
    background-color: #e8f4f8;
    padding: 15px;
    border-left: 4px solid #1f77b4;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🔍 Principal Component Analysis: Theory & Implementation")
    st.markdown("*Understanding Eigenvalues and Eigenvectors in PCA*")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    section = st.sidebar.selectbox(
        "Choose Section:",
        ["Theory & Mathematics", "Interactive Visualization", "Real Dataset Analysis"]
    )
    
    if section == "Theory & Mathematics":
        theory_section()
    elif section == "Interactive Visualization":
        visualization_section()
    else:
        real_data_section()

def theory_section():
    st.header("📚 Mathematical Theory")
    
    # Theory explanation
    st.markdown("""
    <div class="highlight-box">
    <h3>🎯 Core Concept</h3>
    PCA finds the directions of maximum variance in data by computing eigenvalues and eigenvectors of the covariance matrix.
    </div>
    """, unsafe_allow_html=True)
    
    # Mathematical steps
    st.subheader("📐 Mathematical Steps")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="math-container">
        <h4>Step 1: Data Centering</h4>
        Given data matrix <strong>X ∈ ℝⁿˣᵖ</strong>:
        <br><br>
        <strong>X̃ = X - μ</strong>
        <br><br>
        where μ is the mean vector
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="math-container">
        <h4>Step 2: Covariance Matrix</h4>
        <strong>C = (1/(n-1)) X̃ᵀX̃</strong>
        <br><br>
        C is a symmetric p×p matrix containing covariances between features
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="math-container">
        <h4>Step 3: Eigendecomposition</h4>
        <strong>Cv = λv</strong>
        <br><br>
        or in matrix form:
        <br>
        <strong>C = QΛQᵀ</strong>
        <br><br>
        where Q contains eigenvectors, Λ contains eigenvalues
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="math-container">
        <h4>Step 4: Principal Components</h4>
        <strong>Y = X̃Q</strong>
        <br><br>
        Project data onto principal component space
        </div>
        """, unsafe_allow_html=True)
    
    # Key insights
    st.subheader("💡 Key Insights")
    
    insights = [
        "**Eigenvectors** represent the directions of maximum variance (principal components)",
        "**Eigenvalues** represent the amount of variance explained by each component",
        "**First principal component** has the largest eigenvalue (maximum variance)",
        "**Components are orthogonal** to each other (uncorrelated)",
        "**Dimensionality reduction** by keeping only the top k components"
    ]
    
    for insight in insights:
        st.markdown(f"• {insight}")
    
    # Mathematical properties
    st.subheader("🔢 Mathematical Properties")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="math-container">
        <h4>Variance Explained</h4>
        <strong>Var(Y_i) = λᵢ</strong>
        <br><br>
        The variance of the i-th principal component equals the i-th eigenvalue
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="math-container">
        <h4>Total Variance</h4>
        <strong>∑λᵢ = tr(C)</strong>
        <br><br>
        Sum of eigenvalues equals the trace of covariance matrix
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="math-container">
        <h4>Proportion of Variance</h4>
        <strong>Prop_i = λᵢ / ∑λⱼ</strong>
        <br><br>
        Proportion of variance explained by i-th component
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="math-container">
        <h4>Orthogonality</h4>
        <strong>QᵀQ = I</strong>
        <br><br>
        Eigenvectors are orthonormal
        </div>
        """, unsafe_allow_html=True)

def visualization_section():
    st.header("🎨 Interactive Visualization")
    
    # Parameters
    st.sidebar.subheader("Data Parameters")
    n_samples = st.sidebar.slider("Number of samples", 50, 500, 200)
    n_features = st.sidebar.slider("Number of features", 2, 5, 2)
    noise = st.sidebar.slider("Noise level", 0.1, 2.0, 0.5)
    random_state = st.sidebar.slider("Random seed", 1, 100, 42)
    
    # Generate synthetic data
    X, y = make_blobs(n_samples=n_samples, centers=2, n_features=n_features, 
                      cluster_std=noise, random_state=random_state)
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Data")
        if n_features == 2:
            fig = px.scatter(x=X_scaled[:, 0], y=X_scaled[:, 1], 
                           color=y, title="Original 2D Data")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Data visualization (first 2 dimensions):")
            fig = px.scatter(x=X_scaled[:, 0], y=X_scaled[:, 1], 
                           color=y, title="Original Data (First 2 Dimensions)")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("PCA Transformed Data")
        fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], 
                        color=y, title="PCA Transformed Data")
        st.plotly_chart(fig, use_container_width=True)
    
    # Eigenvalues and eigenvectors
    st.subheader("📊 Eigenvalues and Eigenvectors")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Eigenvalues
        eigenvalues = pca.explained_variance_
        eigenvectors = pca.components_
        
        st.write("**Eigenvalues (Variance Explained):**")
        eigenval_df = pd.DataFrame({
            'Component': [f'PC{i+1}' for i in range(len(eigenvalues))],
            'Eigenvalue': eigenvalues,
            'Variance Explained (%)': pca.explained_variance_ratio_ * 100
        })
        st.dataframe(eigenval_df)
        
        # Scree plot
        fig = px.bar(x=eigenval_df['Component'], y=eigenval_df['Eigenvalue'],
                     title="Scree Plot (Eigenvalues)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Eigenvectors (Principal Components):**")
        eigenvec_df = pd.DataFrame(eigenvectors.T, 
                                  columns=[f'PC{i+1}' for i in range(len(eigenvalues))],
                                  index=[f'Feature{i+1}' for i in range(n_features)])
        st.dataframe(eigenvec_df)
        
        # Cumulative variance
        cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
        fig = px.line(x=range(1, len(cumvar)+1), y=cumvar,
                     title="Cumulative Variance Explained",
                     labels={'x': 'Principal Component', 'y': 'Cumulative Variance (%)'})
        fig.add_hline(y=95, line_dash="dash", line_color="red", 
                     annotation_text="95% threshold")
        st.plotly_chart(fig, use_container_width=True)
    
    # Covariance matrix visualization
    st.subheader("🔍 Covariance Matrix Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Original covariance matrix
        cov_matrix = np.cov(X_scaled.T)
        fig = px.imshow(cov_matrix, 
                       title="Original Covariance Matrix",
                       labels=dict(color="Covariance"))
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("**Covariance Matrix:**")
        st.dataframe(pd.DataFrame(cov_matrix, 
                                 columns=[f'F{i+1}' for i in range(n_features)],
                                 index=[f'F{i+1}' for i in range(n_features)]))
    
    with col2:
        # PCA covariance matrix (should be diagonal)
        cov_pca = np.cov(X_pca.T)
        fig = px.imshow(cov_pca, 
                       title="PCA Covariance Matrix",
                       labels=dict(color="Covariance"))
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("**PCA Covariance Matrix (Diagonal):**")
        st.dataframe(pd.DataFrame(cov_pca, 
                                 columns=[f'PC{i+1}' for i in range(n_features)],
                                 index=[f'PC{i+1}' for i in range(n_features)]))

def real_data_section():
    st.header("🌸 Real Dataset Analysis: Iris Dataset")
    
    # Load iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['species'] = [target_names[i] for i in y]
    
    st.subheader("📊 Dataset Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Info:**")
        st.write(f"• Samples: {X.shape[0]}")
        st.write(f"• Features: {X.shape[1]}")
        st.write(f"• Classes: {len(target_names)}")
        st.write("• Features:", feature_names)
        
        st.write("**First 10 rows:**")
        st.dataframe(df.head(10))
    
    with col2:
        st.write("**Statistical Summary:**")
        st.dataframe(df.describe())
    
    # Standardize and apply PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # PCA Results
    st.subheader("🔍 PCA Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Eigenvalues and explained variance
        eigenvalues = pca.explained_variance_
        explained_var_ratio = pca.explained_variance_ratio_
        
        results_df = pd.DataFrame({
            'Principal Component': [f'PC{i+1}' for i in range(len(eigenvalues))],
            'Eigenvalue': eigenvalues,
            'Explained Variance (%)': explained_var_ratio * 100,
            'Cumulative Variance (%)': np.cumsum(explained_var_ratio) * 100
        })
        
        st.write("**Eigenvalues and Variance Explained:**")
        st.dataframe(results_df)
        
        # Scree plot
        fig = px.bar(x=results_df['Principal Component'], 
                     y=results_df['Eigenvalue'],
                     title="Scree Plot - Iris Dataset")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Eigenvectors (loadings)
        loadings_df = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(len(eigenvalues))],
            index=feature_names
        )
        
        st.write("**Principal Component Loadings:**")
        st.dataframe(loadings_df)
        
        # Cumulative variance plot
        cumvar = np.cumsum(explained_var_ratio) * 100
        fig = px.line(x=range(1, len(cumvar)+1), y=cumvar,
                     title="Cumulative Variance Explained - Iris Dataset",
                     labels={'x': 'Principal Component', 'y': 'Cumulative Variance (%)'})
        fig.add_hline(y=95, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    # Visualizations
    st.subheader("📈 Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # PCA scatter plot
        pca_df = pd.DataFrame(X_pca[:, :2], columns=['PC1', 'PC2'])
        pca_df['species'] = df['species']
        
        fig = px.scatter(pca_df, x='PC1', y='PC2', color='species',
                        title="PCA: First Two Principal Components",
                        labels={'PC1': f'PC1 ({explained_var_ratio[0]:.1%} variance)',
                               'PC2': f'PC2 ({explained_var_ratio[1]:.1%} variance)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Biplot
        fig = go.Figure()
        
        # Add data points
        colors = ['red', 'blue', 'green']
        for i, species in enumerate(target_names):
            mask = y == i
            fig.add_trace(go.Scatter(
                x=X_pca[mask, 0], y=X_pca[mask, 1],
                mode='markers',
                name=species,
                marker=dict(color=colors[i])
            ))
        
        # Add loading vectors
        scale = 3
        for i, feature in enumerate(feature_names):
            fig.add_trace(go.Scatter(
                x=[0, pca.components_[0, i] * scale],
                y=[0, pca.components_[1, i] * scale],
                mode='lines+text',
                line=dict(color='black', width=2),
                text=['', feature],
                textposition='top center',
                showlegend=False,
                name=f'Loading {feature}'
            ))
        
        fig.update_layout(title="PCA Biplot - Iris Dataset",
                         xaxis_title=f'PC1 ({explained_var_ratio[0]:.1%} variance)',
                         yaxis_title=f'PC2 ({explained_var_ratio[1]:.1%} variance)')
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.subheader("🎯 Feature Importance Analysis")
    
    # Calculate feature importance for first two PCs
    pc1_importance = np.abs(pca.components_[0])
    pc2_importance = np.abs(pca.components_[1])
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'PC1 Importance': pc1_importance,
        'PC2 Importance': pc2_importance,
        'Combined Importance': pc1_importance + pc2_importance
    })
    
    importance_df = importance_df.sort_values('Combined Importance', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Feature Importance Rankings:**")
        st.dataframe(importance_df)
    
    with col2:
        fig = px.bar(importance_df, x='Feature', y='Combined Importance',
                    title="Feature Importance in First Two PCs")
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation
    st.subheader("📝 Interpretation")
    
    st.markdown(f"""
    <div class="highlight-box">
    <h4>Key Findings:</h4>
    <ul>
    <li><strong>PC1 explains {explained_var_ratio[0]:.1%}</strong> of the variance</li>
    <li><strong>PC2 explains {explained_var_ratio[1]:.1%}</strong> of the variance</li>
    <li><strong>Together, PC1 and PC2 explain {explained_var_ratio[0] + explained_var_ratio[1]:.1%}</strong> of the total variance</li>
    <li><strong>Most important feature:</strong> {importance_df.iloc[0]['Feature']}</li>
    <li><strong>Dimensionality reduction:</strong> From 4D to 2D while retaining most information</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
