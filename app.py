import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import pandas as pd
import numpy as np

# PyTorch Model Architecture
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = torch.sigmoid(self.fc4(x))
        return x


# Cache the scaler loading
@st.cache_resource
def load_scaler():
    try:
        scaler = joblib.load('scaler.pkl')
        return scaler
    except FileNotFoundError:
        st.error("❌ Error: 'scaler.pkl' file not found. Please ensure the scaler file is in the app directory.")
        return None


# Cache the model loading
@st.cache_resource
def load_model():
    try:
        model = MLP(input_size=10)
        model.load_state_dict(torch.load('mlp_model.pth', map_location=torch.device('cpu')))
        model.eval()
        return model
    except FileNotFoundError:
        st.error("❌ Error: 'mlp_model.pth' file not found. Please ensure the model weights file is in the app directory.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None


# Cache the lookup tables
@st.cache_data
def load_user_features():
    try:
        df = pd.read_csv('user_features_lookup.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Error: 'user_features_lookup.csv' file not found.")
        return None


@st.cache_data
def load_product_features():
    try:
        df = pd.read_csv('product_features_lookup.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Error: 'product_features_lookup.csv' file not found.")
        return None


# Page config
st.set_page_config(
    page_title="🛒 Instacart Customer Reorder Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🛒 Instacart Customer Reorder Predictor")
st.markdown("---")

# Display model accuracy score
st.metric("Model Accuracy Score", "XX.X%")

# Load resources
scaler = load_scaler()
model = load_model()
user_features_df = load_user_features()
product_features_df = load_product_features()

# Check if all resources loaded successfully
if scaler is None or model is None or user_features_df is None or product_features_df is None:
    st.stop()

# User ID selectbox
st.subheader("👤 Select User")
user_ids = sorted(user_features_df['user_id'].unique())
selected_user_id = st.selectbox(
    "Choose User ID",
    user_ids,
    key="user_selectbox"
)

# Get user data
user_data = user_features_df[user_features_df['user_id'] == selected_user_id].iloc[0]
total_orders = user_data['total_orders']
user_reorder_rate = user_data['user_reorder_rate']
average_days_between_orders = user_data['average_days_between_orders']
total_products_bought = user_data['total_products_bought']
distinct_products_bought = user_data['distinct_products_bought']
total_reorders = user_data['total_reorders']

# Display user profile info
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Orders", int(total_orders))
with col2:
    st.metric("Reorder Rate", f"{user_reorder_rate * 100:.1f}%")

st.markdown("---")

# Create tabs for two methods
tab1, tab2 = st.tabs(["Method 1: Get Recommendations", "Method 2: Evaluate Single Product"])

# Helper function to calculate suggested quantity
def calculate_suggested_quantity(user_data):
    """Calculate suggested quantity based on user's average purchase behavior"""
    avg_quantity = max(1, int(user_data['total_products_bought'] / max(1, user_data['total_orders'])))
    return avg_quantity

suggested_quantity = calculate_suggested_quantity(user_data)

# ============ METHOD 1: RECOMMENDATIONS ============
with tab1:
    st.subheader("📊 Recommended Products")
    
    # Slider for add_to_cart_order (applies to all products)
    add_to_cart_order_tab1 = st.slider(
        "📦 Position in Cart (add_to_cart_order)",
        min_value=1,
        max_value=50,
        value=3,
        step=1,
        help="The order position at which products are typically added to the cart",
        key="slider_tab1"
    )
    
    # Generate recommendations button
    if st.button("🔮 Get Product Recommendations", use_container_width=True, key="btn_recommendations"):
        if scaler is None or model is None:
            st.error("Error: Scaler or model not loaded properly.")
        else:
            with st.spinner("Generating recommendations... this may take a moment"):
                # Batch process all products for speed
                # Step 1: Create all feature arrays at once
                all_features = []
                product_names = []
                
                for _, product_row in product_features_df.iterrows():
                    product_names.append(product_row['product_name'])
                    
                    feature_row = [
                        add_to_cart_order_tab1,              # Index 0
                        float(total_orders),                # Index 1
                        float(average_days_between_orders), # Index 2
                        float(total_products_bought),       # Index 3
                        float(distinct_products_bought),    # Index 4
                        float(total_reorders),              # Index 5
                        float(user_reorder_rate),           # Index 6
                        float(product_row['total_purchases']),    # Index 7
                        float(product_row['total_reorders']),     # Index 8
                        float(product_row['reorder_ratio'])       # Index 9
                    ]
                    all_features.append(feature_row)
                
                # Step 2: Convert to numpy array and clean NaN values
                all_features_array = np.array(all_features, dtype=np.float32)
                all_features_array = np.nan_to_num(all_features_array, nan=0.0)
                
                # Step 3: Scale all features at once
                scaled_features = scaler.transform(all_features_array)
                
                # Step 4: Convert to tensor and run batch inference
                tensor_features = torch.tensor(scaled_features, dtype=torch.float32)
                
                with torch.no_grad():
                    prediction_probs = model(tensor_features).numpy().flatten()
                
                # Step 5: Create recommendations dataframe
                recommendations_df = pd.DataFrame({
                    'product_name': product_names,
                    'probability': prediction_probs,
                    'will_reorder': prediction_probs >= 0.5
                }).sort_values('probability', ascending=False)
            
            # Display top 15 recommendations
            top_n = 15
            top_recommendations = recommendations_df.head(top_n)
            
            st.write(f"**Top {top_n} Products Most Likely to be Reordered:**")
            st.write(f"*Suggested Order Quantity: {suggested_quantity} units per product*")
            st.markdown("---")
            
            # Display in a more compact format with columns
            for idx in range(0, len(top_recommendations), 2):
                
                rec1 = top_recommendations.iloc[idx]
               
                status1 = "✅" if rec1['will_reorder'] else "❌"
                st.write(f"**{idx + 1}. {status1} {rec1['product_name']}**")
    
                # This keeps the sub-details (Prob, Qty, Reorder) neatly arranged horizontally
                c1, c2, c3 = st.columns([2, 1, 1])
                with c1:
                 st.write(f"Prob: {rec1['probability'] * 100:.1f}%")
                 with c2:
                  st.write(f"Qty: {suggested_quantity}")
                  with c3:
                   if rec1['will_reorder']:
                    st.write("✓ Reorder")
            else:
             st.write("Maybe")
            st.markdown("---")
            
            # Summary stats
            reorder_count = len(top_recommendations[top_recommendations['will_reorder']])
            st.write(f"**Summary:** {reorder_count} out of {top_n} products are likely to be reordered by this user.")


# ============ METHOD 2: SINGLE PRODUCT EVALUATION ============
with tab2:
    st.subheader("🛍️ Evaluate Specific Product")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**User Profile**")
        st.info(
            f"**Total Orders:** {int(total_orders)}\n\n"
            f"**Reorder Rate:** {user_reorder_rate * 100:.1f}%"
        )
    
    with col2:
        st.write("**Select Product**")
        product_names = sorted(product_features_df['product_name'].unique())
        selected_product_name = st.selectbox(
            "Choose Product Name",
            product_names,
            key="product_selectbox_tab2"
        )
        
        # Get product data
        product_data = product_features_df[product_features_df['product_name'] == selected_product_name].iloc[0]
        product_total_purchases = product_data['total_purchases']
        product_reorder_ratio = product_data['reorder_ratio']
        product_total_reorders = product_data['total_reorders']
        
        st.info(
            f"**Total Purchases:** {int(product_total_purchases)}\n\n"
            f"**Reorder Ratio:** {product_reorder_ratio * 100:.1f}%"
        )
    
    st.markdown("---")
    
    
    add_to_cart_order_tab2 = st.slider(
            "Position in Cart",
            min_value=1,
            max_value=50,
            value=3,
            step=1,
            key="slider_tab2"
        )
    
    # Prediction button
    if st.button("🔮 Evaluate Reorder Likelihood", use_container_width=True, key="btn_single_product"):
        # Compile the 10 features in exact sequence order
        features_array = np.array([
            [
                add_to_cart_order_tab2,           # Index 0
                total_orders,                     # Index 1
                average_days_between_orders,      # Index 2
                total_products_bought,            # Index 3
                distinct_products_bought,         # Index 4
                total_reorders,                   # Index 5
                user_reorder_rate,                # Index 6
                product_total_purchases,          # Index 7
                product_total_reorders,           # Index 8
                product_reorder_ratio             # Index 9
            ]
        ])
        
        # Clean NaN values
        features_array = np.nan_to_num(features_array, nan=0.0)
        
        # Scale using scaler
        scaled_features = scaler.transform(features_array)
        
        # Convert to torch tensor
        tensor_features = torch.tensor(scaled_features, dtype=torch.float32)
        
        # Model prediction with no_grad
        with torch.no_grad():
            prediction_prob = model(tensor_features).item()
        
        st.markdown("---")
        
        # Display results
        if prediction_prob >= 0.5:
            st.success("✅ Prediction: Reorder! The customer will likely purchase this item again.")
        else:
            st.error("❌ Prediction: No Reorder.")
        
        # Display metrics
        col_prob, col_qty_order = st.columns(2)
        
        with col_prob:
            st.metric(
                "Model Probability Confidence",
                f"{prediction_prob * 100:.1f}%"
            )
        
     
            
