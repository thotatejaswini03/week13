import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn import tree

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Decision Tree Regression (Pre & Post Pruning)",
    layout="wide"
)

st.title("🌳 Decision Tree Regression (Pre & Post Pruning)")

# -------------------------------------------------
# SIDEBAR – DATA UPLOAD
# -------------------------------------------------
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # -------------------------------------------------
    # FEATURE SELECTION
    # -------------------------------------------------
    st.sidebar.header("🎯 Feature Selection")

    target_column = st.sidebar.selectbox(
        "Select Target Column",
        df.columns
    )

    feature_columns = st.sidebar.multiselect(
        "Select Feature Columns",
        [col for col in df.columns if col != target_column],
        default=[col for col in df.columns if col != target_column][:3]
    )

    # -------------------------------------------------
    # PRUNING STRATEGY
    # -------------------------------------------------
    st.sidebar.header("🌿 Pruning Strategy")
    pruning_method = st.sidebar.radio(
        "Select Pruning Method",
        ["Pre-Pruning", "Post-Pruning"]
    )

    st.sidebar.header("⚙️ Pre-Pruning Parameters")
    max_depth = st.sidebar.slider("Max Depth", 2, 15, 8)
    min_samples_split = st.sidebar.slider("Min Samples Split", 2, 30, 20)
    min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 20, 5)

    ccp_alpha = 0.0
    if pruning_method == "Post-Pruning":
        st.sidebar.header("✂️ Post-Pruning Parameter")
        ccp_alpha = st.sidebar.slider(
            "CCP Alpha", 0.0, 0.05, 0.01, step=0.005
        )

    # -------------------------------------------------
    # MAIN PANEL – DATASET PREVIEW
    # -------------------------------------------------
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # -------------------------------------------------
    # DATA PREPARATION
    # -------------------------------------------------
    X = df[feature_columns]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------------------------------
    # MODEL TRAINING
    # -------------------------------------------------
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        ccp_alpha=ccp_alpha,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)

    # -------------------------------------------------
    # METRIC DISPLAY
    # -------------------------------------------------
    st.markdown(
        f"""
        <div style="background-color:#1f4f2e;padding:15px;border-radius:8px;color:white">
        📈 <b>R² Score:</b> {r2:.3f}
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------------------------------------------------
    # PREDICTION INPUT
    # -------------------------------------------------
    st.subheader("🏠 Prediction Result")

    input_data = {}
    for col in feature_columns:
        input_data[col] = st.number_input(
            f"Enter {col}", float(df[col].mean())
        )

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted {target_column}: {prediction:.2f}")

    # -------------------------------------------------
    # DECISION TREE VISUALIZATION
    # -------------------------------------------------
    st.subheader("🌳 Decision Tree Visualization")

    fig, ax = plt.subplots(figsize=(22, 10))
    tree.plot_tree(
        model,
        feature_names=feature_columns,
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=8,
        ax=ax
    )
    st.pyplot(fig)

    # -------------------------------------------------
    # PRUNING EXPLANATION
    # -------------------------------------------------
    st.subheader("🧠 Pruning Explanation")

    if pruning_method == "Pre-Pruning":
        st.info(
            "Pre-Pruning limits tree growth during training using "
            "max_depth, min_samples_split, and min_samples_leaf."
        )
    else:
        st.info(
            "Post-Pruning removes weak branches after training using "
            "Cost Complexity Pruning (ccp_alpha)."
        )