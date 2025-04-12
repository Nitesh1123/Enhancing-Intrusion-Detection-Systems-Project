import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Intrusion Detection System", layout="centered")

model = joblib.load("ids_model.pkl")

st.title("Intrusion Detection System (IDS)")
st.markdown("Upload a network traffic CSV file to predict **Normal** or **Anomaly** connections.")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(data.head())

    try:
        categorical_cols = ['protocol_type', 'service', 'flag']
        
        # For each categorical column, load the corresponding encoder and transform the data
        for col in categorical_cols:
            encoder = joblib.load(f"{col}_classes.pkl")
            data[col] = data[col].map(lambda s: encoder.transform([s])[0] if s in encoder.classes_ else -1)
    except Exception as e:
        st.error(f"Encoding Error: {e}")

    if st.button("Predict Intrusions"):
        try:
            predictions = model.predict(data)
            output_df = pd.DataFrame({'Prediction': predictions})
            st.success("Prediction complete!")
            st.dataframe(output_df)

            csv = output_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇Download Predictions as CSV",
                data=csv,
                file_name="ids_predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.markdown("---")
    st.subheader("Optional Visualizations")

    if st.checkbox("Show Feature Importance (Random Forest)"):
        try:
            importances = model.feature_importances_
            features = data.columns

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(x=importances, y=features, ax=ax, palette="viridis")
            ax.set_title("Feature Importance Plot")
            st.pyplot(fig)
        except:
            st.warning("Feature importance cannot be displayed.")

    # Correlation Heatmap
    if st.checkbox("Show Correlation Heatmap"):
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = data.corr()
            sns.heatmap(corr_matrix, cmap='coolwarm', ax=ax, linewidths=0.5)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)
        except:
            st.warning("Please upload valid numeric data.")

    # Boxplots
    if st.checkbox("Show Boxplots for Key Features"):
        key_features = ['src_bytes', 'dst_bytes', 'count', 'srv_count']
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()

            for i, feature in enumerate(key_features):
                sns.boxplot(x=data[feature], ax=axes[i])
                axes[i].set_title(f'Boxplot of {feature}')

            plt.tight_layout()
            st.pyplot(fig)
        except:
            st.warning("Boxplots could not be generated.")
