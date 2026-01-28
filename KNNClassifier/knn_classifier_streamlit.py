import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris

# ---------------- Page Config ----------------
st.set_page_config(page_title="KNN Classifier", layout="wide")
st.title("🔍 KNN Classifier – Streamlit Frontend")

# ---------------- Sidebar ----------------
st.sidebar.header("KNN Parameters")
k = st.sidebar.slider("Number of Neighbors (K)", 1, 15, 5, step=2)
weights = st.sidebar.selectbox("Weights", ["uniform", "distance"])
test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.25)

# ---------------- Data Loading ----------------
st.header("Step 1: Load Dataset")

option = st.radio("Choose Dataset", ["Iris Dataset", "Upload CSV"])

df = None

if option == "Iris Dataset":
    iris = load_iris(as_frame=True)
    df = iris.frame
    st.success("Iris dataset loaded")

if option == "Upload CSV":
    file = st.file_uploader("Upload CSV", type="csv")
    if file:
        df = pd.read_csv(file)
        st.success("CSV uploaded successfully")

if df is None:
    st.stop()

# ---------------- Dataset Preview ----------------
st.header("Step 2: Dataset Overview")
st.dataframe(df.head())
st.write("Shape:", df.shape)
st.write("Missing Values:", df.isnull().sum())

# ---------------- Target Selection ----------------
st.header("Step 3: Select Target Column")
target = st.selectbox("Target Column", df.columns)

y = df[target]
X = df.drop(columns=[target])

# Keep numeric only
X = X.select_dtypes(include=np.number)

# ---------------- Train-Test Split ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=test_size,
    random_state=42,
    stratify=y if y.nunique() > 1 else None
)

# ---------------- Model Training ----------------
st.header("Step 4: Train KNN Classifier")

model = KNeighborsClassifier(n_neighbors=k, weights=weights)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ---------------- Results ----------------
st.subheader("Model Performance")

accuracy = accuracy_score(y_test, y_pred)
st.success(f"Accuracy: {accuracy:.2f}")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
st.pyplot(fig)

# ---------------- Prediction Section ----------------
st.header("Step 5: Predict New Sample")

input_data = []
for col in X.columns:
    val = st.number_input(f"{col}", value=0.0)
    input_data.append(val)

if st.button("Predict"):
    input_scaled = scaler.transform([input_data])
    result = model.predict(input_scaled)
    st.success(f"Predicted Class: {result[0]}")