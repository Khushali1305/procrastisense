import streamlit as st
import os

st.set_page_config(page_title="ProcrastiSense — DEBUG", layout="wide")
st.title("🔍 Deployment file diagnostic")

cwd = os.getcwd()
st.write(f"**Current working directory:** `{cwd}`")

st.write("**Full file tree from cwd:**")
tree_lines = []
for root, dirs, files in os.walk(cwd):
    # skip noisy dirs
    dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "venv", ".streamlit")]
    level = root.replace(cwd, "").count(os.sep)
    indent = "  " * level
    tree_lines.append(f"{indent}{os.path.basename(root)}/")
    for f in files:
        tree_lines.append(f"{indent}  {f}")
st.code("\n".join(tree_lines))

st.write("**Specifically checking for the model files:**")
expected = [
    "stage3_models_v2/scaler_v2.pkl",
    "stage3_models_v2/isolation_forest_v2.pkl",
    "stage3_models_v2/logistic_regression_v2.pkl",
    "stage3_models_v2/random_forest_type_v2.pkl",
    "demo_profiles_v2.json",
    "student_productivity_distraction_dataset_20000.csv",
    "hybrid_student_performance_1200.csv",
    "fingerprint_summary_v2.csv",
]
for path in expected:
    full_path = os.path.join(cwd, path)
    exists = os.path.exists(full_path)
    st.write(f"{'✅' if exists else '❌'} `{path}` — {'FOUND' if exists else 'MISSING'} at `{full_path}`")
