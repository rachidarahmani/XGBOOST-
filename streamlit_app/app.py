import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Wine Quality", page_icon="🍷")
st.title("🍷 Prédiction de la qualité du vin")
st.write(
    "Renseigne les caractéristiques du vin pour savoir s'il est **bon** "
    "(qualité ≥ 7) ou **mauvais**. Résultat indicatif."
)

col1, col2 = st.columns(2)
with col1:
    wine_type = st.selectbox("Type de vin", ["red", "white"])
    fixed_acidity = st.number_input("Fixed acidity", 0.0, 20.0, 7.0)
    volatile_acidity = st.number_input("Volatile acidity", 0.0, 2.0, 0.3)
    citric_acid = st.number_input("Citric acid", 0.0, 2.0, 0.3)
    residual_sugar = st.number_input("Residual sugar", 0.0, 70.0, 2.0)
    chlorides = st.number_input("Chlorides", 0.0, 1.0, 0.05, format="%.3f")
with col2:
    free_so2 = st.number_input("Free sulfur dioxide", 0.0, 300.0, 30.0)
    total_so2 = st.number_input("Total sulfur dioxide", 0.0, 500.0, 100.0)
    density = st.number_input("Density", 0.9, 1.1, 0.995, format="%.4f")
    ph = st.number_input("pH", 2.0, 5.0, 3.2)
    sulphates = st.number_input("Sulphates", 0.0, 3.0, 0.5)
    alcohol = st.number_input("Alcohol (%)", 0.0, 20.0, 11.0)

if st.button("Prédire", type="primary"):
    payload = {
        "fixed_acidity": fixed_acidity, "volatile_acidity": volatile_acidity,
        "citric_acid": citric_acid, "residual_sugar": residual_sugar,
        "chlorides": chlorides, "free_sulfur_dioxide": free_so2,
        "total_sulfur_dioxide": total_so2, "density": density, "ph": ph,
        "sulphates": sulphates, "alcohol": alcohol, "wine_type": wine_type,
    }
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
        r.raise_for_status()
        res = r.json()
        if res["prediction"] == 1:
            st.success(f"✅ Vin **BON** (probabilité : {res['probability_good']:.1%})")
        else:
            st.error(
                f"❌ Vin **MAUVAIS** (probabilité d'être bon : "
                f"{res['probability_good']:.1%})"
            )
        st.progress(res["probability_good"])
    except requests.RequestException as e:
        st.error(f"Erreur d'appel à l'API : {e}")

st.divider()
with st.expander("📊 Performances du modèle et variables importantes"):
    try:
        info = requests.get(f"{API_URL}/model-info", timeout=15).json()
        if info["metrics"]:
            st.write("**Métriques sur le jeu de test**")
            st.dataframe(pd.DataFrame([info["metrics"]]), hide_index=True)
        if info["importances"]:
            st.write("**Importance des variables (XGBoost)**")
            st.bar_chart(pd.Series(info["importances"]).sort_values(), horizontal=True)
    except requests.RequestException:
        st.info("Informations du modèle indisponibles.")
