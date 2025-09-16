# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 23:21:11 2025

@author:  Eva María Villar Álvarez
DNI: 77401448P
Email: eva.villar.alvarez@gmail.com / evavil01@ucm.es 

Proyecto: BreastHealth Predictor
Descripción: Aplicación web en Streamlit para predicción personalizada de riesgo de cáncer de mama.
             Incluye carga de modelos entrenados, preprocesamiento de datos, predicción de riesgo,
             explicaciones con SHAP, y visualización interactiva de la contribución de las features.
             
Tecnologías: Python 3, Streamlit, SHAP, scikit-learn, LightGBM

Notas: 
- El script permite introducir datos de pacientes, transformarlos según preprocesador guardado,
  realizar la predicción con el modelo entrenado y mostrar interpretaciones SHAP.
- Se incluye categorización de variables, manejo de valores faltantes y visualizaciones interactivas.

"""

import os
import io
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

import streamlit as st
import streamlit.components.v1 as components
import shap

seed = 12345
os.chdir(r"C:\Users\evama\Desktop")



# --- Cargar modelo y preprocessor ---
with open("Win_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("preprocessor.pkl", "rb") as p:
    preprocessor = pickle.load(p)

with open("explainer.pkl", "rb") as e:
    explainer = pickle.load(e)


# --- Funciones de categorización ---
def categorizar_edad(age):
    if age in ('1', '2', '3'):
        return '18-39'
    elif age == '4':
        return '40-44'
    elif age == '5':
        return '45-49'
    elif age=='6':
        return '50-54'
    elif age=='7':
        return '55-59'
    elif age=='8':
        return '60-64'
    elif age=='9':
        return '65-69'
    elif age == '10':
        return '70-74'
    elif age in ('11', '12', '13'):
        return '>=75'

def categorizar_raza(race):
    if race in ('4', '6'):
        return 'Others'
    elif race == '1':
        return 'White'
    elif race == '2':
        return 'Black'
    elif race == '3':
        return 'AAPI'
    elif race == '5':
        return 'Hisp'
    elif race == '9':
        return np.nan
    else:
        return race


# --- otras funciones

def plot_force_patient(i: int, explainer, shap_values_class1, X_ready_df):
    st.markdown("<h4 style='font-weight:bold;'>Explicación SHAP para la paciente:</h4>", unsafe_allow_html=True)


    # --- Force plot interactivo ---
    force_plot_html = shap.force_plot(
        explainer.expected_value[1],
        shap_values_class1[i, :],
        X_ready_df.iloc[i, :],
        matplotlib=False  
    )

    shap_html_io = io.StringIO()
    shap.save_html(shap_html_io, force_plot_html)
    shap_html = shap_html_io.getvalue()

    components.html(shap_html, height=170, scrolling=True)


    # --- Top 5 features más importantes ---
    top_features = X_ready_df.columns[shap_values_class1[i, :].argsort()[::-1]][:5]
    df_top5 = pd.DataFrame({
        'Feature': top_features,
        'Valor': [X_ready_df.iloc[i][f] for f in top_features],
        'Contribución SHAP': [shap_values_class1[i, X_ready_df.columns.get_loc(f)] for f in top_features]
    })
    df_top5['Valor'] = df_top5['Valor'].round(2)
    df_top5['Contribución SHAP'] = df_top5['Contribución SHAP'].round(3)

    st.markdown("<h4 style='font-weight:bold;'>Top caracteristicas más influyentes en la predicción:</h4>", unsafe_allow_html=True)

    st.dataframe(df_top5)



    
# --- TITULO WEB ----------------------
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("""
        <div>
            <h1 style='margin-bottom:0; line-height:1.1;'>
                BreastHealth Predictor 
                <span style='font-size:0.5em; color:gray;'>Predicción personalizada de salud mamaria</span>
            </h1>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("<div style='margin-top:20px; text-align:center;'>", unsafe_allow_html=True)
    st.image("logo.jpg", width=250) 
    st.markdown("</div>", unsafe_allow_html=True)




st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)



#---------INTRODUCIR DATOS PACIENTE -------------------------------------------

## Diccionarios amigables
age_options = {
    1: "Edad 18-29", 2: "Edad 30-34", 3: "Edad 35-39", 4: "Edad 40-44",
    5: "Edad 45-49", 6: "Edad 50-54", 7: "Edad 55-59", 8: "Edad 60-64",
    9: "Edad 65-69", 10: "Edad 70-74", 11: "Edad 75-79", 12: "Edad 80-84",
    13: "Edad >85"
}

race_options = {
    1: "Blanca", 2: "Negra", 
    3: "Asiática / Isleña del Pacífico", 4: "Nativa americana", 
    5: "Hispana", 6: "Otra / mixta", 9: "Desconocida"
}

fd_options = {0: "No", 1: "Sí", 9: "Desconocido"}

menarche_options = {0: "Edad >14", 1: "Edad 12-13", 
                        2: "Edad <12", 9: "Desconocido"}

first_birth_options = {0: "Edad <20", 1: "Edad 20-24", 
                           2: "Edad 25-29", 3: "Edad >30", 
                           4: "Sin hijos", 9: "Desconocido"}

BIRADS_options = {1: "Casi totalmente grasa", 
                  2: "Densidad fibroglandular dispersa", 
                  3: "Densidad heterogénea", 
                  4: "Extremadamente densa", 9: "Desconocido"}

current_hrt_options = {0: "No", 1: "Sí", 9: "Desconocido"}

menopaus_options = {1: "Pre o peri-menopáusica", 
                    2: "Post-menopáusica", 
                    3: "Menopausia quirúrgica", 9: "Desconocido"}

bmi_options = {1: "10-24.99", 2: "25-29.99", 3: "30-34.99", 
               4: "35 o más", 9: "Desconocido"}

biophx_options = {0: "No", 1: "Sí", 9: "Desconocido"}

breast_cancer_history_options = {0: "No", 1: "Si", 9: "Desconocido"}




## Inputs ---------
st.markdown("<h4 style='font-weight:bold;'>Introduce los datos de la paciente:</h4>", unsafe_allow_html=True)


row1 = st.columns(3)
age_group_5_years =row1[0].selectbox("Grupo de edad de la paciente ", 
                                  list(age_options.values()))
edad = [key for key, val in age_options.items() if val == age_group_5_years][0]


race_eth = row1[1].selectbox("Raza / etnia", list(race_options.values()))
race_eth_val = [k for k,v in race_options.items() if v == race_eth][0]


first_degree = row1[2].selectbox("Familiares de primer grado", 
                               list(fd_options.values()))
first_degree_val = [k for k,v in fd_options.items() if v == first_degree][0]



row2 = st.columns(3)
menarche = row2[0].selectbox("Edad menarquia", list(menarche_options.values()))
menarche_val = [k for k,v in menarche_options.items() if v == menarche][0]


first_birth = row2[1].selectbox("Edad primer parto", list(first_birth_options.values()))
first_birth_val = [k for k,v in first_birth_options.items() if v == first_birth][0]


BIRADS = row2[2].selectbox("Densidad mamaria (BIRADS)", list(BIRADS_options.values()))
BIRADS_val = [k for k,v in BIRADS_options.items() if v == BIRADS][0]



row3 = st.columns(3)
current_hrt = row3[0].selectbox("Terapia hormonal", list(current_hrt_options.values()))
current_hrt_val = [k for k,v in current_hrt_options.items() if v == current_hrt][0]


menopaus = row3[1].selectbox("Estado menopáusico", list(menopaus_options.values()))
menopaus_val = [k for k,v in menopaus_options.items() if v == menopaus][0]


bmi_group = row3[2].selectbox("Grupo IMC", list(bmi_options.values()))
bmi_group_val = [k for k,v in bmi_options.items() if v == bmi_group][0]


row4 = st.columns(3)
biophx = row4[0].selectbox("Biopsia previa", list(biophx_options.values()))
biophx_val = [k for k,v in biophx_options.items() if v == biophx][0]

breast_cancer_history = row4[1].selectbox("Historial previo de cáncer de mama", 
                                     list(breast_cancer_history_options.values()))
breast_cancer_history = [k for k,v in breast_cancer_history_options.items() 
                             if v == breast_cancer_history][0]




# ---- GUARDAR PACIENTE -------------------------------------------------------
if st.button("Guardar paciente y predecir"):
    new_patient = {
        "year": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "age_group_5_years": edad,
        "race_eth": race_eth_val,
        "first_degree_hx": first_degree_val,
        "age_menarche": menarche_val,
        "age_first_birth": first_birth_val,
        "BIRADS_breast_density": BIRADS_val,
        "current_hrt": current_hrt_val,
        "menopaus": menopaus_val,
        "bmi_group": bmi_group_val,
        "biophx": biophx_val,
        "breast_cancer_history": breast_cancer_history    
    }

    # guardar para hacer la predicción
    st.session_state.new_row = pd.DataFrame([new_patient])

    # Escribir la ruta donde se quiera guardar
    # Guarda el CSV en la misma carpeta que el script .py
    csv_file = "pacientes.csv"

    # Crear DataFrame con el nuevo paciente
    new_row = pd.DataFrame([new_patient])
    
    # Si el archivo no existe, guardar con cabecera
    if not os.path.exists(csv_file):
        new_row.to_csv(csv_file, index=False, mode="w")
    else:
        # Si ya existe, añadir sin cabecera
        new_row.to_csv(csv_file, index=False, mode="a", header=False)
   
    
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
    st.markdown("<h4 style='font-weight:bold;'>Resultado:</h4>", unsafe_allow_html=True)
    st.success("Paciente guardado correctamente!")





# --- TRANSFORMACION DE DATOS -------------------------------------------
   
    # Crear DataFrame desde los datos guardados
    st.session_state.new_row = pd.DataFrame([new_patient])
    new_row = pd.DataFrame([new_patient])

    # --- procesar el nuevo registro
    new_row = new_row.iloc[:, 1:-1]
    
    new_names = ['age', 'race', 'first_degree', 'menarche', 'first_birth',
                 'birads', 'hrt', 'menopaus', 'bmi', 'bioph']
    new_row.columns = new_names
    
    
    # --- recategorizar
    new_row["age"] = new_row["age"].astype(str).apply(categorizar_edad)
    new_row["race"] = new_row["race"].astype(str).apply(categorizar_raza)
    
    
    
    # --- Crear prop_missings
    new_row[new_names] = new_row[new_names].replace(9, np.nan)
    new_row['prop_missings'] = new_row.isna().mean(axis=1)
    
    # se hace así porque ahora el cálculo se hace sobre 10 columnas
    # originalmente eran 12 (incluyendo count y cancer en el calculo de %), 
    # de ahí el descuadre
    # de esta forma se corrige para que coincida exactamente
    conditions = [
        new_row['prop_missings'] == 0,
        (new_row['prop_missings'] > 0) & (new_row['prop_missings'] <= 0.1),
        (new_row['prop_missings'] > 0.1) & (new_row['prop_missings'] <= 0.2),
        (new_row['prop_missings'] > 0.2) & (new_row['prop_missings'] <= 0.3),
        new_row['prop_missings'] > 0.3
    ]
    choices = ["0.0", "0.08", "0.17", "0.25", ">0.30"]
    new_row['prop_missings'] = np.select(conditions, choices, default=new_row['prop_missings'])
    new_row['prop_missings'] = new_row['prop_missings'].astype(str)
    
    
    # -- categoricas
    cat_nom = ['race', 'first_degree', 'hrt', 'menopaus', 'bioph']
    row_clean = new_row.copy()
    for col in cat_nom:
        row_clean[col] = row_clean[col].fillna("Unknown").astype(str)
    
    
    # --- Variables ordinales 
    col_ord_indicator = ['menarche', 'first_birth', 'birads', 'bmi']
    for col in col_ord_indicator:
        row_clean[f'{col}_NA'] = row_clean[col].isna().astype(int)
        
    row_clean[['bmi', 'birads']] = row_clean[['bmi', 'birads']].astype(float) - 1.0
    
    
    
    # --- Transformación con preprocessor ---
    cat_binary = ['menarche_NA', 'first_birth_NA','birads_NA','bmi_NA' ]
    cat_w_miss = [col for col in ['menarche', 'first_birth', 'birads', 'bmi']
                         if row_clean[col].isna().any()]
    
    age_order = ["18-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", ">=75"]
    missing_order = ["0.0", "0.08", "0.17", "0.25", ">0.30"]
    
    
    # --- Convertir a DataFrame con nombres originales del preprocessor
    
    X_ready = preprocessor.transform(row_clean)
    original_names = preprocessor.get_feature_names_out()
    clean_names = [f.split("__")[-1] for f in original_names]
    
    X_ready_df = pd.DataFrame(X_ready, columns=clean_names)
        
    
    # --- Seleccionar las 10 variables finales para el modelo ---
    features_a_quitar = ['bmi_NA_1', 'menopaus_2', 'hrt_Unknown', 'menarche_NA_1',
                          'race_Others', 'race_Black', 'race_Hisp', 'first_birth_NA_1', 
                          'menopaus_3', 'birads_NA_1', 'first_degree_Unknown', 
                          'bioph_Unknown', 'menopaus_Unknown', 'race_Unknown']
    
    X_ready_df = X_ready_df.drop(columns=features_a_quitar)




# --- PREDICCION ---------------------------------------------------------
    pred = model.predict(X_ready_df)[0]
    prob = model.predict_proba(X_ready_df)[0,1]
    threshold = 0.45  # índice de Youden
    pred = int(prob >= threshold)

    if pred == 1:
        st.error(f"⚠️ Riesgo alto de desarrollar cáncer. Probabilidad: {prob:.2f}")
    else:
        st.success(f"✅ Riesgo bajo. Probabilidad: {prob*100:.2f}%")
    
    
# -- Shap 
    # Parche para compatibilidad con SHAP
    if not hasattr(np, 'bool'):
        np.bool = bool

    shap_values_class1 = explainer.shap_values(X_ready_df)[1] 
    
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

    plot_force_patient(i=0, explainer=explainer, 
                       shap_values_class1=shap_values_class1, 
                       X_ready_df=X_ready_df)
    
