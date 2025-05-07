"""
APLICACIÓN STREAMLIT PARA DESPLIEGUE DEL MODELO DE PRECIOS DE VIVIENDAS
"""


import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyod.models.knn import KNN
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# Configurar la página
st.set_page_config(
    page_title="Predictor de Precios de Viviendas test",
    page_icon="🏠",
    layout="wide"
)

# Lista de páginas en orden
PAGES = ["Inicio", "Análisis Exploratorio", "Predicción", "Acerca de"]

# Inicializar estado
if 'data_uploaded' not in st.session_state:
    st.session_state.data_uploaded = False
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'page_index' not in st.session_state:
    st.session_state.page_index = 0
if 'reset_app' not in st.session_state:
    st.session_state.reset_app = False

# Función para reiniciar la app
def limpiar_app():
    st.session_state.reset_app = True

# Ejecutar reinicio si se marcó
if st.session_state.reset_app:
    st.session_state.reset_app = False
    st.session_state.data_uploaded = False
    st.session_state.uploaded_file = None
    st.session_state.page_index = 0
    st.rerun()

# Subida de archivo si no hay datos
if not st.session_state.data_uploaded:
    st.title("🏠 Predictor de Precios de Viviendas test")
    st.markdown("### Por favor, sube un archivo CSV para comenzar.")
    uploaded_file = st.file_uploader("Subir archivo de datos", type=["csv"])
    
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.data_uploaded = True

        # Cargar y guardar el DataFrame directamente en el estado
        st.session_state.df = pd.read_csv(uploaded_file)

        st.rerun()
    st.stop()

@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/housing_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Archivos del modelo no encontrados en carpeta 'models/'.")
        return None, None

# Cargar datos y modelo
df = st.session_state.df
model = load_model()

scaler = MinMaxScaler()

if 'limpieza_aplicada' not in st.session_state:
    st.session_state.limpieza_aplicada = False
if 'df_limpio' not in st.session_state:
    st.session_state.df_limpio = None

# Insertar "Limpieza de Datos" al flujo
if "Limpieza de Datos" not in PAGES:
    PAGES.insert(1, "Limpieza de Datos")


# Navegación por botones
page = PAGES[st.session_state.page_index]

col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    if st.session_state.page_index > 0:
        if st.button("⬅️ Anterior"):
            st.session_state.page_index -= 1
            st.rerun()

with col3:
    if st.session_state.page_index < len(PAGES) - 1:
        if st.button("Siguiente ➡️"):
            st.session_state.page_index += 1
            st.rerun()

# Botón para limpiar app
st.button("🧹 Limpiar aplicación", on_click=limpiar_app)

# Contenido por página
if page == "Inicio":
    st.title("🏠 Predictor de Precios de Viviendas test")
    st.markdown("Esta aplicación permite predecir el precio de viviendas basado en características clave.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### 📊 ¿Qué puede hacer esta aplicación?
        - **Explorar datos**
        - **Visualizar relaciones**
        - **Predecir precios**
        """)
        st.subheader("Vista previa de los datos")
        st.dataframe(df.head())
    
    with col2:
        df_copy = df.copy()

        st.markdown("### 📈 Precio promedio por número de habitaciones")
        fig, ax = plt.subplots()
        df_copy['RM_bin'] = pd.cut(df_copy['RM'], bins=5)
        grouped = df_copy.groupby('RM_bin')['PRICE'].mean().reset_index()
        sns.barplot(x='RM_bin', y='PRICE', data=grouped, ax=ax)
        ax.set_xlabel('Habitaciones (agrupado)')
        ax.set_ylabel('Precio promedio')
        plt.xticks(rotation=45)
        st.pyplot(fig)

elif page == "Limpieza de Datos":
    st.header("🌍 Limpieza de Datos")
    st.markdown("""
    En esta sección puedes aplicar una limpieza básica al conjunto de datos:
    - Eliminar valores nulos
    - Eliminar outliers con detección automatizada
    """)

    with st.form("form_limpieza"):
        eliminar_nulos = st.checkbox("Eliminar valores nulos", value=True)
        porcentaje_outliers = st.slider("Porcentaje de outliers a eliminar", 0.0, 0.2, 0.05, step=0.01)
        aplicar = st.form_submit_button("Aplicar Limpieza")

    if aplicar:
        df_limpio = df.copy()

        if eliminar_nulos:
            df_limpio.dropna(inplace=True)

        # Escalar y detectar outliers (solo columnas numéricas)
        columnas_numericas = df_limpio.select_dtypes(include=['int64', 'float64']).columns
        df_limpio[columnas_numericas] = scaler.fit_transform(df_limpio[columnas_numericas])

        modelo_outliers = KNN(contamination=porcentaje_outliers)
        modelo_outliers.fit(df_limpio)
        etiquetas = modelo_outliers.labels_

        df_limpio = df_limpio[etiquetas == 0].reset_index(drop=True)

        st.session_state.df_limpio = df_limpio
        st.session_state.limpieza_aplicada = True
        st.success("✅ Limpieza aplicada correctamente.")
        st.rerun()

    if st.session_state.limpieza_aplicada and st.session_state.df_limpio is not None:
        st.info("Los datos ya han sido limpiados. Puedes volver a aplicar la limpieza si lo deseas.")
        
        st.subheader("Vista previa de los datos originales")
        st.dataframe(st.session_state.df.head())

        st.subheader("Vista previa de los datos limpios")
        st.dataframe(st.session_state.df_limpio.head())

elif page == "Análisis Exploratorio":
    st.header("Análisis Exploratorio de Datos")
    st.markdown("Visualizaciones para entender relaciones entre variables.")
    
    # Matriz de correlación
    st.subheader("Matriz de Correlación")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, annot=True, fmt='.2f')
    st.pyplot(fig)

    # Relaciones
    st.subheader("Relaciones con el Precio")
    col1, col2 = st.columns(2)
    with col1:
        sns.scatterplot(data=df, x='RM', y='PRICE').set_title('RM vs PRICE')
        st.pyplot(plt.gcf())
        plt.clf()
        
        sns.scatterplot(data=df, x='PTRATIO', y='PRICE').set_title('PTRATIO vs PRICE')
        st.pyplot(plt.gcf())
        plt.clf()
        
    with col2:
        sns.scatterplot(data=df, x='LSTAT', y='PRICE').set_title('LSTAT vs PRICE')
        st.pyplot(plt.gcf())
        plt.clf()

        sns.scatterplot(data=df, x='DIS', y='PRICE').set_title('DIS vs PRICE')
        st.pyplot(plt.gcf())
        plt.clf()
    
    # Distribución
    st.subheader("Distribución de Precios")
    sns.histplot(df['PRICE'], kde=True)
    st.pyplot(plt.gcf())
    plt.clf()

    # Exploración interactiva
    st.subheader("Exploración Interactiva")
    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox("Variable X", options=df.columns.tolist())
    with col2:
        y_var = st.selectbox("Variable Y", options=df.columns.tolist(), index=4)
    sns.scatterplot(data=df, x=x_var, y=y_var)
    st.pyplot(plt.gcf())
    plt.clf()

elif page == "Predicción":
    st.header("Predicción de Precios de Viviendas")
    st.markdown("Ingrese características de una vivienda y obtenga una predicción.")

    if model is not None and scaler is not None:
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                rm_min = int(np.floor(df['RM'].min()))
                rm_max = int(np.ceil(df['RM'].max()))
                rm = st.selectbox("Número de habitaciones (RM)", list(range(rm_min, rm_max + 1)), index=rm_max - rm_min)
                lstat = st.slider("% población bajo estatus (LSTAT)", float(df['LSTAT'].min()), float(df['LSTAT'].max()), float(df['LSTAT'].mean()))
            with col2:
                ptratio = st.slider("Ratio alumno-profesor (PTRATIO)", float(df['PTRATIO'].min()), float(df['PTRATIO'].max()), float(df['PTRATIO'].mean()))
                dis = st.slider("Distancia a centros de empleo (DIS)", float(df['DIS'].min()), float(df['DIS'].max()), float(df['DIS'].mean()))
            submit_button = st.form_submit_button("Predecir Precio")

        if submit_button:
            input_data = np.array([[rm, lstat, ptratio, dis]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            st.success(f"El precio predicho para esta vivienda es: ${prediction:.2f}k")
            st.subheader("Interpretación de la predicción")

elif page == "Acerca de":
    st.header("Acerca de")
    st.markdown("""
    Esta aplicación fue desarrollada para demostrar un flujo completo de predicción usando un modelo
    de Machine Learning entrenado con datos de viviendas.

    - Autor: Tú
    - Modelo: Regressor entrenado con sklearn
    """)
