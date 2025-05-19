"""
APLICACIÓN STREAMLIT PARA DESPLIEGUE DEL MODELO DE PRECIOS DE VIVIENDAS
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyod.models.knn import KNN 
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.exceptions import NotFittedError 
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder # Añadido OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer # MUY IMPORTANTE para preprocesamiento
from sklearn.pipeline import Pipeline # MUY IMPORTANTE para el flujo
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBRegressor, XGBClassifier # Asegúrate de tener xgboost instalado

# Métricas
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Configurar la página
st.set_page_config(
    page_title="Analiza tu dataset",
    page_icon="📊",
    layout="wide"
)

# Lista de páginas en orden
PAGES = ["Inicio", "Limpieza de Datos", "Análisis Exploratorio", "Entrenamiento y Predicción"]
# Nuevas variables de sesión (o actualizadas)
if 'target_variable' not in st.session_state: 
    st.session_state.target_variable = None
if 'feature_importances_df' not in st.session_state:
    st.session_state.feature_importances_df = None
if 'selected_model_type' not in st.session_state:
    st.session_state.selected_model_type = None
if 'trained_pipeline' not in st.session_state: 
    st.session_state.trained_pipeline = None
if 'model_performance_metrics' not in st.session_state: # Métricas del último modelo entrenado
    st.session_state.model_performance_metrics = None
if 'all_trained_model_metrics' not in st.session_state: # NUEVO: Para comparar todos los modelos
    st.session_state.all_trained_model_metrics = {}
if 'features_used_in_model' not in st.session_state: 
    st.session_state.features_used_in_model = None
if 'problem_type' not in st.session_state: 
    st.session_state.problem_type = None
if 'label_encoder_target' not in st.session_state: 
    st.session_state.label_encoder_target = None
if "reset_app" not in st.session_state:
    st.session_state.reset_app = False

# Limpiar estas nuevas variables de sesión en limpiar_app()
def limpiar_app():
    st.session_state.reset_app = True

if st.session_state.reset_app:
    keys_to_reset = ['data_uploaded', 'uploaded_file_content', 'df', 
                     'limpieza_aplicada', 'df_limpio', 'scaler_fitted', 
                     'fitted_scaler_instance', 'target_variable',
                     'feature_importances_df', 'selected_model_type', # Nuevas
                     'trained_pipeline', 'model_performance_metrics', # Nuevas
                     'all_trained_model_metrics', # NUEVO
                     'features_used_in_model', 'problem_type', 'label_encoder_target'] # Nuevas
    # ... (resto de tu función limpiar_app)
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    
    st.session_state.data_uploaded = False
    st.session_state.data_uploaded = False
    st.session_state.uploaded_file_content = None
    st.session_state.df = None
    st.session_state.page_index = 0 
    st.session_state.limpieza_aplicada = False
    st.session_state.df_limpio = None
    st.session_state.reset_app = False
    st.session_state.scaler_fitted = False
    st.session_state.fitted_scaler_instance = None # Considerar si este scaler global sigue siendo necesario
    st.session_state.target_variable = None
    st.session_state.feature_importances_df = None
    st.session_state.selected_model_type = None
    st.session_state.trained_pipeline = None
    st.session_state.model_performance_metrics = None
    st.session_state.all_trained_model_metrics = {} # NUEVO
    st.session_state.features_used_in_model = None
    st.session_state.problem_type = None
    st.session_state.label_encoder_target = None
    st.rerun()

# Inicializar estado
if 'data_uploaded' not in st.session_state:
    st.session_state.data_uploaded = False
if 'uploaded_file_content' not in st.session_state:
    st.session_state.uploaded_file_content = None
if 'df' not in st.session_state: 
    st.session_state.df = None
if 'page_index' not in st.session_state:
    st.session_state.page_index = 0
if 'reset_app' not in st.session_state:
    st.session_state.reset_app = False
if 'limpieza_aplicada' not in st.session_state:
    st.session_state.limpieza_aplicada = False
if 'df_limpio' not in st.session_state:
    st.session_state.df_limpio = None
if 'scaler_fitted' not in st.session_state: 
    st.session_state.scaler_fitted = False
if 'fitted_scaler_instance' not in st.session_state: 
    st.session_state.fitted_scaler_instance = None
if 'target_variable' not in st.session_state: # NUEVO: para la variable objetivo
    st.session_state.target_variable = None


# Función para reiniciar la app
def limpiar_app():
    st.session_state.reset_app = True

# Ejecutar reinicio si se marcó
if st.session_state.reset_app:
    keys_to_reset = ['data_uploaded', 'uploaded_file_content', 'df', 
                     'limpieza_aplicada', 'df_limpio', 'scaler_fitted', 
                     'fitted_scaler_instance', 'target_variable'] # Añadido target_variable
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    
    st.session_state.data_uploaded = False
    st.session_state.uploaded_file_content = None
    st.session_state.df = None
    st.session_state.page_index = 0 
    st.session_state.limpieza_aplicada = False
    st.session_state.df_limpio = None
    st.session_state.reset_app = False
    st.session_state.scaler_fitted = False
    st.session_state.fitted_scaler_instance = None
    st.session_state.target_variable = None # Reseteado
    st.rerun()

# Subida de archivo si no hay datos
if not st.session_state.data_uploaded:
    st.title("🔎 Analiza tu dataset")
    st.markdown("### Por favor, sube un archivo CSV para comenzar.")
    uploaded_file = st.file_uploader("Subir archivo de datos", type=["csv"], key="file_uploader_main")
    st.markdown("""
                
Con esta aplicación, podrás de forma sencilla:
                
🔍 **Explorar tu Información**  
Sube tu archivo y obtén un resumen rápido, visualiza tus datos y selecciona tu variable objetivo.

✨ **Preparar tus Datos**  
Limpia tu información eliminando datos faltantes o valores atípicos para análisis más precisos.

📊 **Descubrir Insights**  
Visualiza patrones y correlaciones entre las variables.

🧠 **Entrenar tus Propios Modelos**  
Te ayudaremos a identificar las variables más importantes y podrás entrenar modelos predictivos (como Random Forest, Árboles de Decisión o XGBoost) para tu variable objetivo.

🎯 **Comparar y Predecir**  
Compara cuál modelo funciona mejor con tus datos y úsalo para hacer nuevas predicciones.
""")


    if uploaded_file is not None:
        st.session_state.uploaded_file_content = uploaded_file 
        st.session_state.data_uploaded = True
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.session_state.target_variable = None # Resetear variable objetivo si se sube nuevo archivo
        except Exception as e:
            st.error(f"Error al leer el archivo CSV: {e}")
            st.session_state.data_uploaded = False 
            st.session_state.df = None
            st.stop()
        st.rerun()
    st.stop()

df = st.session_state.df

if df is None: 
    st.error("Error: El DataFrame no está cargado. Intenta subir el archivo de nuevo.")
    if st.button("Reintentar Carga de Archivo"):
        limpiar_app() 
    st.stop()

@st.cache_resource
def load_model_from_path(model_path='models/housing_model.pkl'):
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        return None
    except Exception:
        return None

model = load_model_from_path() 

if st.session_state.fitted_scaler_instance is None:
    st.session_state.fitted_scaler_instance = MinMaxScaler()


# Navegación por botones Anterior/Siguiente
page_nav_cols = st.columns([1, 8, 1]) 

with page_nav_cols[0]: 
    if st.session_state.page_index > 0:
        if st.button("⬅️ Anterior", use_container_width=True):
            st.session_state.page_index -= 1
            st.rerun()

with page_nav_cols[2]: 
    if st.session_state.page_index < len(PAGES) - 1:
        if st.button("Siguiente ➡️", use_container_width=True):
            st.session_state.page_index += 1
            st.rerun()

if st.button("🧹 Limpiar y Reiniciar Aplicación", type="secondary"):
    limpiar_app()

st.markdown("---") 

page = PAGES[st.session_state.page_index] 

# --- CONTENIDO POR PÁGINA ---

if page == "Inicio":
    st.title(f"🔎 {page}: Visión General del Dataset") 
    st.markdown("Bienvenido/a. Aquí obtendrás una visión general de tus datos y podrás definir tu variable objetivo.")
    st.markdown("---")
    # 1. Resumen General del Dataset (sin cambios)
    st.header("1. Resumen General del Dataset")
    if df is not None and not df.empty:
        col_summary1, col_summary2 = st.columns(2)
        with col_summary1:
            st.subheader("🔢 Dimensiones")
            st.write(f"**Número de Filas:** {df.shape[0]}")
            st.write(f"**Número de Columnas:** {df.shape[1]}")
            st.subheader("📜 Tipos de Datos por Columna")
            df_types = df.dtypes.astype(str).reset_index() 
            df_types.columns = ['Columna', 'Tipo de Dato']
            st.dataframe(df_types, use_container_width=True, height=300)
        with col_summary2:
            st.subheader("❓ Valores Faltantes")
            missing_values = df.isnull().sum().reset_index()
            missing_values.columns = ['Columna', 'Valores Faltantes']
            missing_values_filtered = missing_values[missing_values['Valores Faltantes'] > 0] 
            if not missing_values_filtered.empty:
                st.dataframe(missing_values_filtered, use_container_width=True, height=300)
            else:
                st.success("✅ ¡Excelente! No se encontraron valores faltantes en tu dataset.")
        with st.expander("📊 Ver Estadísticas Descriptivas (Columnas Numéricas)"):
            st.markdown("""

Estas métricas te ofrecen un resumen general del comportamiento de cada variable numérica en tu conjunto de datos:

- **count**: Número de valores no nulos (sin contar vacíos).
- **mean**: Promedio de los valores.
- **std**: Desviación estándar, indica qué tanto varían los datos respecto a la media.
- **min**: Valor mínimo observado.
- **25% (Q1)**: Primer cuartil, el 25% de los datos son menores o iguales a este valor.
- **50% (Q2 / mediana)**: Valor central de los datos, el 50% de los valores están por debajo y el 50% por encima.
- **75% (Q3)**: Tercer cuartil, el 75% de los datos son menores o iguales a este valor.
- **max**: Valor máximo observado.

Estas estadísticas son útiles para entender la distribución, identificar posibles outliers y guiar decisiones de limpieza o transformación de los datos.
""")

            numeric_cols = df.select_dtypes(include=np.number)
            if not numeric_cols.empty:
                st.dataframe(numeric_cols.describe().T)
            else:
                st.info("No hay columnas numéricas para mostrar estadísticas descriptivas.")
        with st.expander("📝 Ver Estadísticas Descriptivas (Columnas Categóricas/Objeto)"):

            categorical_cols = df.select_dtypes(include=['object', 'category'])
            st.markdown("""
### 🧾 Estadísticas Descriptivas (Columnas Categóricas / Objeto)

Estas métricas permiten entender la distribución general de las variables no numéricas:

- **count**: Número de valores no nulos (sin contar vacíos).
- **unique**: Cantidad de valores únicos distintos en la columna.
- **top**: Valor más frecuente (modo).
- **freq**: Frecuencia del valor más común (cuántas veces aparece el top).

Estas estadísticas son útiles para identificar la categoría predominante, verificar la diversidad de respuestas y detectar posibles valores anómalos o dominantes en tus variables categóricas.
""")

            if not categorical_cols.empty:
                st.dataframe(categorical_cols.describe().T)
            else:
                st.info("No hay columnas categóricas/objeto para mostrar estadísticas descriptivas.")
        st.markdown("---")
    else:
        st.warning("El DataFrame está vacío o no se ha cargado correctamente.")

    # 2. Vista Previa del Dataset (sin cambios)
    st.header("2. Vista Previa del Dataset")
    if df is not None and not df.empty:
        # ... (código existente slider y dataframe) ...
        num_rows_preview = st.slider(
            "Selecciona el número de filas para previsualizar:",
            min_value=1, max_value=min(20, df.shape[0]), 
            value=min(5, df.shape[0]), key="preview_slider_inicio" 
        )
        st.dataframe(df.head(num_rows_preview), use_container_width=True)
    st.markdown("---")

    # 3. Definir Variable Objetivo (NUEVO)
    st.header("3. Definir Variable Objetivo")
    st.markdown("""
    Seleccionar una variable objetivo ayudará a enfocar algunos de los análisis y visualizaciones 
    en las páginas siguientes, especialmente en la sección de 'Análisis Exploratorio'.
    """)

    if df is not None and not df.empty:
        column_options = [None] + df.columns.tolist()
        
        # Determinar el índice de la selección actual o None
        current_target_val = st.session_state.get('target_variable', None)
        if current_target_val in column_options:
            current_target_index = column_options.index(current_target_val)
        else: # Si el target guardado no está en las opciones (p.ej. se borró), default a None
            st.session_state.target_variable = None
            current_target_index = 0 # Índice de None

        selected_target = st.selectbox(
            "Selecciona tu variable objetivo:",
            options=column_options,
            index=current_target_index,
            format_func=lambda x: "No seleccionar" if x is None else x,
            key="target_variable_selector_inicio"
        )

        # Actualizar solo si hay un cambio real para evitar reruns innecesarios si el valor es el mismo
        if selected_target != st.session_state.target_variable:
            st.session_state.target_variable = selected_target
            st.rerun() # Rerun para que el estado se actualice y se refleje en la UI

        if st.session_state.target_variable:
            st.success(f"Variable objetivo seleccionada: **{st.session_state.target_variable}**")
            if st.session_state.target_variable not in df.columns: # Doble chequeo por si acaso
                 st.warning(f"Advertencia: La variable objetivo '{st.session_state.target_variable}' previamente seleccionada ya no existe en el dataset. Por favor, selecciona una nueva.")
                 st.session_state.target_variable = None
                 st.rerun()
        else:
            st.info("No se ha seleccionado una variable objetivo. Algunos análisis específicos del objetivo estarán desactivados o serán más generales.")



elif page == "Limpieza de Datos":
    st.title(f"🧹 {page}: Preparación de Datos") 
    # ... (resto del código de Limpieza de Datos sin cambios relevantes a target_variable aquí) ...
    # Importante: Si la limpieza elimina la variable objetivo, el usuario será notificado en la página de EDA.
    st.markdown("""
    En esta sección puedes aplicar una limpieza básica al conjunto de datos. 
    Los cambios aquí afectarán las páginas subsiguientes de análisis y predicción.
    Los datos procesados se guardarán como 'datos limpios'.
    """)

    if st.session_state.limpieza_aplicada and st.session_state.df_limpio is not None:
        st.success("🎉 ¡Los datos ya fueron procesados con los parámetros de limpieza anteriores!")
        st.markdown("Si deseas aplicar una nueva limpieza, los datos originales se usarán como base.")
        if st.button("Restaurar datos originales para nueva limpieza", key="restore_limpieza"):
            st.session_state.limpieza_aplicada = False
            st.rerun()

    df_para_limpiar = df.copy() 
    
    st.subheader("Opciones de Limpieza")
    with st.form("form_limpieza"):
        eliminar_nulos = st.checkbox("Eliminar filas con valores nulos (NaN)", value=True)
        columnas_numericas_limpieza = df_para_limpiar.select_dtypes(include=np.number).columns.tolist()
        if not columnas_numericas_limpieza:
            porcentaje_outliers = 0.0
            st.info("No hay columnas numéricas en el dataset para la detección de outliers con KNN.")
        else:
            porcentaje_outliers = st.slider("Porcentaje de outliers a identificar y eliminar (usando KNN)", 0.0, 0.25, 0.05, step=0.01,
                                            help="Esto se aplica a las columnas numéricas...",
                                            disabled=not bool(columnas_numericas_limpieza))
        aplicar_limpieza = st.form_submit_button("🚀 Aplicar Limpieza")

    if aplicar_limpieza:
        with st.spinner("Aplicando limpieza... Por favor espera."):
            df_resultado_limpieza = df_para_limpiar.copy()
            original_shape = df_resultado_limpieza.shape
            if eliminar_nulos:
                # ... (lógica de eliminar nulos) ...
                nulos_antes = df_resultado_limpieza.isnull().sum().sum()
                df_resultado_limpieza.dropna(inplace=True)
                nulos_despues = df_resultado_limpieza.isnull().sum().sum()
                st.write(f"Se eliminaron {nulos_antes - nulos_despues} celdas con valores nulos.")
                if df_resultado_limpieza.empty:
                    st.error("El dataset quedó vacío después de eliminar nulos.")
                    st.stop()

            if porcentaje_outliers > 0 and not df_resultado_limpieza.empty and columnas_numericas_limpieza:
                # ... (lógica de outliers) ...
                columnas_numericas_actuales = df_resultado_limpieza.select_dtypes(include=np.number).columns
                if not columnas_numericas_actuales.empty:
                    df_temp_scaled = df_resultado_limpieza.copy()
                    try:
                        temp_scaler_for_outliers = MinMaxScaler().fit(df_para_limpiar[columnas_numericas_actuales])
                        df_temp_scaled[columnas_numericas_actuales] = temp_scaler_for_outliers.transform(df_temp_scaled[columnas_numericas_actuales])
                        modelo_outliers = KNN(contamination=porcentaje_outliers)
                        modelo_outliers.fit(df_temp_scaled[columnas_numericas_actuales])
                        etiquetas_outliers = modelo_outliers.labels_
                        outliers_eliminados = np.sum(etiquetas_outliers == 1)
                        df_resultado_limpieza = df_temp_scaled[etiquetas_outliers == 0]
                        st.write(f"Se identificaron y eliminaron {outliers_eliminados} filas como outliers.")
                        if df_resultado_limpieza.empty:
                            st.error("El dataset quedó vacío después de eliminar outliers.")
                            st.stop()
                    except Exception as e:
                        st.error(f"Error durante la detección de outliers: {e}.")
                else:
                    st.info("No quedan columnas numéricas para la detección de outliers después del primer paso.")
            
            st.session_state.df_limpio = df_resultado_limpieza.reset_index(drop=True)
            st.session_state.limpieza_aplicada = True
            final_shape = st.session_state.df_limpio.shape
            st.write(f"Dimensiones originales: {original_shape}, Dimensiones después de limpieza: {final_shape}")
        st.success("✅ Limpieza procesada.")
        st.rerun() 

    if st.session_state.limpieza_aplicada and st.session_state.df_limpio is not None:
        # ... (comparación de datos) ...
        st.subheader("Comparación de Datos")
        col_orig, col_limp = st.columns(2)
        with col_orig:
            st.markdown("**Datos Originales (Primeras filas)**")
            st.write(f"Dimensiones: {df.shape}")
            st.dataframe(df.head(), height=200, use_container_width=True)
        with col_limp:
            st.markdown("**Datos Limpios (Primeras filas)**")
            st.write(f"Dimensiones: {st.session_state.df_limpio.shape}")
            st.dataframe(st.session_state.df_limpio.head(), height=200, use_container_width=True)
        if st.session_state.df_limpio.equals(df) and (df.shape == st.session_state.df_limpio.shape):
             st.info("Los datos limpios son idénticos a los originales.")


elif page == "Análisis Exploratorio":
    st.title(f"📊 {page}: Entendiendo tus Datos") 
    
    df_eda = st.session_state.df_limpio if st.session_state.limpieza_aplicada and st.session_state.df_limpio is not None else df
    target_var_name = st.session_state.get('target_variable', None)

    if st.session_state.limpieza_aplicada and st.session_state.df_limpio is not None:
        st.info("Mostrando análisis sobre los **datos limpios**.")
    else:
        st.info("Mostrando análisis sobre los **datos originales**. Puedes aplicar la limpieza en la página 'Limpieza de Datos'.")

    if df_eda.empty:
        st.warning("El dataset para análisis está vacío. Verifica los pasos anteriores.")
        st.stop()

    # Verificar si la variable objetivo seleccionada aún existe
    if target_var_name and target_var_name not in df_eda.columns:
        st.warning(f"La variable objetivo '{target_var_name}' fue seleccionada pero ya no existe en el dataset procesado. "
                   "Por favor, selecciona una nueva variable objetivo en la página de 'Inicio'.")
        st.session_state.target_variable = None # Resetearla
        target_var_name = None # Asegurar que es None para la lógica subsiguiente

    st.markdown("Visualizaciones para entender relaciones entre variables.")
    st.markdown("---")
    
    # Sección de Análisis Enfocado en la Variable Objetivo
    if target_var_name:
        st.header(f"🔍 Análisis Enfocado en: '{target_var_name}'")
        target_series = df_eda[target_var_name]

        # A. Distribución de la Variable Objetivo
        st.subheader(f"A. Distribución de '{target_var_name}'")
        fig_target_dist, ax_target_dist = plt.subplots(figsize=(8, 5))
        if pd.api.types.is_numeric_dtype(target_series):
            sns.histplot(target_series, kde=True, ax=ax_target_dist, color="skyblue")
            ax_target_dist.set_title(f"Distribución de {target_var_name} (Numérica)")
        elif pd.api.types.is_categorical_dtype(target_series) or target_series.dtype == 'object':
            # Limitar el número de categorías para el countplot para evitar gráficos muy grandes
            top_n = 20 
            if target_series.nunique() > top_n:
                st.caption(f"Mostrando las {top_n} categorías más frecuentes de '{target_var_name}'.")
                value_counts = target_series.value_counts().nlargest(top_n)
                sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax_target_dist, palette="viridis")
            else:
                 sns.countplot(y=target_series, ax=ax_target_dist, order=target_series.value_counts().index, palette="viridis")
            ax_target_dist.set_title(f"Distribución de {target_var_name} (Categórica)")
            plt.xticks(rotation=45, ha='right')
        else:
            st.info(f"El tipo de dato de '{target_var_name}' ({target_series.dtype}) no es directamente graficable como numérica o categórica aquí.")
        plt.tight_layout()
        st.pyplot(fig_target_dist)
        plt.clf() # Limpiar figura actual

        # B. Relaciones con otras variables (dependiendo del tipo de target)
        st.subheader(f"B. Relaciones de otras variables con '{target_var_name}'")
        other_vars = df_eda.columns.drop(target_var_name, errors='ignore')

        if pd.api.types.is_numeric_dtype(target_series): # Target Numérico
            numeric_cols_for_scatter = df_eda[other_vars].select_dtypes(include=np.number).columns
            if len(numeric_cols_for_scatter) > 0:
                default_scatter_cols = numeric_cols_for_scatter.tolist()[:min(4, len(numeric_cols_for_scatter))]
                selected_cols_scatter = st.multiselect(
                    f"Selecciona variables numéricas para comparar con '{target_var_name}' (Scatter Plots):",
                    options=numeric_cols_for_scatter.tolist(),
                    default=default_scatter_cols,
                    key="scatter_vs_numeric_target"
                )
                if selected_cols_scatter:
                    cols_per_row = st.number_input("Gráficos por fila (scatter):", 1, 4, 2, key="cols_scatter_target")
                    num_rows = (len(selected_cols_scatter) + cols_per_row - 1) // cols_per_row
                    fig_scatter, axes_scatter = plt.subplots(num_rows, cols_per_row, figsize=(6 * cols_per_row, 5 * num_rows), squeeze=False)
                    axes_scatter = axes_scatter.flatten()
                    for i, col_name in enumerate(selected_cols_scatter):
                        sns.scatterplot(data=df_eda, x=col_name, y=target_var_name, ax=axes_scatter[i], alpha=0.7, color="coral")
                        axes_scatter[i].set_title(f'{col_name} vs {target_var_name}')
                    for j in range(len(selected_cols_scatter), len(axes_scatter)): fig_scatter.delaxes(axes_scatter[j])
                    plt.tight_layout()
                    st.pyplot(fig_scatter)
                    plt.clf()
            else:
                st.info(f"No hay otras variables numéricas para generar scatter plots contra '{target_var_name}'.")

        elif pd.api.types.is_categorical_dtype(target_series) or target_series.dtype == 'object': # Target Categórico
            numeric_cols_for_boxplot = df_eda[other_vars].select_dtypes(include=np.number).columns
            if len(numeric_cols_for_boxplot) > 0:
                selected_numeric_for_boxplot = st.selectbox(
                    f"Selecciona una variable numérica para comparar con '{target_var_name}' (Box Plots):",
                    options=numeric_cols_for_boxplot.tolist(),
                    key="boxplot_vs_categorical_target"
                )
                if selected_numeric_for_boxplot:
                    fig_boxplot, ax_boxplot = plt.subplots(figsize=(10, 6))
                    # Limitar categorías en eje X del boxplot si son muchas
                    order_boxplot = target_series.value_counts().nlargest(10).index if target_series.nunique() > 10 else None
                    if order_boxplot is not None: st.caption(f"Mostrando boxplots para las 10 categorías más frecuentes de '{target_var_name}'.")

                    sns.boxplot(data=df_eda, x=target_var_name, y=selected_numeric_for_boxplot, ax=ax_boxplot, palette="Set2", order=order_boxplot)
                    ax_boxplot.set_title(f'{selected_numeric_for_boxplot} por categorías de {target_var_name}')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig_boxplot)
                    plt.clf()
            else:
                st.info(f"No hay variables numéricas para generar box plots contra '{target_var_name}'.")
            # Podrías añadir aquí comparaciones con otras variables categóricas (e.g., stacked bar charts)
        st.markdown("---")
    else:
        st.info("No se ha seleccionado una variable objetivo en la página de 'Inicio'. "
                  "Los siguientes análisis son generales.")
        st.markdown("---")

    # Análisis Generales (no dependen directamente de una variable objetivo predefinida)
    st.header("🔬 Análisis Generales del Dataset")

    st.subheader("Matriz de Correlación (Columnas Numéricas)")
    st.markdown("""
### 🔗 ¿Cómo Interpretar una Matriz de Correlación?

Una matriz de correlación muestra qué tan relacionadas están dos variables numéricas entre sí. Los valores van de **-1 a 1**:

- **1**: Correlación positiva perfecta – cuando una variable sube, la otra también.
- **0**: Sin correlación – no hay una relación lineal aparente.
- **-1**: Correlación negativa perfecta – cuando una variable sube, la otra baja.

#### Consejos para interpretar:
- Busca valores cercanos a **1 o -1** para identificar relaciones fuertes.
- Una correlación alta no siempre significa causalidad.
- Puedes usar esto para identificar variables redundantes o relevantes para modelos predictivos.

Puedes hacer clic en los valores o usar un heatmap para ver rápidamente qué pares de variables tienen relaciones fuertes o débiles.
""")

    numeric_cols_eda = df_eda.select_dtypes(include=np.number)
    if len(numeric_cols_eda.columns) > 1:
        corr = numeric_cols_eda.corr()
        fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0, 
                    square=True, linewidths=.5, annot=True, fmt='.2f', annot_kws={"size": 8}, ax=ax_corr)
        plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
        plt.tight_layout(); st.pyplot(fig_corr); plt.clf()
    else:
        st.info("No hay suficientes columnas numéricas para generar una matriz de correlación.")

    st.subheader("Distribución de Variables Individuales")
    # ... (código existente de distribución de variables, sin cambios importantes) ...
    dist_cols = df_eda.columns.tolist()
    # Intenta preseleccionar la variable objetivo si está definida y existe
    default_dist_index = 0
    if target_var_name and target_var_name in dist_cols:
        default_dist_index = dist_cols.index(target_var_name)
    
    selected_col_dist = st.selectbox(
        "Selecciona una variable para ver su distribución:",
        options=dist_cols, index=default_dist_index, key="dist_selector_general"
    )
    if selected_col_dist:
        fig_dist_gen, ax_dist_gen = plt.subplots(figsize=(8,5))
        if pd.api.types.is_numeric_dtype(df_eda[selected_col_dist]):
            sns.histplot(df_eda[selected_col_dist], kde=True, ax=ax_dist_gen, color="teal")
        else: 
            top_n_dist = 20
            if df_eda[selected_col_dist].nunique() > top_n_dist:
                st.caption(f"Mostrando las {top_n_dist} categorías más frecuentes.")
                value_counts_dist = df_eda[selected_col_dist].value_counts().nlargest(top_n_dist)
                sns.barplot(x=value_counts_dist.index, y=value_counts_dist.values, ax=ax_dist_gen, palette="coolwarm")
            else:
                sns.countplot(y=df_eda[selected_col_dist], ax=ax_dist_gen, order = df_eda[selected_col_dist].value_counts().index, palette="coolwarm")
            plt.xticks(rotation=45, ha='right')
        ax_dist_gen.set_title(f'Distribución de {selected_col_dist}')
        plt.tight_layout(); st.pyplot(fig_dist_gen); plt.clf()

    st.subheader("Exploración Interactiva")
    if len(df_eda.columns) > 1:
        col1_exp, col2_exp = st.columns(2)
        all_cols_eda = df_eda.columns.tolist()
        with col1_exp:
            x_var = st.selectbox("Variable X:", options=all_cols_eda, index=0, key="x_var_eda_general")
        with col2_exp:
            y_var_options = [col for col in all_cols_eda if col != x_var]
            if not y_var_options: y_var_options = all_cols_eda 
            y_var_default_index = 0
            if target_var_name and target_var_name in y_var_options and x_var != target_var_name:
                y_var_default_index = y_var_options.index(target_var_name)
            y_var = st.selectbox("Variable Y:", options=y_var_options, index=y_var_default_index, key="y_var_eda_general")
        if x_var and y_var:
            fig_sc_int, ax_sc_int = plt.subplots(figsize=(8,6))

            hue_var_options = [None] + [col for col in all_cols_eda if col != x_var and col != y_var and df_eda[col].nunique() < 20]
            hue_var = None
            if hue_var_options != [None]:
                hue_var = st.selectbox("Variable para color (Hue, opcional):", options=hue_var_options, index=0, key="hue_var_eda_general")

            try:
                sns.scatterplot(data=df_eda, x=x_var, y=y_var, hue=hue_var, ax=ax_sc_int, palette="magma", alpha=0.7)
                ax_sc_int.set_title(f'{x_var} vs {y_var}' + (f' (Color por {hue_var})' if hue_var else ''))
                if hue_var: ax_sc_int.legend(title=hue_var, bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout(); st.pyplot(fig_sc_int); plt.clf()
            except Exception as e:
                st.error(f"No se pudo generar el gráfico: {e}")

# ... (código de las páginas anteriores) ...

elif page == "Entrenamiento y Predicción":
    st.title(f"🚀 {page}: Construye, Evalúa y Usa tu Modelo")

    # --- PASO 0: PREPARACIÓN DE DATOS ---
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("Por favor, primero carga un dataset en la página de 'Inicio'.")
        st.stop()

    df_source = st.session_state.df_limpio if st.session_state.limpieza_aplicada and st.session_state.df_limpio is not None else st.session_state.df
    
    if df_source.empty:
        st.warning("El dataset está vacío. Verifica los pasos anteriores.")
        st.stop()

    target_var_name = st.session_state.get('target_variable', None)
    if not target_var_name or target_var_name not in df_source.columns:
        st.error("Variable objetivo no definida o no encontrada. Por favor, selecciónala en la página de 'Inicio'.")
        st.stop()

    st.info(f"Usando variable objetivo: **{target_var_name}**")
    y = df_source[target_var_name].copy() # Series para la variable objetivo
    X_original = df_source.drop(columns=[target_var_name]).copy() # DataFrame de características originales

    # Determinar tipo de problema y preprocesar 'y' para clasificación si es necesario
    if pd.api.types.is_numeric_dtype(y):
        st.session_state.problem_type = "Regression"
    elif y.nunique() > 1:
        st.session_state.problem_type = "Classification"
        le_target = LabelEncoder()
        y = pd.Series(le_target.fit_transform(y), name=target_var_name) # y ahora es numérico
        st.session_state.label_encoder_target = le_target # Guardar para decodificar predicciones
        st.caption(f"Variable objetivo '{target_var_name}' codificada para clasificación. Clases originales: {list(le_target.classes_)}")
    else:
        st.error("La variable objetivo no es adecuada (p.ej., tiene un solo valor único).")
        st.stop()
    st.write(f"Tipo de problema detectado: **{st.session_state.problem_type}**")
    st.markdown("---")

    # --- PASO 1: CÁLCULO Y SELECCIÓN DE CARACTERÍSTICAS ---
    st.header("1. Selección de Características")
    
    # Usar solo columnas numéricas o categóricas simples para el cálculo de importancia inicial
    # Para RandomForest, es mejor tener todo numérico.
    X_for_importance = X_original.copy()
    categorical_cols_for_imp = X_for_importance.select_dtypes(include=['object', 'category']).columns
    
    # Simplificación: Para el cálculo de importancia, se usarán solo las numéricas o se intentará una codificación simple.
    # Una opción es usar RandomForest que puede dar importancia incluso con OHE, pero el OHE puede diluir la importancia.
    # Otra es usar SelectKBest con f_regression/chi2.
    # Por simplicidad, usaremos RF en las numéricas y mostraremos una advertencia para las categóricas en este paso.
    
    numeric_cols_for_imp = X_for_importance.select_dtypes(include=np.number).columns
    if not numeric_cols_for_imp.empty:
        X_numeric_for_imp = X_for_importance[numeric_cols_for_imp]
        
        if st.button("Calcular Importancia de Características Numéricas (con Random Forest)", key="calc_imp_btn"):
            with st.spinner("Calculando..."):
                if st.session_state.problem_type == "Regression":
                    model_imp = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                else: # Classification
                    model_imp = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                
                try:
                    model_imp.fit(X_numeric_for_imp, y) # y ya está codificada si es clasificación
                    importances = model_imp.feature_importances_
                    st.session_state.feature_importances_df = pd.DataFrame({
                        'Característica': X_numeric_for_imp.columns,
                        'Importancia': importances
                    }).sort_values(by='Importancia', ascending=False).reset_index(drop=True)
                except Exception as e:
                    st.error(f"Error al calcular importancia: {e}")
                    st.session_state.feature_importances_df = None
        
        if st.session_state.feature_importances_df is not None:
            st.subheader("Importancia de Características Numéricas:")
            num_to_show = st.slider("Mostrar N características más importantes:", 
                                    min_value=1, 
                                    max_value=len(st.session_state.feature_importances_df), 
                                    value=min(10, len(st.session_state.feature_importances_df)),
                                    key="num_imp_feat_slider")
            
            fig_imp, ax_imp = plt.subplots(figsize=(10, num_to_show * 0.5 if num_to_show * 0.5 > 4 else 4)) # Ajustar tamaño
            top_features_df = st.session_state.feature_importances_df.head(num_to_show)
            sns.barplot(x='Importancia', y='Característica', data=top_features_df, ax=ax_imp, palette="viridis")
            ax_imp.set_title(f"Top {num_to_show} Características Numéricas Más Importantes")
            plt.tight_layout()
            st.pyplot(fig_imp)
            plt.clf()

            default_selected_features = st.session_state.feature_importances_df['Característica'].head(min(5, len(st.session_state.feature_importances_df))).tolist()
            st.info(f"Sugerencia (Top {len(default_selected_features)} numéricas): {', '.join(default_selected_features)}")
        else:
            default_selected_features = X_original.columns.tolist()[:3] # Fallback
    else:
        st.warning("No hay características numéricas para calcular la importancia con el método actual. "
                   "Se usarán todas las características disponibles para la selección.")
        default_selected_features = X_original.columns.tolist()[:3] # Fallback

    if not categorical_cols_for_imp.empty:
        st.caption(f"Características categóricas encontradas: {', '.join(categorical_cols_for_imp)}. "
                   "Estas se codificarán (One-Hot Encoding) si las seleccionas para el modelo. "
                   "Su importancia no se calculó directamente en el gráfico anterior.")

    # Selección de características por el usuario (de las originales)
    st.session_state.features_used_in_model = st.multiselect(
        "Selecciona las características para entrenar el modelo (numéricas y categóricas):",
        options=X_original.columns.tolist(),
        default=default_selected_features,
        key="feature_selector_model"
    )

    if not st.session_state.features_used_in_model:
        st.warning("Por favor, selecciona al menos una característica para entrenar.")
        st.stop()
    st.markdown("---")

    # --- PASO 2: SELECCIÓN DEL TIPO DE MODELO ---
    st.header("2. Selección y Configuración del Modelo")
    model_choices = ["Random Forest", "Decision Tree", "XGBoost"]
    st.session_state.selected_model_type = st.selectbox("Elige el tipo de modelo:", model_choices, key="model_type_selector")

    # (Opcional) Hiperparámetros básicos para cada modelo
    # Ejemplo para Random Forest:
    # if st.session_state.selected_model_type == "Random Forest":
    #     n_estimators = st.slider("Número de árboles (n_estimators):", 50, 500, 100, step=10)
    #     max_depth = st.slider("Profundidad máxima (max_depth):", 3, 20, 10, step=1)
    st.markdown("---")

    # --- PASO 3: ENTRENAMIENTO DEL MODELO ---
    st.header("3. Entrenamiento y Evaluación del Modelo")
    if st.button(f"🚀 Entrenar Modelo: {st.session_state.selected_model_type}", key="train_btn"):
        with st.spinner(f"Entrenando {st.session_state.selected_model_type}..."):
            X_selected_df = X_original[st.session_state.features_used_in_model].copy()

            # Identificar tipos de columnas para el ColumnTransformer
            numeric_features = X_selected_df.select_dtypes(include=np.number).columns.tolist()
            categorical_features = X_selected_df.select_dtypes(include=['object', 'category']).columns.tolist()

            # Crear preprocesador
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', MinMaxScaler(), numeric_features), # Escalar numéricas
                    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features) # OHE para categóricas
                ], 
                remainder='passthrough' # Dejar otras columnas (si las hubiera, aunque no debería si seleccionamos bien)
            )

            # Definir el modelo
            model_params = {'random_state': 42} # Parámetros comunes
            if st.session_state.problem_type == "Regression":
                if st.session_state.selected_model_type == "Random Forest":
                    model = RandomForestRegressor(**model_params) # Añadir n_estimators, max_depth si se configuraron
                elif st.session_state.selected_model_type == "Decision Tree":
                    model = DecisionTreeRegressor(**model_params)
                elif st.session_state.selected_model_type == "XGBoost":
                    model = XGBRegressor(**model_params, objective='reg:squarederror') # objective para regresión
            else: # Classification
                if st.session_state.selected_model_type == "Random Forest":
                    model = RandomForestClassifier(**model_params)
                elif st.session_state.selected_model_type == "Decision Tree":
                    model = DecisionTreeClassifier(**model_params)
                elif st.session_state.selected_model_type == "XGBoost":
                    model = XGBClassifier(**model_params, use_label_encoder=False, eval_metric='logloss' if y.nunique()==2 else 'mlogloss')

            # Crear y entrenar el Pipeline
            st.session_state.trained_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            X_train, X_test, y_train, y_test = train_test_split(X_selected_df, y, test_size=0.25, random_state=42, stratify=y if st.session_state.problem_type == "Classification" and y.nunique() > 1 else None)
            
            try:
                st.session_state.trained_pipeline.fit(X_train, y_train)
                y_pred_test = st.session_state.trained_pipeline.predict(X_test)
                
                metrics = {}
                if st.session_state.problem_type == "Regression":
                    metrics['R-squared'] = r2_score(y_test, y_pred_test)
                    metrics['MSE'] = mean_squared_error(y_test, y_pred_test)
                    metrics['RMSE'] = np.sqrt(metrics['MSE'])
                else: # Classification
                    # Si y_test fue codificada y el pipeline predice la clase codificada:
                    metrics['Accuracy'] = accuracy_score(y_test, y_pred_test)
                    metrics['Precision'] = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
                    metrics['Recall'] = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
                    metrics['F1-Score'] = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
                    # Matriz de confusión (necesita clases decodificadas para etiquetas)
                    # cm = confusion_matrix(y_test, y_pred_test) # y_test son las clases codificadas
                    # if st.session_state.label_encoder_target:
                    #     cm_labels = st.session_state.label_encoder_target.classes_
                    #     # Graficar CM
                
                st.session_state.model_performance_metrics = metrics
                st.success(f"Modelo '{st.session_state.selected_model_type}' entrenado y evaluado exitosamente!")
                st.balloons()

            except Exception as e:
                st.error(f"Error durante el entrenamiento o evaluación: {e}")
                st.session_state.trained_pipeline = None
                st.session_state.model_performance_metrics = None
    
    # Mostrar métricas si el modelo está entrenado
    if st.session_state.trained_pipeline and st.session_state.model_performance_metrics:
        st.subheader("Métricas de Desempeño del Modelo (en conjunto de prueba):")
        for metric_name, metric_value in st.session_state.model_performance_metrics.items():
            st.write(f"- **{metric_name}:** {metric_value:.4f}")
        
        # (Opcional) Mostrar matriz de confusión para clasificación
        if st.session_state.problem_type == "Classification":
            y_pred_test_for_cm = st.session_state.trained_pipeline.predict(X_test) # X_test ya está definida arriba
            # y_test_for_cm = y_test # y_test ya está definida arriba y es la codificada
            
            # Decodificar etiquetas para la matriz de confusión si es posible
            y_test_decoded_cm = st.session_state.label_encoder_target.inverse_transform(y_test) if st.session_state.label_encoder_target else y_test
            y_pred_decoded_cm = st.session_state.label_encoder_target.inverse_transform(y_pred_test_for_cm) if st.session_state.label_encoder_target else y_pred_test_for_cm
            
            cm_labels_display = st.session_state.label_encoder_target.classes_ if st.session_state.label_encoder_target else np.unique(np.concatenate((y_test_decoded_cm, y_pred_decoded_cm)))

            cm = confusion_matrix(y_test_decoded_cm, y_pred_decoded_cm, labels=cm_labels_display)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cm_labels_display, yticklabels=cm_labels_display, ax=ax_cm)
            ax_cm.set_xlabel('Predicho')
            ax_cm.set_ylabel('Verdadero')
            ax_cm.set_title('Matriz de Confusión')
            st.pyplot(fig_cm)
            plt.clf()
        st.markdown("---")

    # --- PASO 4: REALIZAR NUEVAS PREDICCIONES ---
    if st.session_state.trained_pipeline:
        st.header("4. Realizar Nuevas Predicciones")
        st.markdown("Ingresa los valores para las características con las que se entrenó el modelo:")

        # Usar las características originales que el usuario seleccionó
        features_for_input = st.session_state.features_used_in_model 
        
        with st.form("new_prediction_form_custom_model"):
            input_data_dict = {}
            form_cols = st.columns(2) # Para organizar el formulario
            
            for i, feature_name in enumerate(features_for_input):
                with form_cols[i % 2]:
                    original_col_series = X_original[feature_name] # Tomar de X_original para rangos y tipo
                    if pd.api.types.is_numeric_dtype(original_col_series):
                        min_val = float(original_col_series.min())
                        max_val = float(original_col_series.max())
                        mean_val = float(original_col_series.mean())
                        # Asegurar que min_val <= mean_val <= max_val
                        if not (min_val <= mean_val <= max_val): mean_val = min_val 
                        input_data_dict[feature_name] = st.number_input(
                            f"{feature_name}", 
                            min_value=min_val, 
                            max_value=max_val, 
                            value=mean_val, 
                            key=f"pred_input_{feature_name}"
                        )
                    elif pd.api.types.is_categorical_dtype(original_col_series) or original_col_series.dtype == 'object':
                        unique_values = original_col_series.unique().tolist()
                        input_data_dict[feature_name] = st.selectbox(
                            f"{feature_name}", 
                            options=unique_values, 
                            index=0, 
                            key=f"pred_input_{feature_name}"
                        )
                    else: # Fallback para tipos raros
                        input_data_dict[feature_name] = st.text_input(f"{feature_name}", key=f"pred_input_{feature_name}")
            
            submit_pred_btn = st.form_submit_button("✨ Obtener Predicción")

        if submit_pred_btn:
            try:
                # Crear DataFrame con los datos de entrada en el orden correcto
                input_df = pd.DataFrame([input_data_dict], columns=features_for_input)
                
                # El pipeline se encargará del preprocesamiento (OHE, escalado)
                prediction_coded = st.session_state.trained_pipeline.predict(input_df)
                prediction_proba = None
                if hasattr(st.session_state.trained_pipeline, "predict_proba"):
                    prediction_proba = st.session_state.trained_pipeline.predict_proba(input_df)

                # Decodificar si es clasificación y tenemos el LabelEncoder
                final_prediction_display = prediction_coded[0]
                if st.session_state.problem_type == "Classification" and st.session_state.label_encoder_target:
                    final_prediction_display = st.session_state.label_encoder_target.inverse_transform(prediction_coded)[0]

                st.success(f"Resultado de la Predicción: **{final_prediction_display}**")

                if prediction_proba is not None and st.session_state.problem_type == "Classification":
                    st.write("Probabilidades por clase:")
                    classes_ = st.session_state.label_encoder_target.classes_ if st.session_state.label_encoder_target else \
                               st.session_state.trained_pipeline.named_steps['model'].classes_ # Fallback
                    proba_df = pd.DataFrame(prediction_proba, columns=classes_)
                    st.dataframe(proba_df)

            except Exception as e:
                st.error(f"Error al realizar la predicción: {e}")
                st.error("Asegúrate de que los datos de entrada sean válidos y compatibles.")
