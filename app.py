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

# Configurar la página
st.set_page_config(
    page_title="Analiza tu dataset",
    page_icon="📊",
    layout="wide"
)

# Lista de páginas en orden
PAGES = ["Inicio", "Limpieza de Datos", "Análisis Exploratorio", "Predicción", "Acerca de"]

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
            # ... (código existente) ...
            numeric_cols = df.select_dtypes(include=np.number)
            if not numeric_cols.empty:
                st.dataframe(numeric_cols.describe().T)
            else:
                st.info("No hay columnas numéricas para mostrar estadísticas descriptivas.")
        with st.expander("📝 Ver Estadísticas Descriptivas (Columnas Categóricas/Objeto)"):
            # ... (código existente) ...
            categorical_cols = df.select_dtypes(include=['object', 'category'])
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
    st.header("3. Definir Variable Objetivo (Opcional)")
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
    st.markdown("---")
    

    # 5. Próximos Pasos (Numeración actualizada)
    st.header(f"5. Próximos Pasos: {PAGES[1]} 🧼")
    st.info(f"""
    En la **siguiente página ("{PAGES[1]}")**, nos adentraremos en el proceso de limpieza.
    Este es un paso fundamental para asegurar la calidad y fiabilidad de tus análisis y modelos predictivos.
    """)


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
                        df_resultado_limpieza = df_resultado_limpieza[etiquetas_outliers == 0]
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
    # ... (código existente de matriz de correlación, sin cambios) ...
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

    st.subheader("Exploración Interactiva (Scatter Plot Bivariado)")
    # ... (código existente de exploración interactiva, sin cambios importantes) ...
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
            hue_var = st.selectbox("Variable para color (Hue, opcional):", options=hue_var_options, index=0, key="hue_var_eda_general")
            try:
                sns.scatterplot(data=df_eda, x=x_var, y=y_var, hue=hue_var if hue_var else None, ax=ax_sc_int, palette="magma", alpha=0.7)
                ax_sc_int.set_title(f'{x_var} vs {y_var}' + (f' (Color por {hue_var})' if hue_var else ''))
                if hue_var: ax_sc_int.legend(title=hue_var, bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout(); st.pyplot(fig_sc_int); plt.clf()
            except Exception as e:
                st.error(f"No se pudo generar el gráfico: {e}")


elif page == "Predicción":
    st.title(f"🤖 {page}: Estimación de Precios") 
    # ... (resto del código de Predicción sin cambios directos por target_variable,
    # ya que la predicción depende de las features que espera el modelo) ...
    df_for_pred_ranges = st.session_state.df_limpio if st.session_state.limpieza_aplicada and st.session_state.df_limpio is not None else df
    if model is None:
        st.error("El archivo del modelo ('models/housing_model.pkl') no se pudo cargar...")
        st.stop()
    st.markdown("Ingresa las características de una vivienda para obtener una estimación de precio.")
    expected_features_model = ['RM', 'LSTAT', 'PTRATIO', 'DIS'] 
    st.info(f"El modelo actual espera las siguientes características en este orden: **{', '.join(expected_features_model)}**.")

    if not st.session_state.scaler_fitted:
        # ... (lógica de ajuste del scaler) ...
        st.warning("El escalador (Scaler) se ajustará ahora...")
        numeric_features_in_df = [f for f in expected_features_model if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
        if not numeric_features_in_df:
            st.error(f"Ninguna de las características esperadas por el modelo ({', '.join(expected_features_model)}) "
                       "es numérica o se encuentra en el dataset cargado...")
            st.stop()
        if len(numeric_features_in_df) < len(expected_features_model):
             st.warning(f"Algunas características esperadas por el modelo no son numéricas o no están en el dataset...")
        try:
            st.session_state.fitted_scaler_instance.fit(df[numeric_features_in_df])
            st.session_state.scaler_fitted = True
            st.success("Escalador ajustado.")
        except Exception as e:
            st.error(f"Error al ajustar el escalador: {e}.")
            st.stop()
    
    with st.form("prediction_form"):
        # ... (formulario de predicción) ...
        st.write("Por favor, ingresa los valores para las características requeridas por el modelo:")
        input_values = {}
        cols_form = st.columns(2)
        for i, feature in enumerate(expected_features_model):
            with cols_form[i % 2]: 
                if feature in df_for_pred_ranges.columns and pd.api.types.is_numeric_dtype(df_for_pred_ranges[feature]):
                    min_val = float(df_for_pred_ranges[feature].min())
                    max_val = float(df_for_pred_ranges[feature].max())
                    mean_val = float(df_for_pred_ranges[feature].mean())
                    step_val = 0.01 if (max_val - min_val) < 10 else 0.1 # ...
                    if feature == 'RM' and min_val.is_integer() and max_val.is_integer():
                         # ... (selectbox RM) ...
                         rm_min_val = int(min_val); rm_max_val = int(max_val)
                         if rm_min_val >= rm_max_val: rm_max_val = rm_min_val + 1 
                         default_rm_index = np.clip(int(mean_val) - rm_min_val, 0, rm_max_val - rm_min_val) if rm_max_val > rm_min_val else 0
                         input_values[feature] = st.selectbox(f"{feature}:", list(range(rm_min_val, rm_max_val + 1)), index=default_rm_index)
                    else: 
                        input_values[feature] = st.slider(f"{feature}:", min_val, max_val, value=mean_val, step=step_val)
                else:
                    input_values[feature] = st.number_input(f"{feature} (valor numérico):", value=0.0, format="%.2f",
                                                             help=f"'{feature}' no se encontró como numérica...")
        submit_button = st.form_submit_button("🏷️ Predecir Precio")

    if submit_button:
        # ... (lógica de predicción) ...
        if not st.session_state.scaler_fitted:
            st.error("El escalador no está ajustado...")
        else:
            try:
                input_data_list = [input_values[feature] for feature in expected_features_model]
                input_df_for_scaling = pd.DataFrame([input_data_list], columns=expected_features_model)
                numeric_features_scaler_was_fitted_with = st.session_state.fitted_scaler_instance.feature_names_in_.tolist() \
                    if hasattr(st.session_state.fitted_scaler_instance, 'feature_names_in_') else \
                    df[expected_features_model].select_dtypes(include=np.number).columns.tolist()
                input_df_to_scale_actual = input_df_for_scaling[numeric_features_scaler_was_fitted_with]
                input_scaled_values = st.session_state.fitted_scaler_instance.transform(input_df_to_scale_actual)
                prediction = model.predict(input_scaled_values)[0]
                st.success(f"💰 El precio estimado para la vivienda es: **${prediction:,.2f}k**")
            except NotFittedError:
                 st.error("El Escalador (Scaler) no ha sido ajustado (fitted) correctamente...")
            except ValueError as ve:
                st.error(f"Error al escalar o predecir: {ve}...")
            except Exception as e:
                st.error(f"Ocurrió un error durante la predicción: {e}")


elif page == "Acerca de":
    st.title(f"ℹ️ {page} Esta Aplicación") 
    # ... (código existente) ...
    st.markdown("""
    Esta aplicación fue desarrollada para demostrar un flujo interactivo de análisis de datos y predicción...
    **Autor/Desarrollador:** Adaptado y mejorado por Asistente AI. ¡Personaliza esta sección!
    **Versión:** 1.2.0 (Mayo 2025) 
    **Consideraciones Importantes:** ...
    """)
    st.markdown("---")
    st.markdown(f"Fecha y Hora Actual: {pd.Timestamp.now(tz='America/Bogota').strftime('%Y-%m-%d %H:%M:%S %Z')}")