"""
APLICACIÓN STREAMLIT PARA DESPLIEGUE DEL MODELO DE PRECIOS DE VIVIENDAS
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyod.models.knn import KNN # Asegúrate de que pyod esté en tu requirements.txt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# Configurar la página
st.set_page_config(
    page_title="Analiza tu dataset",
    page_icon="📊",
    layout="wide"
)

# Lista de páginas en orden
PAGES = ["Inicio", "Análisis Exploratorio", "Predicción", "Acerca de"]

# Inicializar estado
if 'data_uploaded' not in st.session_state:
    st.session_state.data_uploaded = False
if 'uploaded_file_content' not in st.session_state: # Para almacenar el contenido del archivo
    st.session_state.uploaded_file_content = None
if 'df' not in st.session_state: # DataFrame principal
    st.session_state.df = None
if 'page_index' not in st.session_state:
    st.session_state.page_index = 0
if 'reset_app' not in st.session_state:
    st.session_state.reset_app = False

# Función para reiniciar la app
def limpiar_app():
    st.session_state.reset_app = True

# Ejecutar reinicio si se marcó
if st.session_state.reset_app:
    # Guardar el índice de página actual si queremos volver a la misma página después de limpiar
    # current_page_index = st.session_state.page_index
    
    keys_to_reset = ['data_uploaded', 'uploaded_file_content', 'df', 
                     'limpieza_aplicada', 'df_limpio']
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    
    st.session_state.data_uploaded = False
    st.session_state.uploaded_file_content = None
    st.session_state.df = None
    st.session_state.page_index = 0 # Siempre volver al inicio después de limpiar
    st.session_state.limpieza_aplicada = False
    st.session_state.df_limpio = None
    st.session_state.reset_app = False
    st.rerun()

# Subida de archivo si no hay datos
if not st.session_state.data_uploaded:
    st.title("🔎 Analiza tu dataset")
    st.markdown("### Por favor, sube un archivo CSV para comenzar.")
    uploaded_file = st.file_uploader("Subir archivo de datos", type=["csv"], key="file_uploader_main")
    
    if uploaded_file is not None:
        st.session_state.uploaded_file_content = uploaded_file # Guardar el objeto archivo
        st.session_state.data_uploaded = True
        st.session_state.df = pd.read_csv(uploaded_file)
        st.rerun()
    st.stop()

# Si los datos están cargados, df se toma de session_state
df = st.session_state.df

if df is None: # Doble chequeo por si acaso
    st.error("Error al cargar el DataFrame. Por favor, intenta subir el archivo de nuevo.")
    st.button("Reiniciar Aplicación", on_click=limpiar_app)
    st.stop()

@st.cache_resource
def load_model_from_path(model_path='models/housing_model.pkl'): # Renombrado para evitar conflicto si hay otra función load_model
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        # No mostrar error aquí directamente, se manejará en la página de predicción
        return None
    except Exception as e:
        # st.error(f"Error al cargar el modelo: {e}")
        return None

# Cargar modelo (se usará en la página de predicción)
model = load_model_from_path() 

# Scaler se inicializa aquí pero se ajusta (fit) en la página de limpieza o predicción si es necesario
scaler = MinMaxScaler()

if 'limpieza_aplicada' not in st.session_state:
    st.session_state.limpieza_aplicada = False
if 'df_limpio' not in st.session_state:
    st.session_state.df_limpio = None

# Insertar "Limpieza de Datos" al flujo si no existe
if "Limpieza de Datos" not in PAGES:
    PAGES.insert(1, "Limpieza de Datos") # Insertar después de "Inicio"


# Navegación por botones (Ahora en la sidebar para más espacio en contenido)
st.sidebar.title("Navegación")
current_page_name = PAGES[st.session_state.page_index]
selected_page = st.sidebar.radio("Ir a:", PAGES, index=st.session_state.page_index, key="nav_radio")

if selected_page != current_page_name:
    st.session_state.page_index = PAGES.index(selected_page)
    st.rerun()

page = PAGES[st.session_state.page_index] # Definir la página actual

# Botón para limpiar app en la sidebar
st.sidebar.button("🧹 Limpiar y Reiniciar Aplicación", on_click=limpiar_app, type="primary", use_container_width=True)


# --- CONTENIDO POR PÁGINA ---

if page == "Inicio":
    st.title("🔎 Analiza tu Dataset: Página de Inicio")
    st.markdown("Hola! Aquí obtendrás una visión general de tus datos antes de sumergirnos en el análisis detallado y la limpieza.")
    st.markdown("---")

    # 1. Resumen General del Dataset
    st.header("1. Resumen General del Dataset")

    if df is not None and not df.empty:
        col_summary1, col_summary2 = st.columns(2)

        with col_summary1:
            st.subheader("🔢 Dimensiones")
            st.write(f"**Número de Filas:** {df.shape[0]}")
            st.write(f"**Número de Columnas:** {df.shape[1]}")

            st.subheader("📜 Tipos de Datos por Columna")
            df_types = df.dtypes.astype(str).reset_index() # Convertir tipos a string para mostrar
            df_types.columns = ['Columna', 'Tipo de Dato']
            st.dataframe(df_types, use_container_width=True, height=300)

        with col_summary2:
            st.subheader("❓ Valores Faltantes")
            missing_values = df.isnull().sum().reset_index()
            missing_values.columns = ['Columna', 'Valores Faltantes']
            missing_values = missing_values[missing_values['Valores Faltantes'] > 0] 

            if not missing_values.empty:
                st.dataframe(missing_values, use_container_width=True, height=300)
            else:
                st.success("✅ ¡Excelente! No se encontraron valores faltantes en tu dataset.")

        with st.expander("📊 Ver Estadísticas Descriptivas (Columnas Numéricas)"):
            numeric_cols = df.select_dtypes(include=np.number)
            if not numeric_cols.empty:
                st.dataframe(numeric_cols.describe().T)
            else:
                st.info("No hay columnas numéricas para mostrar estadísticas descriptivas.")
        
        with st.expander("📝 Ver Estadísticas Descriptivas (Columnas Categóricas/Objeto)"):
            categorical_cols = df.select_dtypes(include=['object', 'category'])
            if not categorical_cols.empty:
                st.dataframe(categorical_cols.describe().T)
            else:
                st.info("No hay columnas categóricas/objeto para mostrar estadísticas descriptivas.")
        st.markdown("---")
    else:
        st.warning("El DataFrame está vacío o no se ha cargado correctamente.")


    # 2. Vista Previa del Dataset
    st.header("2. Vista Previa del Dataset")
    if df is not None and not df.empty:
        num_rows_preview = st.slider(
            "Selecciona el número de filas para previsualizar:",
            min_value=1,
            max_value=min(20, df.shape[0]), 
            value=min(5, df.shape[0]), # Valor inicial no puede ser mayor que max_value
            key="preview_slider_inicio" 
        )
        st.dataframe(df.head(num_rows_preview), use_container_width=True)
        st.markdown("---")

    # 3. Breve resumen de lo que sigue en la siguiente página
    st.header("3. Próximos Pasos: Limpieza de Datos 🧼")
    st.info("""
    En la **siguiente página ("Limpieza de Datos")**, nos adentraremos en el proceso de **Limpieza de Datos**.
    Este es un paso fundamental para asegurar la calidad y fiabilidad de tus análisis y modelos predictivos. Cubriremos:
    - Estrategias para manejar valores faltantes.
    - Identificación y tratamiento de filas duplicadas (si aplica).
    - Detección y manejo de outliers (valores atípicos).
    - Verificación y posible conversión de tipos de datos.

    ¡Prepárate para refinar tu dataset y dejarlo listo para el análisis profundo!
    """)
    st.markdown("---")


elif page == "Limpieza de Datos":
    st.header("🧹 Limpieza de Datos")
    st.markdown("""
    En esta sección puedes aplicar una limpieza básica al conjunto de datos. 
    Los cambios aquí afectarán las páginas subsiguientes de análisis y predicción.
    """)

    if st.session_state.limpieza_aplicada and st.session_state.df_limpio is not None:
        st.success("🎉 ¡Los datos ya fueron procesados con los parámetros de limpieza anteriores!")
        st.markdown("Si deseas aplicar una nueva limpieza, los datos originales se usarán como base.")
        if st.button("Restaurar datos originales para nueva limpieza"):
            st.session_state.limpieza_aplicada = False
            st.session_state.df_limpio = None
            st.rerun()

    # Usar df original para la limpieza
    df_para_limpiar = df.copy() 
    
    st.subheader("Opciones de Limpieza")
    with st.form("form_limpieza"):
        eliminar_nulos = st.checkbox("Eliminar filas con valores nulos (NaN)", value=True)
        
        columnas_numericas_limpieza = df_para_limpiar.select_dtypes(include=np.number).columns.tolist()
        if not columnas_numericas_limpieza:
            st.warning("No hay columnas numéricas en el dataset para la detección de outliers.")
            porcentaje_outliers = 0.0
        else:
            porcentaje_outliers = st.slider("Porcentaje de outliers a identificar y eliminar (usando KNN)", 0.0, 0.25, 0.05, step=0.01,
                                            help="Esto se aplica a las columnas numéricas. Un valor de 0.05 significa que se considera outlier al 5% de los datos más anómalos.")
        
        aplicar_limpieza = st.form_submit_button("🚀 Aplicar Limpieza")

    if aplicar_limpieza:
        with st.spinner("Aplicando limpieza... Por favor espera."):
            df_resultado_limpieza = df_para_limpiar.copy()

            if eliminar_nulos:
                nulos_antes = df_resultado_limpieza.isnull().sum().sum()
                df_resultado_limpieza.dropna(inplace=True)
                nulos_despues = df_resultado_limpieza.isnull().sum().sum()
                st.write(f"Se eliminaron {nulos_antes - nulos_despues} valores nulos.")
                if df_resultado_limpieza.empty:
                    st.error("El dataset quedó vacío después de eliminar nulos. Revisa tus datos o los parámetros.")
                    st.stop()


            if porcentaje_outliers > 0 and not df_resultado_limpieza.empty:
                columnas_numericas_actuales = df_resultado_limpieza.select_dtypes(include=np.number).columns
                if not columnas_numericas_actuales.empty:
                    # Escalar solo para la detección de outliers, luego usar el df original sin escalar pero filtrado
                    df_scaled_for_outliers = df_resultado_limpieza.copy()
                    df_scaled_for_outliers[columnas_numericas_actuales] = scaler.fit_transform(df_scaled_for_outliers[columnas_numericas_actuales])
                    
                    try:
                        modelo_outliers = KNN(contamination=porcentaje_outliers)
                        modelo_outliers.fit(df_scaled_for_outliers[columnas_numericas_actuales])
                        etiquetas_outliers = modelo_outliers.labels_
                        
                        original_rows = len(df_resultado_limpieza)
                        df_resultado_limpieza = df_resultado_limpieza[etiquetas_outliers == 0]
                        outliers_eliminados = original_rows - len(df_resultado_limpieza)
                        st.write(f"Se identificaron y eliminaron {outliers_eliminados} filas como outliers.")
                        if df_resultado_limpieza.empty:
                            st.error("El dataset quedó vacío después de eliminar outliers. Considera un porcentaje menor.")
                            st.stop()
                    except Exception as e:
                        st.error(f"Error durante la detección de outliers: {e}. Prueba con un porcentaje menor o revisa tus datos.")

            st.session_state.df_limpio = df_resultado_limpieza.reset_index(drop=True)
            st.session_state.limpieza_aplicada = True
        st.success("✅ Limpieza procesada.")
        st.rerun() # Rerun para mostrar los datos limpios

    if st.session_state.limpieza_aplicada and st.session_state.df_limpio is not None:
        st.subheader("Comparación de Datos")
        col_orig, col_limp = st.columns(2)
        with col_orig:
            st.markdown("**Datos Originales (Primeras filas)**")
            st.write(f"Dimensiones: {df.shape}")
            st.dataframe(df.head(), height=200)
        with col_limp:
            st.markdown("**Datos Limpios (Primeras filas)**")
            st.write(f"Dimensiones: {st.session_state.df_limpio.shape}")
            st.dataframe(st.session_state.df_limpio.head(), height=200)
        
        if st.session_state.df_limpio.equals(df):
             st.info("Los datos limpios son idénticos a los originales (no se aplicaron cambios o los cambios no alteraron el df).")


elif page == "Análisis Exploratorio":
    st.header("📊 Análisis Exploratorio de Datos")
    
    # Decidir qué DataFrame usar: el limpio si está disponible, sino el original
    df_eda = st.session_state.df_limpio if st.session_state.limpieza_aplicada and st.session_state.df_limpio is not None else df
    
    if st.session_state.limpieza_aplicada and st.session_state.df_limpio is not None:
        st.info("Mostrando análisis sobre los **datos limpios**.")
    else:
        st.info("Mostrando análisis sobre los **datos originales**. Puedes aplicar la limpieza en la página 'Limpieza de Datos'.")

    if df_eda.empty:
        st.warning("El dataset para análisis está vacío. Verifica los pasos anteriores.")
        st.stop()

    st.markdown("Visualizaciones para entender relaciones entre variables.")
    
    # Matriz de correlación
    st.subheader("Matriz de Correlación (Columnas Numéricas)")
    numeric_cols_eda = df_eda.select_dtypes(include=np.number)
    if len(numeric_cols_eda.columns) > 1:
        corr = numeric_cols_eda.corr()
        fig, ax = plt.subplots(figsize=(12, 10)) # Aumentado tamaño
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0, # Ajustado vmax/vmin
                    square=True, linewidths=.5, annot=True, fmt='.2f', annot_kws={"size": 8}) # Tamaño de annot
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No hay suficientes columnas numéricas para generar una matriz de correlación.")

    # Relaciones con el Precio (si 'PRICE' existe)
    if 'PRICE' in df_eda.columns and pd.api.types.is_numeric_dtype(df_eda['PRICE']):
        st.subheader("Relaciones con la variable 'PRICE'")
        numeric_cols_for_scatter = df_eda.select_dtypes(include=np.number).columns.drop('PRICE', errors='ignore')
        
        if len(numeric_cols_for_scatter) > 0:
            selected_cols_scatter = st.multiselect(
                "Selecciona columnas numéricas para comparar con 'PRICE':",
                options=numeric_cols_for_scatter.tolist(),
                default=numeric_cols_for_scatter.tolist()[:min(4, len(numeric_cols_for_scatter))] # Default primeras 4 o menos
            )

            if selected_cols_scatter:
                num_plots = len(selected_cols_scatter)
                cols_per_row = 2
                num_rows = (num_plots + cols_per_row - 1) // cols_per_row

                fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(7 * cols_per_row, 5 * num_rows))
                axes = axes.flatten() # Asegurar que axes sea siempre iterable

                for i, col_name in enumerate(selected_cols_scatter):
                    if i < len(axes): # Para no exceder el número de subplots
                        sns.scatterplot(data=df_eda, x=col_name, y='PRICE', ax=axes[i])
                        axes[i].set_title(f'{col_name} vs PRICE')
                        axes[i].set_xlabel(col_name)
                        axes[i].set_ylabel('PRICE')
                
                # Ocultar ejes no usados si el número de plots no es par
                for j in range(num_plots, len(axes)):
                    fig.delaxes(axes[j])

                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Selecciona al menos una columna para visualizar su relación con 'PRICE'.")
        else:
            st.info("No hay otras columnas numéricas para comparar con 'PRICE'.")
    else:
        st.info("La columna 'PRICE' no existe o no es numérica en el dataset actual.")
    
    # Distribución (general)
    st.subheader("Distribución de Variables Numéricas")
    numeric_cols_dist = df_eda.select_dtypes(include=np.number).columns
    if len(numeric_cols_dist) > 0:
        selected_col_dist = st.selectbox(
            "Selecciona una variable numérica para ver su distribución:",
            options=numeric_cols_dist.tolist(),
            index = numeric_cols_dist.get_loc('PRICE') if 'PRICE' in numeric_cols_dist else 0
        )
        if selected_col_dist:
            fig_dist, ax_dist = plt.subplots(figsize=(8,5))
            sns.histplot(df_eda[selected_col_dist], kde=True, ax=ax_dist)
            ax_dist.set_title(f'Distribución de {selected_col_dist}')
            st.pyplot(fig_dist)
    else:
        st.info("No hay columnas numéricas para mostrar distribuciones.")

    # Exploración interactiva
    st.subheader("Exploración Interactiva (Scatter Plot)")
    if len(df_eda.columns) > 1:
        col1_exp, col2_exp = st.columns(2)
        with col1_exp:
            x_var = st.selectbox("Variable X:", options=df_eda.columns.tolist(), index=0, key="x_var_eda")
        with col2_exp:
            # Intentar que Y no sea igual a X por defecto
            y_var_default_index = 1 if len(df_eda.columns) > 1 else 0
            if df_eda.columns[y_var_default_index] == x_var and len(df_eda.columns) > 2:
                 y_var_default_index = 2
            elif df_eda.columns[y_var_default_index] == x_var and len(df_eda.columns) <= 1 : # Si solo hay una columna o X e Y son la misma y solo hay 2
                 y_var_default_index = 0


            y_var = st.selectbox("Variable Y:", options=df_eda.columns.tolist(), index=y_var_default_index, key="y_var_eda")
        
        if x_var and y_var:
            fig_scatter_interactive, ax_scatter_interactive = plt.subplots(figsize=(8,6))
            sns.scatterplot(data=df_eda, x=x_var, y=y_var, ax=ax_scatter_interactive)
            ax_scatter_interactive.set_title(f'{x_var} vs {y_var}')
            st.pyplot(fig_scatter_interactive)
    else:
        st.info("Se necesitan al menos dos columnas para la exploración interactiva.")


elif page == "Predicción":
    st.header("🤖 Predicción de Precios de Viviendas")
    
    df_pred = st.session_state.df_limpio if st.session_state.limpieza_aplicada and st.session_state.df_limpio is not None else df
    
    if st.session_state.limpieza_aplicada and st.session_state.df_limpio is not None:
        st.info("La predicción se basará en un modelo entrenado y podría usar características de los **datos limpios** si el modelo fue entrenado así.")
    else:
        st.info("La predicción se basará en un modelo entrenado. Considera limpiar los datos primero para mejores resultados si el modelo espera datos limpios.")

    if model is None:
        st.error("El archivo del modelo ('models/housing_model.pkl') no fue encontrado o no se pudo cargar. Por favor, asegúrate de que el archivo exista en la carpeta 'models/'.")
        st.markdown("La funcionalidad de predicción no está disponible.")
        st.stop()
        
    st.markdown("Ingresa las características de una vivienda para obtener una predicción de precio.")
    
    # Características esperadas por el modelo (ejemplo Boston Housing)
    # RM, LSTAT, PTRATIO, DIS (Estas deben coincidir con las que usó tu modelo)
    # Es CRUCIAL que estas columnas sean las mismas y en el mismo orden que las que usó tu modelo para entrenar.
    # También, el scaler debe ser ajustado (fitted) con los datos de entrenamiento originales.
    # Aquí asumimos que el scaler global 'scaler' fue ajustado previamente o se ajustará con los datos correctos.
    # Para este ejemplo, si el modelo usa estas 4, y df_pred tiene esas columnas:
    
    expected_features = ['RM', 'LSTAT', 'PTRATIO', 'DIS'] # AJUSTA ESTO A LAS FEATURES DE TU MODELO
    
    missing_model_features = [f for f in expected_features if f not in df_pred.columns]
    if missing_model_features:
        st.warning(f"Las siguientes columnas requeridas por el modelo no se encuentran en el dataset cargado/limpio: {', '.join(missing_model_features)}. La predicción podría no ser precisa o fallar.")
        # Podrías detenerte aquí o permitir continuar bajo advertencia.

    with st.form("prediction_form"):
        st.write("Por favor, ingresa los valores para las siguientes características:")
        
        input_values = {}
        cols_form = st.columns(2)

        # Crear inputs dinámicamente para las características esperadas
        for i, feature in enumerate(expected_features):
            with cols_form[i % 2]: # Alternar entre columnas
                if feature in df_pred.columns and pd.api.types.is_numeric_dtype(df_pred[feature]):
                    min_val = float(df_pred[feature].min())
                    max_val = float(df_pred[feature].max())
                    mean_val = float(df_pred[feature].mean())
                    if feature == 'RM': # Ejemplo de selectbox para RM
                         # Asegurar que rm_min sea menor que rm_max
                        rm_min_val = int(np.floor(min_val))
                        rm_max_val = int(np.ceil(max_val))
                        if rm_min_val >= rm_max_val : rm_max_val = rm_min_val +1 # simple fix
                        
                        default_rm_index = int(np.clip(int(mean_val) - rm_min_val, 0, rm_max_val - rm_min_val))
                        input_values[feature] = st.selectbox(
                            f"{feature} (Número de habitaciones):", 
                            list(range(rm_min_val, rm_max_val + 1)), 
                            index=default_rm_index
                        )
                    else: # Slider para otras numéricas
                        input_values[feature] = st.slider(
                            f"{feature}:", 
                            min_value=min_val, 
                            max_value=max_val, 
                            value=mean_val,
                            step = 0.01 if (max_val - min_val) < 10 else 0.1 if (max_val - min_val) < 100 else 1.0 # dynamic step
                        )
                else:
                    # Si la columna no existe o no es numérica, pedir un input de texto
                    input_values[feature] = st.number_input(f"{feature} (valor numérico):", value=0.0, format="%.2f")
                    if feature not in df_pred.columns:
                        st.caption(f"Advertencia: '{feature}' no está en el dataset cargado.")
                    elif not pd.api.types.is_numeric_dtype(df_pred[feature]):
                        st.caption(f"Advertencia: '{feature}' no es una columna numérica en el dataset.")


        submit_button = st.form_submit_button("🏷️ Predecir Precio")

    if submit_button:
        try:
            # Crear DataFrame con los inputs en el orden correcto
            input_data_list = [input_values[feature] for feature in expected_features]
            input_df = pd.DataFrame([input_data_list], columns=expected_features)

            # Escalar los datos de entrada
            # IMPORTANTE: El scaler DEBE ser el mismo que se usó para entrenar el modelo
            # y debe ser ajustado (fit) con los datos de entrenamiento ORIGINALES.
            # Si tu 'scaler' global no fue fiteado con los datos correctos, la predicción será incorrecta.
            # Aquí se asume que el scaler fue fiteado con las columnas 'expected_features' del dataset de entrenamiento.
            
            # Ejemplo de cómo podrías haber fiteado el scaler (esto debería hacerse UNA VEZ, no en cada predicción)
            # if 'scaler_fitted' not in st.session_state:
            #     training_data_for_scaler = pd.read_csv('ruta_a_tus_datos_de_entrenamiento.csv') # Cargar datos de entrenamiento
            #     scaler.fit(training_data_for_scaler[expected_features])
            #     st.session_state.scaler_fitted = True
            
            # Por ahora, asumimos que el scaler global ya está fiteado correctamente.
            # Si el scaler no ha sido fiteado con estas columnas específicas, dará error o resultados incorrectos.
            # Una mejor práctica es guardar el scaler fiteado junto con el modelo.
            try:
                input_scaled = scaler.transform(input_df) # MinMaxScaler espera un DataFrame
            except NotFittedError: # sklearn.exceptions.NotFittedError
                 st.error("El Escalador (Scaler) no ha sido ajustado (fitted). Esto es un problema de configuración del desarrollador. Se necesita ajustar el scaler con los datos de entrenamiento.")
                 st.stop()
            except ValueError as ve: # Si las features no coinciden con las del fit
                st.error(f"Error al escalar los datos de entrada: {ve}. Asegúrate de que las características de entrada coincidan con las que se usó para ajustar el escalador.")
                st.stop()


            prediction = model.predict(input_scaled)[0]
            st.success(f"💰 El precio predicho para la vivienda es: **${prediction:,.2f}k**") # k para miles
            
            # Podrías añadir más detalles o interpretación aquí
            # st.subheader("Interpretación de la predicción")
            # st.write("Basado en las características proporcionadas...")

        except Exception as e:
            st.error(f"Ocurrió un error durante la predicción: {e}")
            st.warning("Asegúrate de que el modelo y el escalador estén configurados correctamente y que los datos de entrada sean válidos.")


elif page == "Acerca de":
    st.header("ℹ️ Acerca de Esta Aplicación")
    st.markdown("""
    Esta aplicación fue desarrollada para demostrar un flujo interactivo de análisis de datos y predicción 
    utilizando un modelo de Machine Learning, específicamente para la estimación de precios de viviendas.

    **Características Clave:**
    - Carga y visualización de datasets en formato CSV.
    - Resumen estadístico y vista previa de los datos.
    - Herramientas básicas de limpieza de datos (manejo de nulos y outliers).
    - Análisis exploratorio de datos (EDA) con visualizaciones interactivas.
    - Predicción de precios utilizando un modelo pre-entrenado.

    **Tecnologías Utilizadas:**
    - **Streamlit:** Para la creación de la interfaz web interactiva.
    - **Pandas:** Para la manipulación y análisis de datos.
    - **Scikit-learn:** Para el preprocesamiento (escalado) y el modelo de Machine Learning.
    - **Matplotlib & Seaborn:** Para la generación de gráficos y visualizaciones.
    - **PyOD:** Para la detección de outliers (KNN).
    - **Joblib:** Para la carga del modelo serializado.

    **Autor/Desarrollador:**
    * [Tu Nombre o Nombre del Equipo Aquí]

    **Versión:**
    * 1.0.0 (Mayo 2025)
    
    **Consideraciones:**
    * El modelo de predicción (`housing_model.pkl`) y el escalador deben estar correctamente configurados y ser compatibles con los datos de entrada.
    * La precisión de las predicciones depende de la calidad del modelo y de los datos con los que fue entrenado.
    """)
    st.markdown("---")
    st.markdown("Gracias por usar esta aplicación.")