"""
EDA Interactivo - Dataset Agrícola (Fincas / Cultivos)
Autor: Generado con Claude
Ejecutar con: streamlit run main_app.py
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# --------------------------------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="EDA - Dataset Agrícola",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Columnas esperadas del dataset (se usan como ayuda para clasificar tipos,
# pero la app también funciona si difieren ligeramente)
COL_ID = "ID_Finca"
COL_DEPTO = "Departamento"
COL_CULTIVO = "Tipo_Cultivo"
COL_AREA = "Area_Hectareas"
COL_PROD = "Produccion_Anual_Ton"
COL_RIEGO = "Sistema_Riego_Tecnificado"
COL_TECNIF = "Nivel_Tecnificacion"
COL_PRECIO = "Precio_Venta_Por_Ton_COP"
COL_SUELO = "Tipo_Suelo"
COL_FECHA = "Fecha_Ultima_Auditoria"

EXPECTED_NUMERIC = [COL_AREA, COL_PROD, COL_PRECIO]
EXPECTED_CATEGORICAL = [COL_DEPTO, COL_CULTIVO, COL_RIEGO, COL_TECNIF, COL_SUELO]
EXPECTED_DATE = [COL_FECHA]
EXPECTED_ID = [COL_ID]
EXPECTED_COLUMNS = EXPECTED_ID + EXPECTED_CATEGORICAL[:2] + [COL_AREA, COL_PROD, COL_RIEGO, COL_TECNIF, COL_PRECIO, COL_SUELO, COL_FECHA]


@st.cache_data(show_spinner=False)
def generate_sample_data(n=150, seed=42):
    """Genera un dataset sintético de fincas para probar la app sin necesidad de un CSV."""
    rng = np.random.default_rng(seed)
    departamentos = ["Antioquia", "Valle del Cauca", "Cundinamarca", "Tolima", "Huila", "Santander"]
    cultivos = ["Café", "Cacao", "Aguacate", "Plátano", "Caña de azúcar", "Maíz"]
    suelos = ["Franco", "Arcilloso", "Arenoso", "Limoso"]
    niveles = ["Bajo", "Medio", "Alto"]

    inicio = pd.Timestamp("2023-01-01")
    fin = pd.Timestamp("2025-12-31")
    dias_totales = (fin - inicio).days

    df = pd.DataFrame({
        COL_ID: [f"F-{i:04d}" for i in range(1, n + 1)],
        COL_DEPTO: rng.choice(departamentos, n),
        COL_CULTIVO: rng.choice(cultivos, n),
        COL_AREA: np.round(rng.gamma(shape=3, scale=5, size=n), 2),
        COL_RIEGO: rng.choice(["Sí", "No"], n, p=[0.4, 0.6]),
        COL_TECNIF: rng.choice(niveles, n, p=[0.3, 0.45, 0.25]),
        COL_SUELO: rng.choice(suelos, n),
        COL_FECHA: [inicio + pd.Timedelta(days=int(d)) for d in rng.integers(0, dias_totales, n)],
    })
    tecnif_factor = df[COL_TECNIF].map({"Bajo": 0.8, "Medio": 1.0, "Alto": 1.3})
    df[COL_PROD] = np.round(df[COL_AREA] * rng.normal(3.5, 0.6, n) * tecnif_factor, 2)
    df[COL_PRECIO] = np.round(rng.normal(1_800_000, 250_000, n), -3)
    return df


# --------------------------------------------------------------------------------------
# UTILIDADES
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(file, sep, decimal, encoding):
    return pd.read_csv(file, sep=sep, decimal=decimal, encoding=encoding)


def infer_column_types(df: pd.DataFrame):
    """Clasifica columnas en numéricas, categóricas, fecha e identificador."""
    numeric_cols, categorical_cols, date_cols, id_cols = [], [], [], []

    for col in df.columns:
        if col in EXPECTED_ID:
            id_cols.append(col)
            continue
        if col in EXPECTED_DATE:
            date_cols.append(col)
            continue
        if col in EXPECTED_NUMERIC:
            numeric_cols.append(col)
            continue
        if col in EXPECTED_CATEGORICAL:
            categorical_cols.append(col)
            continue

        # Fallback: inferencia automática para columnas no reconocidas
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        elif "fecha" in col.lower() or "date" in col.lower():
            date_cols.append(col)
        elif df[col].nunique(dropna=True) <= max(20, int(len(df) * 0.05)):
            categorical_cols.append(col)
        else:
            id_cols.append(col)

    return numeric_cols, categorical_cols, date_cols, id_cols


def coerce_types(df, numeric_cols, date_cols):
    df = df.copy()
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
    return df


def kpi_card(label, value, help_text=None):
    st.metric(label, value, help=help_text)


def download_df_button(df, filename, label):
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button(label, csv_buf.getvalue(), file_name=filename, mime="text/csv")


# --------------------------------------------------------------------------------------
# CARGA DE DATOS
# --------------------------------------------------------------------------------------
st.title("🌾 EDA Interactivo - Dataset Agrícola de Fincas")
st.caption(
    "Análisis exploratorio cuantitativo, cualitativo y gráfico de un dataset de "
    "fincas, cultivos, producción y tecnificación."
)

with st.sidebar:
    st.header("⚙️ Carga de datos")
    data_source = st.radio(
        "Fuente de datos",
        ["📂 Subir archivo CSV", "🧪 Usar datos de ejemplo"],
        index=0,
        help="Elige 'Usar datos de ejemplo' si quieres explorar la app sin tener un archivo a la mano.",
    )

    uploaded_file = None
    sep, decimal, encoding = ",", ".", "utf-8"

    if data_source == "📂 Subir archivo CSV":
        uploaded_file = st.file_uploader(
            "Arrastra tu archivo aquí o haz clic para seleccionarlo",
            type=["csv"],
            help="Solo archivos .csv. Tamaño máximo recomendado: 200 MB.",
        )

        with st.expander("Opciones de lectura del CSV"):
            sep = st.selectbox("Separador", [",", ";", "\t", "|"], index=0)
            decimal = st.selectbox("Separador decimal", [".", ","], index=0)
            encoding = st.selectbox("Encoding", ["utf-8", "latin1", "utf-8-sig"], index=0)

        st.download_button(
            "⬇️ Descargar plantilla CSV vacía",
            data=pd.DataFrame(columns=EXPECTED_COLUMNS).to_csv(index=False).encode("utf-8"),
            file_name="plantilla_fincas.csv",
            mime="text/csv",
            help="Descarga un CSV vacío con las columnas y el orden esperado.",
        )

# --- Resolución de la fuente de datos ---------------------------------------------------
if data_source == "🧪 Usar datos de ejemplo":
    raw_df = generate_sample_data()
    st.success("🧪 Usando **datos de ejemplo** generados automáticamente (150 fincas simuladas).")
elif uploaded_file is not None:
    try:
        raw_df = load_csv(uploaded_file, sep, decimal, encoding)
        st.toast(f"Archivo '{uploaded_file.name}' cargado correctamente ✅")
    except Exception as e:
        st.error(f"No se pudo leer el archivo: {e}")
        st.stop()
else:
    st.info(
        "⬅️ Sube un archivo CSV desde la barra lateral, o elige **'Usar datos de ejemplo'** "
        "para explorar la app sin tener un archivo a la mano.\n\n"
        "**Columnas esperadas:** `ID_Finca`, `Departamento`, `Tipo_Cultivo`, `Area_Hectareas`, "
        "`Produccion_Anual_Ton`, `Sistema_Riego_Tecnificado`, `Nivel_Tecnificacion`, "
        "`Precio_Venta_Por_Ton_COP`, `Tipo_Suelo`, `Fecha_Ultima_Auditoria`."
    )
    st.stop()

# --- Vista previa y validación / mapeo de columnas --------------------------------------
with st.expander("👀 Vista previa y validación de columnas", expanded=False):
    st.write(f"**Filas leídas:** {raw_df.shape[0]}  |  **Columnas leídas:** {raw_df.shape[1]}")
    st.dataframe(raw_df.head(10), use_container_width=True)

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in raw_df.columns]
    if missing_cols:
        st.warning(
            f"No se encontraron **{len(missing_cols)}** columna(s) esperada(s): {missing_cols}. "
            "Puedes mapearlas manualmente a las columnas de tu archivo, o continuar y la app "
            "intentará inferir los tipos automáticamente."
        )
        rename_map = {}
        for exp_col in missing_cols:
            choice = st.selectbox(
                f"¿Qué columna de tu archivo corresponde a `{exp_col}`?",
                ["(No aplica)"] + list(raw_df.columns),
                key=f"map_{exp_col}",
            )
            if choice != "(No aplica)":
                rename_map[choice] = exp_col
        if rename_map:
            raw_df = raw_df.rename(columns=rename_map)
            st.success(f"Columnas remapeadas: {rename_map}")
    else:
        st.success("✅ Todas las columnas esperadas están presentes.")

numeric_cols, categorical_cols, date_cols, id_cols = infer_column_types(raw_df)
df = coerce_types(raw_df, numeric_cols, date_cols)

# --------------------------------------------------------------------------------------
# FILTROS (SIDEBAR)
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.header("🔎 Filtros")
    filtered_df = df.copy()

    for col in categorical_cols:
        options = sorted(df[col].dropna().unique().tolist())
        selected = st.multiselect(f"{col}", options, default=options)
        if selected:
            filtered_df = filtered_df[filtered_df[col].isin(selected)]

    for col in numeric_cols:
        col_min, col_max = float(df[col].min()), float(df[col].max())
        if np.isnan(col_min) or np.isnan(col_max) or col_min == col_max:
            continue
        rng = st.slider(
            f"{col}", min_value=col_min, max_value=col_max,
            value=(col_min, col_max),
        )
        filtered_df = filtered_df[filtered_df[col].between(rng[0], rng[1])]

    for col in date_cols:
        valid_dates = df[col].dropna()
        if valid_dates.empty:
            continue
        min_d, max_d = valid_dates.min().date(), valid_dates.max().date()
        date_range = st.date_input(f"{col}", value=(min_d, max_d))
        if isinstance(date_range, tuple) and len(date_range) == 2:
            filtered_df = filtered_df[
                (filtered_df[col].dt.date >= date_range[0])
                & (filtered_df[col].dt.date <= date_range[1])
            ]

    st.caption(f"Registros tras filtros: **{len(filtered_df)} / {len(df)}**")

if filtered_df.empty:
    st.warning("Los filtros seleccionados no devuelven ningún registro.")
    st.stop()

# --------------------------------------------------------------------------------------
# KPIs GENERALES
# --------------------------------------------------------------------------------------
st.subheader("📌 Panorama general")
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    kpi_card("N° de fincas", f"{len(filtered_df):,}")
with k2:
    if COL_AREA in filtered_df:
        kpi_card("Área total (ha)", f"{filtered_df[COL_AREA].sum():,.1f}")
with k3:
    if COL_PROD in filtered_df:
        kpi_card("Producción total (Ton)", f"{filtered_df[COL_PROD].sum():,.1f}")
with k4:
    if COL_PRECIO in filtered_df:
        kpi_card("Precio prom. (COP/Ton)", f"{filtered_df[COL_PRECIO].mean():,.0f}")
with k5:
    if COL_RIEGO in filtered_df:
        pct_riego = (
            filtered_df[COL_RIEGO].astype(str).str.lower().isin(["si", "sí", "true", "1", "yes"])
        ).mean() * 100
        kpi_card("% Riego tecnificado", f"{pct_riego:,.1f}%")

st.divider()

# --------------------------------------------------------------------------------------
# TABS PRINCIPALES
# --------------------------------------------------------------------------------------
tab_overview, tab_quant, tab_qual, tab_graph, tab_corr, tab_data = st.tabs(
    [
        "🧭 Resumen y calidad de datos",
        "🔢 Análisis cuantitativo",
        "🔤 Análisis cualitativo",
        "📈 Análisis gráfico",
        "🔗 Correlaciones y cruces",
        "🗂️ Datos filtrados",
    ]
)

# ---------------------------- TAB 1: RESUMEN / CALIDAD ---------------------------------
with tab_overview:
    st.markdown("### Estructura del dataset")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.write(f"**Filas:** {df.shape[0]}  |  **Columnas:** {df.shape[1]}")
        st.write("**Tipos de columna detectados:**")
        st.write(f"- Numéricas: {numeric_cols}")
        st.write(f"- Categóricas: {categorical_cols}")
        st.write(f"- Fecha: {date_cols}")
        st.write(f"- Identificador: {id_cols}")
    with c2:
        st.markdown("**Valores nulos por columna**")
        nulls = df.isna().sum()
        nulls_pct = (df.isna().mean() * 100).round(2)
        nulls_df = pd.DataFrame({"Nulos": nulls, "% Nulos": nulls_pct})
        st.dataframe(nulls_df.style.background_gradient(cmap="Reds", subset=["% Nulos"]))

    st.markdown("### Duplicados")
    dup_count = df.duplicated().sum()
    dup_id_count = df.duplicated(subset=id_cols).sum() if id_cols else 0
    d1, d2 = st.columns(2)
    d1.metric("Filas totalmente duplicadas", int(dup_count))
    d2.metric(f"Duplicados por {id_cols[0] if id_cols else 'ID'}", int(dup_id_count))

    st.markdown("### Vista previa")
    st.dataframe(df.head(20), use_container_width=True)

# ---------------------------- TAB 2: CUANTITATIVO ---------------------------------------
with tab_quant:
    st.markdown("### Estadística descriptiva")
    if numeric_cols:
        desc = filtered_df[numeric_cols].describe().T
        desc["mediana"] = filtered_df[numeric_cols].median()
        desc["varianza"] = filtered_df[numeric_cols].var()
        desc["asimetría"] = filtered_df[numeric_cols].skew()
        desc["curtosis"] = filtered_df[numeric_cols].kurt()
        desc["CV (%)"] = (desc["std"] / desc["mean"] * 100).round(2)
        st.dataframe(desc.style.format(precision=2), use_container_width=True)
        download_df_button(desc.reset_index(), "estadistica_descriptiva.csv", "⬇️ Descargar estadística descriptiva")
    else:
        st.info("No se detectaron columnas numéricas.")

    st.markdown("### Detección de valores atípicos (IQR)")
    if numeric_cols:
        col_sel = st.selectbox("Selecciona variable numérica", numeric_cols, key="outlier_col")
        s = filtered_df[col_sel].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = s[(s < lower) | (s > upper)]
        o1, o2, o3 = st.columns(3)
        o1.metric("Límite inferior", f"{lower:,.2f}")
        o2.metric("Límite superior", f"{upper:,.2f}")
        o3.metric("N° de atípicos", f"{len(outliers)} ({len(outliers)/len(s)*100:.1f}%)")
        fig_box = px.box(filtered_df, y=col_sel, points="outliers", title=f"Boxplot de {col_sel}")
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("### Prueba de normalidad (Shapiro-Wilk)")
    if numeric_cols:
        col_norm = st.selectbox("Variable a evaluar", numeric_cols, key="norm_col")
        sample = filtered_df[col_norm].dropna()
        if 3 <= len(sample) <= 5000:
            stat, p_value = stats.shapiro(sample)
            n1, n2 = st.columns(2)
            n1.metric("Estadístico W", f"{stat:.4f}")
            n2.metric("p-valor", f"{p_value:.4f}")
            if p_value < 0.05:
                st.warning("Se rechaza H0: la variable **no** sigue una distribución normal (p < 0.05).")
            else:
                st.success("No se rechaza H0: la variable es compatible con una distribución normal (p ≥ 0.05).")
        else:
            st.info("El tamaño de muestra debe estar entre 3 y 5000 para esta prueba.")

# ---------------------------- TAB 3: CUALITATIVO -----------------------------------------
with tab_qual:
    st.markdown("### Frecuencias de variables categóricas")
    if categorical_cols:
        cat_sel = st.selectbox("Selecciona variable categórica", categorical_cols, key="cat_sel")
        freq = filtered_df[cat_sel].value_counts(dropna=False).rename("Frecuencia").to_frame()
        freq["% del total"] = (freq["Frecuencia"] / freq["Frecuencia"].sum() * 100).round(2)
        c1, c2 = st.columns([1, 1.4])
        with c1:
            st.dataframe(freq, use_container_width=True)
            download_df_button(freq.reset_index(), f"frecuencias_{cat_sel}.csv", "⬇️ Descargar tabla de frecuencias")
        with c2:
            chart_type = st.radio("Tipo de gráfico", ["Barras", "Pastel"], horizontal=True, key="qual_chart")
            if chart_type == "Barras":
                fig = px.bar(freq.reset_index(), x=cat_sel, y="Frecuencia", text="Frecuencia",
                             title=f"Distribución de {cat_sel}")
            else:
                fig = px.pie(freq.reset_index(), names=cat_sel, values="Frecuencia",
                             title=f"Distribución de {cat_sel}")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Tabla de contingencia (2 variables categóricas)")
        if len(categorical_cols) >= 2:
            c3, c4 = st.columns(2)
            with c3:
                var1 = st.selectbox("Variable 1", categorical_cols, index=0, key="cross1")
            with c4:
                remaining = [c for c in categorical_cols if c != var1]
                var2 = st.selectbox("Variable 2", remaining, index=0, key="cross2")
            crosstab = pd.crosstab(filtered_df[var1], filtered_df[var2])
            st.dataframe(crosstab, use_container_width=True)
            fig_hm = px.imshow(crosstab, text_auto=True, aspect="auto",
                                title=f"Mapa de calor: {var1} vs {var2}", color_continuous_scale="Blues")
            st.plotly_chart(fig_hm, use_container_width=True)

            # Chi-cuadrado de independencia
            try:
                chi2, p, dof, _ = stats.chi2_contingency(crosstab)
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("Chi-cuadrado", f"{chi2:.2f}")
                cc2.metric("p-valor", f"{p:.4f}")
                cc3.metric("Grados de libertad", dof)
                if p < 0.05:
                    st.warning("Existe asociación estadísticamente significativa entre las variables (p < 0.05).")
                else:
                    st.success("No hay evidencia de asociación significativa entre las variables (p ≥ 0.05).")
            except Exception:
                st.info("No fue posible calcular la prueba Chi-cuadrado con los datos actuales.")
    else:
        st.info("No se detectaron columnas categóricas.")

# ---------------------------- TAB 4: GRÁFICO --------------------------------------------
with tab_graph:
    st.markdown("### Distribución de una variable numérica")
    if numeric_cols:
        num_sel = st.selectbox("Variable numérica", numeric_cols, key="hist_col")
        color_by = st.selectbox("Colorear por (opcional)", ["Ninguno"] + categorical_cols, key="hist_color")
        fig_hist = px.histogram(
            filtered_df, x=num_sel, nbins=30, marginal="box",
            color=None if color_by == "Ninguno" else color_by,
            title=f"Distribución de {num_sel}",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("### Comparación numérica por categoría")
    if numeric_cols and categorical_cols:
        c1, c2 = st.columns(2)
        with c1:
            num_y = st.selectbox("Variable numérica (Y)", numeric_cols, key="box_num")
        with c2:
            cat_x = st.selectbox("Variable categórica (X)", categorical_cols, key="box_cat")
        fig_box2 = px.box(filtered_df, x=cat_x, y=num_y, color=cat_x,
                           title=f"{num_y} por {cat_x}")
        st.plotly_chart(fig_box2, use_container_width=True)

        fig_bar_agg = px.bar(
            filtered_df.groupby(cat_x, as_index=False)[num_y].mean(),
            x=cat_x, y=num_y, title=f"Promedio de {num_y} por {cat_x}",
        )
        st.plotly_chart(fig_bar_agg, use_container_width=True)

    st.markdown("### Relación entre dos variables numéricas")
    if len(numeric_cols) >= 2:
        c1, c2, c3 = st.columns(3)
        with c1:
            x_var = st.selectbox("Eje X", numeric_cols, index=0, key="scatter_x")
        with c2:
            y_var = st.selectbox("Eje Y", numeric_cols, index=min(1, len(numeric_cols) - 1), key="scatter_y")
        with c3:
            color_var = st.selectbox("Color (opcional)", ["Ninguno"] + categorical_cols, key="scatter_color")
        fig_scatter = px.scatter(
            filtered_df, x=x_var, y=y_var,
            color=None if color_var == "Ninguno" else color_var,
            trendline="ols", hover_data=id_cols,
            title=f"{y_var} vs {x_var}",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    if date_cols and numeric_cols:
        st.markdown("### Serie temporal")
        c1, c2 = st.columns(2)
        with c1:
            date_sel = st.selectbox("Variable de fecha", date_cols, key="ts_date")
        with c2:
            num_ts = st.selectbox("Variable numérica", numeric_cols, key="ts_num")
        ts_df = filtered_df.dropna(subset=[date_sel]).sort_values(date_sel)
        if not ts_df.empty:
            fig_ts = px.line(
                ts_df.groupby(pd.Grouper(key=date_sel, freq="M"))[num_ts].mean().reset_index(),
                x=date_sel, y=num_ts, markers=True,
                title=f"Evolución mensual promedio de {num_ts}",
            )
            st.plotly_chart(fig_ts, use_container_width=True)

# ---------------------------- TAB 5: CORRELACIONES ---------------------------------------
with tab_corr:
    st.markdown("### Matriz de correlación (numéricas)")
    if len(numeric_cols) >= 2:
        method = st.radio("Método", ["pearson", "spearman", "kendall"], horizontal=True)
        corr = filtered_df[numeric_cols].corr(method=method)
        fig_corr = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title=f"Matriz de correlación ({method})",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("### Matriz de dispersión (scatter matrix)")
        dims = st.multiselect("Variables a incluir", numeric_cols, default=numeric_cols[:4])
        color_scatter = st.selectbox("Color por", ["Ninguno"] + categorical_cols, key="splom_color")
        if len(dims) >= 2:
            fig_splom = px.scatter_matrix(
                filtered_df, dimensions=dims,
                color=None if color_scatter == "Ninguno" else color_scatter,
            )
            st.plotly_chart(fig_splom, use_container_width=True)
    else:
        st.info("Se necesitan al menos 2 columnas numéricas para calcular correlaciones.")

# ---------------------------- TAB 6: DATOS ------------------------------------------------
with tab_data:
    st.markdown("### Datos filtrados")
    st.dataframe(filtered_df, use_container_width=True)
    download_df_button(filtered_df, "datos_filtrados.csv", "⬇️ Descargar datos filtrados (CSV)")
