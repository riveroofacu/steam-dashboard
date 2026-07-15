"""
Paso 5 - Dashboard interactivo en Streamlit
----------------------------------------------------------
Dashboard para explorar el dataset final de Steam (steam_final.csv)
y visualizar los resultados del modelo de clasificación entrenado
en el paso anterior.

Requisitos cumplidos:
- 3 visualizaciones distintas
- 2 filtros interactivos (género y rango de precio)
- Sección de hallazgos/conclusiones
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Steam - Análisis de Minería de Datos", layout="wide")

# ---------- Carga de datos ----------
@st.cache_data
def cargar_datos():
    df = pd.read_csv("steam_final.csv")
    df["genero_principal"] = df["genero"].apply(lambda g: g.split(",")[0].strip())
    return df

@st.cache_data
def cargar_metricas():
    return pd.read_csv("metricas_modelos.csv")

df = cargar_datos()
df_metricas = cargar_metricas()


def aplicar_tema_oscuro(fig):
    """Aplica el tema oscuro de Plotly y fondo transparente a un gráfico,
    para que combine con el tema neón del dashboard."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


st.title("🎮 Análisis de Videojuegos en Steam")
st.markdown("Proyecto de Minería de Datos — Fuentes: API de SteamSpy + Web Scraping de la tienda de Steam")

# ---------- KPIs ----------
col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

juego_mas_jugado = df.loc[df["owners_estimado"].idxmax(), "name"]

col_kpi1.metric("Total de juegos", f"{len(df):,}")
col_kpi2.metric("Precio promedio", f"${df['price'].mean():.2f}")
col_kpi3.metric("% positivo promedio", f"{df['porcentaje_positivo'].mean():.1f}%")
col_kpi4.metric("Juego con más owners", juego_mas_jugado)

# ---------- Filtros (sidebar) ----------
st.sidebar.header("Filtros")

generos_disponibles = sorted(df["genero_principal"].unique())
genero_seleccionado = st.sidebar.selectbox(
    "Género",
    options=["Todos"] + generos_disponibles
)

precio_min, precio_max = float(df["price"].min()), float(df["price"].max())
rango_precio = st.sidebar.slider(
    "Rango de precio (USD)",
    min_value=precio_min,
    max_value=precio_max,
    value=(precio_min, precio_max)
)

# Aplicamos los filtros
df_filtrado = df[
    (df["price"] >= rango_precio[0]) & (df["price"] <= rango_precio[1])
]
if genero_seleccionado != "Todos":
    df_filtrado = df_filtrado[df_filtrado["genero_principal"] == genero_seleccionado]

st.sidebar.markdown(f"**Juegos filtrados: {len(df_filtrado)}**")

# ---------- Visualización 1: Top 15 juegos más jugados ----------
st.subheader("🏆 Top 15 juegos más jugados (por dueños estimados)")

top_15 = df_filtrado.sort_values("owners_estimado", ascending=False).head(15)

if len(top_15) > 0:
    fig_top = px.bar(
        top_15,
        x="owners_estimado",
        y="name",
        orientation="h",
        labels={"owners_estimado": "Dueños estimados", "name": "Juego", "rating": "Rating"},
        color="rating",
        color_discrete_map={"Positivo": "#00e5ff", "Negativo": "#bd00ff"}
    )
    fig_top.update_layout(yaxis={"categoryorder": "total ascending"})
    fig_top = aplicar_tema_oscuro(fig_top)
    st.plotly_chart(fig_top, use_container_width=True)
else:
    st.info("No hay juegos que coincidan con los filtros seleccionados.")

# ---------- Visualización 2: Box plot - % positivo por rango de precio ----------
st.subheader("💰 Relación entre precio y recepción del juego")

if len(df_filtrado) > 0:
    df_boxplot = df_filtrado.copy()

    # Agrupamos los juegos en rangos de precio, para que la relación se
    # lea de forma clara sin superposición de puntos individuales
    bins = [-0.01, 0, 10, 25, 40, float("inf")]
    etiquetas = ["Gratis", "0-10", "10-25", "25-40", "40+"]
    df_boxplot["rango_precio"] = pd.cut(df_boxplot["price"], bins=bins, labels=etiquetas)

    fig_box = px.box(
        df_boxplot,
        x="rango_precio",
        y="porcentaje_positivo",
        category_orders={"rango_precio": etiquetas},
        labels={"rango_precio": "Rango de precio (USD)", "porcentaje_positivo": "% Reviews positivas"},
        color="rango_precio",
        color_discrete_sequence=["#00e5ff", "#7d5fff", "#bd00ff", "#e100ff", "#ff2ec4"]
    )
    fig_box.update_layout(showlegend=False)
    fig_box = aplicar_tema_oscuro(fig_box)
    st.plotly_chart(fig_box, use_container_width=True)
else:
    st.info("No hay datos suficientes para este gráfico con los filtros actuales.")

# ---------- Visualización 3: Histograma de distribución de precios ----------
st.subheader("📊 Distribución de precios")

if len(df_filtrado) > 0:
    fig_hist = px.histogram(
        df_filtrado,
        x="price",
        nbins=30,
        labels={"price": "Precio (USD)"},
        color_discrete_sequence=["#00e5ff"]
    )
    fig_hist.update_layout(yaxis_title="Cantidad de juegos")
    fig_hist = aplicar_tema_oscuro(fig_hist)
    st.plotly_chart(fig_hist, use_container_width=True)
else:
    st.info("No hay datos suficientes para este gráfico con los filtros actuales.")

# ---------- Visualización 4: Matriz de confusión del modelo ----------
st.subheader("🎯 Resultado del modelo de clasificación (Random Forest)")

col1, col2 = st.columns([1, 1])

with col1:
    # Valores obtenidos en el Paso 4 (clasificacion.py)
    # Orden: filas = real, columnas = predicho | [Positivo, Negativo]
    matriz_confusion = [[62, 27], [37, 53]]

    fig_matriz = go.Figure(data=go.Heatmap(
        z=matriz_confusion,
        x=["Predicho: Positivo", "Predicho: Negativo"],
        y=["Real: Positivo", "Real: Negativo"],
        text=matriz_confusion,
        texttemplate="%{text}",
        colorscale=[[0, "#1a1a2e"], [0.5, "#7d5fff"], [1, "#00e5ff"]]
    ))
    fig_matriz.update_layout(title="Matriz de Confusión")
    fig_matriz = aplicar_tema_oscuro(fig_matriz)
    st.plotly_chart(fig_matriz, use_container_width=True)

with col2:
    st.markdown("**Comparación de modelos:**")
    st.dataframe(df_metricas, use_container_width=True, hide_index=True)

# ---------- Hallazgos ----------
st.subheader("💡 Hallazgos y Conclusiones")

st.markdown("""
- **El precio es la variable que más influye en la recepción de un juego**: en el modelo de Random Forest,
  el precio explicó más del 50% de la importancia total de las features, por encima de la cantidad de
  jugadores o el descuento aplicado.
- **Los juegos pagos muestran mejor y más consistente recepción que los gratuitos**: el análisis por rango
  de precio muestra que los juegos gratis tienen una mediana de aprobación más baja (~78%) y mayor
  dispersión, mientras que los rangos pagos rondan 85-90% de forma más estable — el precio podría actuar
  como filtro de calidad o de expectativas.
- **Popularidad no es sinónimo de buena recepción**: los 3 juegos más jugados del dataset (PUBG, Apex
  Legends, CS:GO) están clasificados como "Negativo", lo que sugiere que el éxito masivo —sobre todo en
  juegos free-to-play— no siempre viene acompañado de una satisfacción alta de la comunidad.
- **Random Forest superó claramente a la Regresión Logística** (64% vs 50% de accuracy), lo que sugiere
  que la relación entre las características de un juego y su recepción no es lineal.
""")

st.caption("Fuentes de datos: SteamSpy API (steamspy.com) y tienda de Steam (store.steampowered.com)")