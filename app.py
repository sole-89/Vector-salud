"""
Vector Salud: gestión de camas e insumos hospitalarios mediante vectores,
matrices, sumatorias y una función cuadrática.

Proyecto universitario de Funciones y Matrices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

RECINTOS = ["Hospital 1", "Hospital 2", "Hospital 3"]
INSUMOS = ["Camas", "Oxígeno", "Mascarillas"]

STOCK_INICIAL = np.array([[12, 40, 30], [4, 10, 15], [9, 25, 20]], dtype=float)
DEMANDA_INICIAL = np.array([[10, 30, 25], [9, 28, 22], [6, 20, 24]], dtype=float)
COSTOS_INICIALES = [50.0, 8.0, 3.0]
A_INICIAL, B_INICIAL, C_INICIAL = -0.25, 6.0, 10.0

AZUL_PETROLEO = "#0B6E8A"
AZUL_OSCURO = "#0F2A43"

st.set_page_config(
    page_title="Vector Salud",
    page_icon=":material/local_hospital:",
    layout="wide",
)

st.markdown(
    """
    <style>
        :root {
            font-size: 17px;
        }

        /* Tema claro forzado, independiente del modo del sistema o navegador. */
        html, body, [class*="css"], .stApp, .stAppViewContainer,
        section[data-testid="stMain"], .block-container,
        [data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            color: #0F2A43 !important;
        }

        /* Títulos y acentos en azul petróleo; título principal grande. */
        h1, h2, h3, h4, h5, h6, .stTitle, .stHeader,
        [data-testid="stTitle"], [data-testid="stHeader"] {
            color: #0B6E8A !important;
        }
        h1, [data-testid="stTitle"] {
            font-size: 3rem !important;
            line-height: 1.1 !important;
            font-weight: 800 !important;
        }

        /* Texto general, etiquetas y leyendas. */
        .stCaption, [data-testid="stCaptionContainer"],
        .stWidgetLabel, [data-testid="stWidgetLabel"], label {
            color: #0F2A43 !important;
        }

        /* Tablas legibles: fondo blanco y encabezados gris azulado claro. */
        .stDataFrame, [data-testid="stDataFrame"],
        .stTable, [data-testid="stTable"] {
            background-color: #FFFFFF !important;
            color: #0F2A43 !important;
            border: 1px solid #C9DCE5 !important;
            border-radius: 8px !important;
        }
        .stDataFrame [role="columnheader"],
        [data-testid="stDataFrame"] [role="columnheader"],
        .stTable thead th {
            background-color: #EEF3F6 !important;
            color: #0F2A43 !important;
            font-weight: 700 !important;
        }

        /* Campos numéricos con texto azul oscuro sobre fondo blanco. */
        input, input[type="number"],
        .stNumberInput input,
        [data-testid="stNumberInput"] input {
            background-color: #FFFFFF !important;
            color: #0F2A43 !important;
            border: 1px solid #B8CCD8 !important;
            border-radius: 6px !important;
        }

        /* Tarjetas de resultados y botones de acento. */
        .stMetric, [data-testid="stMetric"],
        .stContainer, [data-testid="stContainer"] {
            background-color: #EEF3F6 !important;
            color: #0F2A43 !important;
            border: 1px solid #C9DCE5 !important;
            border-radius: 10px !important;
        }
        .stMetricLabel, [data-testid="stMetricLabel"] {
            color: #0F2A43 !important;
        }
        .stMetricValue, [data-testid="stMetricValue"] {
            color: #0B6E8A !important;
            font-weight: 700 !important;
        }
        .stButton > button {
            background-color: #0B6E8A !important;
            color: #FFFFFF !important;
            border: 1px solid #0B6E8A !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }
        .stButton > button:hover {
            background-color: #0F2A43 !important;
            color: #FFFFFF !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def matriz_inicial(valores: np.ndarray) -> pd.DataFrame:
    """Construye una matriz editable con recintos e insumos rotulados."""
    return pd.DataFrame(valores, index=RECINTOS, columns=INSUMOS)


def inicializar_estado() -> None:
    """Prepara el estado de la sesión una sola vez."""
    if "version_datos" not in st.session_state:
        st.session_state.version_datos = 0
        st.session_state.stock = matriz_inicial(STOCK_INICIAL)
        st.session_state.demanda = matriz_inicial(DEMANDA_INICIAL)
        st.session_state.costos = list(COSTOS_INICIALES)
        st.session_state.coeficientes = {
            "a": A_INICIAL,
            "b": B_INICIAL,
            "c": C_INICIAL,
        }


def restaurar_datos() -> None:
    """Restaura todos los datos de ejemplo y fuerza la recreación de los editores."""
    st.session_state.version_datos += 1
    st.session_state.stock = matriz_inicial(STOCK_INICIAL)
    st.session_state.demanda = matriz_inicial(DEMANDA_INICIAL)
    st.session_state.costos = list(COSTOS_INICIALES)
    st.session_state.coeficientes = {
        "a": A_INICIAL,
        "b": B_INICIAL,
        "c": C_INICIAL,
    }
    st.session_state.aviso_restauracion = True


def formato_cantidad(valor: float) -> str:
    """Muestra cantidades enteras sin decimales y separador de miles español."""
    if np.isclose(valor, round(valor), atol=1e-9):
        return f"{int(round(valor)):,}".replace(",", ".")
    return f"{valor:,.2f}".replace(",", ".")


def calcular_traslados(
    deficit: np.ndarray,
) -> tuple[list[tuple[str, str, str, float]], np.ndarray]:
    """Distribuye los excedentes desde los sobrantes hacia los mayores faltantes."""
    pendiente = np.maximum(deficit, 0.0).astype(float)
    excedentes = np.maximum(-deficit, 0.0).astype(float)
    traslados: list[tuple[str, str, str, float]] = []

    for indice_insumo, insumo in enumerate(INSUMOS):
        receptores = sorted(
            (
                indice
                for indice in range(len(RECINTOS))
                if pendiente[indice, indice_insumo] > 1e-9
            ),
            key=lambda indice: (-pendiente[indice, indice_insumo], indice),
        )

        for receptor in receptores:
            while pendiente[receptor, indice_insumo] > 1e-9:
                origen = int(np.argmax(excedentes[:, indice_insumo]))
                if excedentes[origen, indice_insumo] <= 1e-9:
                    break

                cantidad = min(
                    pendiente[receptor, indice_insumo],
                    excedentes[origen, indice_insumo],
                )
                traslados.append(
                    (
                        insumo,
                        RECINTOS[origen],
                        RECINTOS[receptor],
                        float(cantidad),
                    )
                )
                pendiente[receptor, indice_insumo] -= cantidad
                excedentes[origen, indice_insumo] -= cantidad

    return traslados, pendiente


inicializar_estado()

st.title("Vector Salud", icon=":material/local_hospital:")
st.caption(
    "Gestión de camas e insumos de la red pública mediante vectores, matrices, "
    "sumatorias y una función cuadrática."
)

# -----------------------------------------------------------------------------
# 1. Stock S y demanda D
# -----------------------------------------------------------------------------
st.header("1. Stock S y demanda D", icon=":material/database:")
with st.container(horizontal=True, vertical_alignment="bottom"):
    st.button(
        "Cargar datos de ejemplo",
        icon=":material/refresh:",
        on_click=restaurar_datos,
    )
    st.caption("Al modificar cualquier celda o parámetro, los resultados se recalculan de inmediato.")

if st.session_state.pop("aviso_restauracion", False):
    st.toast("Se restauraron los datos de ejemplo.")

version = st.session_state.version_datos
st.caption("Las cantidades se expresan en unidades físicas por hospital e insumo.")

columnas_editables = {
    insumo: st.column_config.NumberColumn(
        insumo,
        min_value=0.0,
        step=1.0,
        format="%d",
    )
    for insumo in INSUMOS
}

col_stock, col_demanda = st.columns(2)
with col_stock:
    st.subheader("Stock disponible (S)")
    stock = st.data_editor(
        st.session_state.stock,
        key=f"editor_stock_{version}",
        column_config=columnas_editables,
        num_rows="fixed",
        alt="Matriz de stock disponible por recinto e insumo",
    )
    st.session_state.stock = stock.copy()

with col_demanda:
    st.subheader("Demanda requerida (D)")
    demanda = st.data_editor(
        st.session_state.demanda,
        key=f"editor_demanda_{version}",
        column_config=columnas_editables,
        num_rows="fixed",
        alt="Matriz de demanda requerida por recinto e insumo",
    )
    st.session_state.demanda = demanda.copy()

if stock.isna().to_numpy().any() or demanda.isna().to_numpy().any():
    st.error("Se requiere completar todas las celdas de las matrices S y D para continuar.")
    st.stop()

S = st.session_state.stock.to_numpy(dtype=float)
D = st.session_state.demanda.to_numpy(dtype=float)

# -----------------------------------------------------------------------------
# 2. Matriz de déficit
# -----------------------------------------------------------------------------
F = D - S
st.header("2. Déficit F = D − S", icon=":material/table_chart:")
st.caption("La matriz de déficit se calcula elemento a elemento como F = D − S.")


def colorear_deficit(valor: float) -> str:
    if valor > 1e-9:
        return (
            "color: #8E1B1B; background-color: #FCE8E6; "
            "font-weight: 700; text-align: center;"
        )
    if valor < -1e-9:
        return (
            "color: #12603A; background-color: #E4F3EA; "
            "font-weight: 600; text-align: center;"
        )
    return (
        "color: #5F6D72; background-color: #EDF1F2; "
        "text-align: center;"
    )


deficit_df = pd.DataFrame(F, index=RECINTOS, columns=INSUMOS)
st.dataframe(
    deficit_df.style.map(colorear_deficit),
    column_config={
        insumo: st.column_config.NumberColumn(insumo, format="%+.0f")
        for insumo in INSUMOS
    },
    alt="Matriz de déficit: valores positivos en rojo, negativos en verde y ceros en gris",
)
st.caption("Rojo: faltante · Verde: excedente · Gris: equilibrio.")

# -----------------------------------------------------------------------------
# 3. Totales de la red
# -----------------------------------------------------------------------------
st.header("3. Totales de la red", icon=":material/calculate:")
faltante_total = np.maximum(F, 0.0).sum(axis=0)
excedente_disponible = np.maximum(-F, 0.0).sum(axis=0)
cantidad_por_reponer = np.maximum(faltante_total - excedente_disponible, 0.0)

resumen_df = pd.DataFrame(
    {
        "Insumo": INSUMOS,
        "Faltante total": faltante_total,
        "Excedente disponible": excedente_disponible,
        "Cantidad por reponer": cantidad_por_reponer,
    }
)
st.dataframe(
    resumen_df,
    hide_index=True,
    column_config={
        "Insumo": st.column_config.TextColumn("Insumo"),
        **{
            columna: st.column_config.NumberColumn(columna, format="%d")
            for columna in (
                "Faltante total",
                "Excedente disponible",
                "Cantidad por reponer",
            )
        },
    },
    alt="Sumatorias de faltantes, excedentes y cantidades por reponer para cada insumo",
)

# -----------------------------------------------------------------------------
# 4. Traslados propuestos
# -----------------------------------------------------------------------------
st.header("4. Traslados propuestos entre hospitales", icon=":material/swap_horiz:")
st.caption(
    "Se priorizan los recintos con mayor faltante y se toma el excedente disponible "
    "de mayor tamaño en cada paso."
)

traslados, faltante_pendiente = calcular_traslados(F)
if traslados:
    lineas_traslados = [
        f"- {formato_cantidad(cantidad)} de {insumo.lower()}: {origen} → {destino}"
        for insumo, origen, destino, cantidad in traslados
    ]
    st.markdown("\n".join(lineas_traslados))
else:
    st.caption("No se requieren traslados con los datos actuales.")

sin_cubrir = [
    f"- {recinto}: {formato_cantidad(faltante_pendiente[indice_recinto, indice_insumo])} "
    f"de {insumo.lower()}"
    for indice_recinto, recinto in enumerate(RECINTOS)
    for indice_insumo, insumo in enumerate(INSUMOS)
    if faltante_pendiente[indice_recinto, indice_insumo] > 1e-9
]
if sin_cubrir:
    st.markdown("**Faltantes que quedan sin cubrir**")
    st.markdown("\n".join(sin_cubrir))
else:
    st.success(
        "Todos los faltantes quedan cubiertos con el excedente disponible.",
        icon=":material/check_circle:",
    )

# -----------------------------------------------------------------------------
# 5. Costo de reposición
# -----------------------------------------------------------------------------
st.header("5. Costo de reponer el faltante", icon=":material/payments:")
st.subheader("Costos unitarios (p)")
st.caption("Valores en miles de pesos por unidad repuesta.")
columnas_costo = st.columns(3)
costos_actuales: list[float] = []
for indice, (columna, insumo) in enumerate(zip(columnas_costo, INSUMOS)):
    with columna:
        costos_actuales.append(
            float(
                st.number_input(
                    insumo,
                    min_value=0.0,
                    step=1.0,
                    value=float(st.session_state.costos[indice]),
                    key=f"costo_{indice}_{version}",
                    format="%.2f",
                )
            )
        )
st.session_state.costos = costos_actuales
p = np.array(costos_actuales, dtype=float)

st.caption(
    "Para cada hospital se calcula el producto punto entre sus faltantes positivos "
    "y el vector de costos unitarios p. Cifras en miles de pesos."
)

faltantes_positivos = np.maximum(F, 0.0)
costo_por_insumo = faltantes_positivos * p
costo_df = pd.DataFrame(costo_por_insumo, index=RECINTOS, columns=INSUMOS)
costo_df["Total por hospital"] = costo_por_insumo.sum(axis=1)
total_red = float(costo_df["Total por hospital"].sum())

st.dataframe(
    costo_df,
    column_config={
        insumo: st.column_config.NumberColumn(insumo, format="%.0f")
        for insumo in [*INSUMOS, "Total por hospital"]
    },
    alt="Costo de reposición por recinto, insumo y total de la red",
)
st.metric(
    "Costo total de la red (miles de pesos)",
    formato_cantidad(total_red),
    border=True,
)

# -----------------------------------------------------------------------------
# 6. Modelo cuadrático de llegadas
# -----------------------------------------------------------------------------
st.header("6. Modelo cuadrático de llegadas a urgencia", icon=":material/query_stats:")
st.caption("f(t) = a·t² + b·t + c, donde t corresponde a la hora del día.")

coeficientes = st.session_state.coeficientes
columnas_coeficientes = st.columns(3)
valores_coeficientes: dict[str, float] = {}
for columna, clave, etiqueta in zip(
    columnas_coeficientes,
    ("a", "b", "c"),
    ("Coeficiente a", "Coeficiente b", "Coeficiente c"),
):
    with columna:
        valores_coeficientes[clave] = float(
            st.number_input(
                etiqueta,
                value=float(coeficientes[clave]),
                key=f"coeficiente_{clave}_{version}",
                format="%.2f",
            )
        )
st.session_state.coeficientes = valores_coeficientes

a = valores_coeficientes["a"]
b = valores_coeficientes["b"]
c = valores_coeficientes["c"]

horas_diarias = np.arange(24)
llegadas_diarias = a * horas_diarias**2 + b * horas_diarias + c
total_diario = float(llegadas_diarias.sum())

tiempo_punta: float | None = None
pacientes_punta: float | None = None
if np.isclose(a, 0.0):
    st.error(
        "El coeficiente a debe ser distinto de cero para determinar el vértice "
        "de la función cuadrática."
    )
elif a > 0:
    st.warning(
        "Con un coeficiente a positivo la parábola tiene un mínimo; por ello no "
        "se define una hora punta mediante el vértice."
    )
else:
    tiempo_punta = float(-b / (2.0 * a))
    pacientes_punta = float(a * tiempo_punta**2 + b * tiempo_punta + c)
    metrica_hora, metrica_pacientes, metrica_total = st.columns(3)
    metrica_hora.metric("Hora punta", f"{formato_cantidad(tiempo_punta)} h", border=True)
    metrica_pacientes.metric(
        "Pacientes en la hora punta",
        f"{formato_cantidad(pacientes_punta)} por hora",
        border=True,
    )
    metrica_total.metric(
        "Total diario (t = 0 a 23)",
        f"{formato_cantidad(total_diario)} pacientes",
        border=True,
    )

horas_grafico = np.arange(25)
curva_df = pd.DataFrame(
    {
        "Hora": horas_grafico,
        "Pacientes por hora": a * horas_grafico**2 + b * horas_grafico + c,
    }
)

opcion_grafico: dict = {
    "tooltip": {"trigger": "axis"},
    "grid": {"left": 70, "right": 35, "top": 45, "bottom": 60},
    "xAxis": {
        "type": "value",
        "min": 0,
        "max": 24,
        "name": "Hora",
        "nameLocation": "middle",
        "nameGap": 32,
    },
    "yAxis": {
        "type": "value",
        "name": "Pacientes por hora",
        "nameLocation": "middle",
        "nameGap": 55,
    },
    "dataset": {"source": curva_df},
    "series": [
        {
            "type": "line",
            "encode": {"x": "Hora", "y": "Pacientes por hora"},
            "smooth": False,
            "showSymbol": False,
            "lineStyle": {"width": 3, "color": AZUL_PETROLEO},
            "itemStyle": {"color": AZUL_PETROLEO},
        }
    ],
}

if tiempo_punta is not None and 0.0 <= tiempo_punta <= 24.0:
    opcion_grafico["series"][0]["markPoint"] = {
        "symbol": "pin",
        "symbolSize": 58,
        "itemStyle": {"color": AZUL_OSCURO},
        "label": {"formatter": "Hora punta", "color": "#FFFFFF"},
        "data": [{"coord": [tiempo_punta, pacientes_punta]}],
    }
    opcion_grafico["series"][0]["markLine"] = {
        "symbol": "none",
        "lineStyle": {"type": "dashed", "color": AZUL_OSCURO, "width": 2},
        "label": {"formatter": f"t = {formato_cantidad(tiempo_punta)}"},
        "data": [{"xAxis": tiempo_punta}],
    }

st.echarts_chart(
    opcion_grafico,
    height=380,
    alt="Curva de llegadas a urgencia por hora, con el vértice marcado como hora punta",
)
