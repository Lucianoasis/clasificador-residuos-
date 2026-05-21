"""
♻️ Clasificador de Residuos — Sistema Experto
Análisis de Datos II
Motor principal: experta (KnowledgeEngine)
"""

import os
import re
import unicodedata
import importlib
import sys
import types

import streamlit as st
import pandas as pd

# ── Patch para experta en Python 3.10+ ────────────────────────────────────────
import collections
import collections.abc
for name in ("Mapping", "MutableMapping", "Callable", "Sequence",
             "MutableSequence", "Set", "MutableSet", "Iterable",
             "Iterator", "KeysView", "ValuesView", "ItemsView"):
    if not hasattr(collections, name):
        setattr(collections, name, getattr(collections.abc, name))

from experta import KnowledgeEngine, Fact, Field, Rule, MATCH, AS, NOT, OR, AND, L, W, TEST, DefFacts

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RUTA_KEYWORDS = os.path.join(DATA_DIR, "keywords.csv")
RUTA_REGLAS   = os.path.join(DATA_DIR, "reglas.csv")
RUTA_AMBIGUOS = os.path.join(DATA_DIR, "ambiguos.csv")

ICONOS_CATEGORIA = {
    "reciclable":  "♻️",
    "organico":    "🌱",
    "especial":    "⚠️",
    "raee":        "🖥️",
    "peligroso":   "☢️",
    "basura":      "🚫",
    "desconocido": "❓",
}

# ══════════════════════════════════════════════════════════════════════════════
# FACTS
# ══════════════════════════════════════════════════════════════════════════════
class Residuo(Fact):
    tipo = Field(str, mandatory=True)

class Clasificacion(Fact):
    tipo         = Field(str)
    categoria    = Field(str)
    subcategoria = Field(str)
    contenedor   = Field(str)
    instrucciones= Field(list)
    errores      = Field(list)
    impacto      = Field(str)
    urgencia     = Field(str)

# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def cargar_sistema():
    df_kw  = pd.read_csv(RUTA_KEYWORDS, encoding="utf-8")
    df_reg = pd.read_csv(RUTA_REGLAS,   encoding="utf-8")
    df_amb = pd.read_csv(RUTA_AMBIGUOS, encoding="utf-8")

    keywords_dict = {}
    for _, row in df_kw.iterrows():
        keywords_dict.setdefault(row["tipo"], []).append(str(row["keyword"]).strip())

    reglas_dict = {}
    for _, row in df_reg.iterrows():
        reglas_dict[row["tipo"]] = {
            "categoria":    row["categoria"],
            "subcategoria": row["subcategoria"],
            "contenedor":   row["contenedor"],
            "instrucciones":[i.strip() for i in str(row["instrucciones"]).split(";") if i.strip()],
            "errores":      [e.strip() for e in str(row["errores_comunes"]).split(";") if e.strip()],
            "impacto":      row["impacto"],
            "urgencia":     row["urgencia"],
        }

    ambiguos_dict = {}
    for _, row in df_amb.iterrows():
        t = row["termino"]
        if t not in ambiguos_dict:
            ambiguos_dict[t] = {"pregunta": row["pregunta"], "opciones": {}}
        ambiguos_dict[t]["opciones"][str(row["opcion_num"])] = (row["tipo"], row["descripcion"])

    return keywords_dict, reglas_dict, ambiguos_dict


# ══════════════════════════════════════════════════════════════════════════════
# MOTOR EXPERTA (generado dinámicamente desde CSV)
# ══════════════════════════════════════════════════════════════════════════════
def construir_motor(reglas_dict: dict) -> type:
    """Genera una subclase de KnowledgeEngine con una @Rule por cada tipo del CSV."""
    atributos = {"resultado": None}

    for tipo, datos in reglas_dict.items():
        def hacer_regla(t, d):
            def metodo(self, r=MATCH.r):
                self.resultado = dict(tipo=t, **d)
            metodo.__name__ = f"regla_{t}"
            return Rule(AS.r << Residuo(tipo=L(t)))(metodo)
        atributos[f"regla_{tipo}"] = hacer_regla(tipo, datos)

    return type("ClasificadorDinamico", (KnowledgeEngine,), atributos)


def clasificar_con_experta(tipo: str, reglas_dict: dict) -> dict | None:
    """Ejecuta el motor de experta para un tipo dado y devuelve el resultado."""
    Motor = construir_motor(reglas_dict)
    engine = Motor()
    engine.reset()
    engine.declare(Residuo(tipo=tipo))
    engine.run()
    return engine.resultado


# ══════════════════════════════════════════════════════════════════════════════
# DETECCIÓN DE TIPO (desde keywords)
# ══════════════════════════════════════════════════════════════════════════════
def normalizar(texto: str) -> str:
    t = texto.lower().strip()
    t = unicodedata.normalize("NFD", t)
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return t


def detectar_tipo(texto: str, keywords_dict: dict, ambiguos_dict: dict):
    """
    Devuelve:
      - str con el tipo si hay coincidencia exacta
      - ("ambiguo", termino) si el texto es ambiguo
      - "desconocido" si no hay match
    """
    texto_norm = normalizar(texto)

    # Evaluar de mayor a menor longitud de keyword (más específico primero)
    pares = [(tipo, kw) for tipo, kws in keywords_dict.items() for kw in kws]
    pares.sort(key=lambda x: len(x[1]), reverse=True)

    for tipo, kw in pares:
        patron = r"\b" + re.escape(normalizar(kw)) + r"\b"
        if re.search(patron, texto_norm):
            return tipo

    # Verificar si es ambiguo
    for termino in ambiguos_dict:
        patron = r"\b" + re.escape(normalizar(termino)) + r"\b"
        if re.search(patron, texto_norm):
            return ("ambiguo", termino)

    return "desconocido"


# ══════════════════════════════════════════════════════════════════════════════
# UI — RESULTADO
# ══════════════════════════════════════════════════════════════════════════════
def mostrar_resultado(resultado: dict):
    cat = resultado.get("categoria", "").lower()
    icono = next((v for k, v in ICONOS_CATEGORIA.items() if k in cat), "♻️")

    st.success(f"{icono} **{resultado['categoria']}** — {resultado['subcategoria']}")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Contenedor", resultado["contenedor"])
        st.metric("Urgencia",   resultado["urgencia"])
    with col2:
        st.metric("Impacto",    resultado["impacto"])

    if resultado.get("instrucciones"):
        st.subheader("📋 Instrucciones")
        for i, inst in enumerate(resultado["instrucciones"], 1):
            st.write(f"{i}. {inst}")

    if resultado.get("errores"):
        st.subheader("❌ Errores comunes a evitar")
        for err in resultado["errores"]:
            st.warning(err)


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="♻️ Clasificador de Residuos",
    page_icon="♻️",
    layout="centered",
)

st.title("♻️ Clasificador de Residuos")
st.caption("Sistema Experto · Análisis de Datos II")

# Cargar datos
try:
    KEYWORDS_DICT, REGLAS_DICT, AMBIGUOS_DICT = cargar_sistema()
except FileNotFoundError as e:
    st.error(f"No se encontraron los archivos CSV en la carpeta `data/`: {e}")
    st.stop()

# Sidebar — métricas
with st.sidebar:
    st.header("⚙️ Sistema")
    st.metric("Tipos de residuo",  len(REGLAS_DICT))
    st.metric("Keywords cargadas", sum(len(v) for v in KEYWORDS_DICT.values()))
    st.metric("Términos ambiguos", len(AMBIGUOS_DICT))
    st.divider()
    st.caption("Motor: **experta** (KnowledgeEngine)")
    st.caption("Reglas generadas dinámicamente desde CSV")

    # API Key Gemini (opcional)
    gemini_key = st.text_input("🔑 API Key Gemini (opcional)", type="password",
                                value=st.secrets.get("GEMINI_API_KEY", ""))

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_texto, tab_imagen, tab_mapa = st.tabs(["💬 Por texto", "📷 Por imagen", "📍 Puntos Verdes"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Por texto
# ─────────────────────────────────────────────────────────────────────────────
with tab_texto:
    st.subheader("¿Qué residuo querés clasificar?")

    # Inicializar session_state
    if "texto_input" not in st.session_state:
        st.session_state.texto_input = ""
    if "tipo_detectado" not in st.session_state:
        st.session_state.tipo_detectado = None
    if "ambiguo_termino" not in st.session_state:
        st.session_state.ambiguo_termino = None
    if "resultado_final" not in st.session_state:
        st.session_state.resultado_final = None

    # Ejemplos rápidos
    ejemplos = ["botella", "pila", "papel", "restos de comida",
                "celular viejo", "aceite de cocina", "cartón", "vidrio"]
    st.write("**Ejemplos rápidos:**")
    cols = st.columns(4)
    for i, ej in enumerate(ejemplos):
        if cols[i % 4].button(ej, key=f"ej_{i}"):
            st.session_state.texto_input = ej
            st.session_state.tipo_detectado = None
            st.session_state.ambiguo_termino = None
            st.session_state.resultado_final = None

    texto = st.text_input("O escribí el material:", value=st.session_state.texto_input,
                           placeholder="ej: botella de plástico, pila AA, restos de comida...")

    if st.button("🔍 Clasificar", type="primary") and texto.strip():
        st.session_state.texto_input = texto
        st.session_state.resultado_final = None
        st.session_state.ambiguo_termino = None

        deteccion = detectar_tipo(texto, KEYWORDS_DICT, AMBIGUOS_DICT)

        if isinstance(deteccion, tuple) and deteccion[0] == "ambiguo":
            st.session_state.ambiguo_termino = deteccion[1]
            st.session_state.tipo_detectado = None
        elif deteccion == "desconocido":
            st.session_state.tipo_detectado = "desconocido"
            st.session_state.ambiguo_termino = None
        else:
            st.session_state.tipo_detectado = deteccion
            st.session_state.ambiguo_termino = None

    # ── Caso ambiguo: mostrar opciones y guardar elección ────────────────────
    if st.session_state.ambiguo_termino:
        termino = st.session_state.ambiguo_termino
        info = AMBIGUOS_DICT[termino]
        st.info(f'🤔 **"{termino}"** puede ser de distintos materiales.')

        opciones = info["opciones"]
        labels = [f"{v[1]}" for v in opciones.values()]
        tipos  = [v[0] for v in opciones.values()]

        eleccion = st.radio(info["pregunta"], labels, key="radio_ambiguo", index=None)

        if eleccion is not None:
            idx = labels.index(eleccion)
            tipo_elegido = tipos[idx]
            resultado = clasificar_con_experta(tipo_elegido, REGLAS_DICT)
            if resultado:
                st.session_state.resultado_final = resultado
            else:
                st.session_state.resultado_final = {"categoria": "Desconocido",
                    "subcategoria": "-", "contenedor": "-",
                    "instrucciones": [], "errores": [],
                    "impacto": "-", "urgencia": "-"}
            # Limpiar estado de ambigüedad para no volver a mostrar el radio
            st.session_state.ambiguo_termino = None

    # ── Caso tipo directo ────────────────────────────────────────────────────
    if st.session_state.tipo_detectado:
        tipo = st.session_state.tipo_detectado
        if tipo == "desconocido":
            st.warning("❓ No se reconoció el material. Intentá ser más específico.")
            st.info("Ejemplo: en lugar de 'botella', escribí 'botella de plástico' o 'botella de vidrio'.")
        else:
            resultado = clasificar_con_experta(tipo, REGLAS_DICT)
            if resultado:
                st.session_state.resultado_final = resultado
            st.session_state.tipo_detectado = None

    # ── Mostrar resultado final ───────────────────────────────────────────────
    if st.session_state.resultado_final:
        st.divider()
        mostrar_resultado(st.session_state.resultado_final)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Por imagen (Gemini)
# ─────────────────────────────────────────────────────────────────────────────
with tab_imagen:
    st.subheader("Subí una foto del residuo")

    if not gemini_key:
        st.info("Ingresá tu API Key de Gemini en el panel lateral para usar esta función.")
    else:
        imagen = st.file_uploader("Seleccioná una imagen", type=["jpg", "jpeg", "png", "webp"])
        if imagen:
            st.image(imagen, caption="Imagen cargada", use_column_width=True)
            if st.button("🔍 Analizar imagen", type="primary"):
                with st.spinner("Analizando con Gemini..."):
                    try:
                        from google import genai as gai
                        client = gai.Client(api_key=gemini_key)
                        img_bytes = imagen.read()
                        mime = imagen.type

                        prompt = (
                            "Sos un experto en clasificación de residuos para reciclaje. "
                            "Analizá la imagen y describí en UNA sola línea qué material es "
                            "(por ejemplo: 'botella de plástico', 'lata de aluminio', 'pila', etc.). "
                            "Respondé SOLO con el nombre del material, sin explicaciones adicionales."
                        )
                        resp = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=[prompt, {"mime_type": mime, "data": img_bytes}]
                        )
                        material_detectado = resp.text.strip()
                        st.success(f"Gemini identificó: **{material_detectado}**")

                        # Clasificar con el motor experta
                        deteccion = detectar_tipo(material_detectado, KEYWORDS_DICT, AMBIGUOS_DICT)
                        if isinstance(deteccion, tuple):
                            st.info(f'"{deteccion[1]}" es ambiguo. Usá la pestaña de texto para especificar.')
                        elif deteccion == "desconocido":
                            st.warning("No se encontró regla para ese material.")
                        else:
                            resultado = clasificar_con_experta(deteccion, REGLAS_DICT)
                            if resultado:
                                st.divider()
                                mostrar_resultado(resultado)
                    except Exception as e:
                        st.error(f"Error al consultar Gemini: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Puntos Verdes
# ─────────────────────────────────────────────────────────────────────────────
with tab_mapa:
    st.subheader("📍 Puntos Verdes más cercanos (CABA)")

    RUTA_PV = os.path.join(DATA_DIR, "puntos-verdes.csv")
    if not os.path.exists(RUTA_PV):
        st.warning("No se encontró `data/puntos-verdes.csv`. Subilo desde el portal de datos de CABA.")
        st.markdown("[Descargar dataset oficial](https://data.buenosaires.gob.ar/dataset/puntos-verdes)")
    else:
        col_lat, col_lon = st.columns(2)
        lat = col_lat.number_input("Latitud",  value=-34.6037, format="%.6f")
        lon = col_lon.number_input("Longitud", value=-58.3816, format="%.6f")
        n_cercanos = st.slider("Mostrar los N más cercanos", 3, 20, 5)

        if st.button("📍 Buscar puntos cercanos", type="primary"):
            import math
            df_pv = pd.read_csv(RUTA_PV, encoding="utf-8")

            # Detectar columnas de coordenadas
            lat_col = next((c for c in df_pv.columns if "lat" in c.lower()), None)
            lon_col = next((c for c in df_pv.columns if "lon" in c.lower() or "lng" in c.lower()), None)

            if not lat_col or not lon_col:
                st.error("No se encontraron columnas de latitud/longitud en el CSV.")
            else:
                def distancia(r):
                    dlat = math.radians(r[lat_col] - lat)
                    dlon = math.radians(r[lon_col] - lon)
                    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat)) * \
                        math.cos(math.radians(r[lat_col])) * math.sin(dlon/2)**2
                    return 6371 * 2 * math.asin(math.sqrt(a)) * 1000  # metros

                df_pv = df_pv.dropna(subset=[lat_col, lon_col])
                df_pv["distancia_m"] = df_pv.apply(distancia, axis=1)
                cercanos = df_pv.nsmallest(n_cercanos, "distancia_m")

                for _, row in cercanos.iterrows():
                    dist = row["distancia_m"]
                    nombre = row.get("nombre", row.get("direccion", "Punto Verde"))
                    st.write(f"📌 **{nombre}** — {dist:.0f} m")

                # Mapa
                mapa_df = cercanos[[lat_col, lon_col]].rename(
                    columns={lat_col: "lat", lon_col: "lon"})
                st.map(mapa_df)
