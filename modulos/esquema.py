"""
Esquema único del dataset PetIA Shop  (Integrante 1).

TODO el proyecto (generador, validación, formulario, modelos) debe leer
columnas, categorías y rangos desde aquí, para que nadie use nombres distintos.

Predictoras (5):
    peso_kg          numérica continua
    altura_cm        numérica continua
    edad_anios       numérica entera
    nivel_actividad  categórica ordinal  (Baja < Media < Alta)
    tipo_pelaje      categórica nominal  (Corto, Largo, Rizado, Sin pelo, Plumas, Caparazón)
Etiquetas (2):
    tipo             Perro, Gato, Ave, Tortuga
    raza             20 razas / variedades (5 por tipo)
"""

COLUMNA_ID = "id"
PREDICTORAS = ["peso_kg", "altura_cm", "edad_anios", "nivel_actividad", "tipo_pelaje"]
NUMERICAS = ["peso_kg", "altura_cm", "edad_anios"]
CATEGORICAS = ["nivel_actividad", "tipo_pelaje"]
ETIQUETAS = ["tipo", "raza"]
COLUMNAS = [COLUMNA_ID] + PREDICTORAS + ETIQUETAS

NIVELES_ACTIVIDAD = ["Baja", "Media", "Alta"]          # orden ordinal 0,1,2
TIPOS_PELAJE = ["Corto", "Largo", "Rizado", "Sin pelo", "Plumas", "Caparazón"]
TIPOS = ["Perro", "Gato", "Ave", "Tortuga"]

# Rangos físicos aceptados (validación de calidad)
RANGOS = {
    "peso_kg":    (0.01, 120.0),
    "altura_cm":  (3.0, 100.0),
    "edad_anios": (0, 80),
}

# Pelajes compatibles con cada tipo (regla de coherencia)
PELAJE_POR_TIPO = {
    "Perro":   ["Corto", "Largo", "Rizado", "Sin pelo"],
    "Gato":    ["Corto", "Largo", "Sin pelo"],
    "Ave":     ["Plumas"],
    "Tortuga": ["Caparazón"],
}

# Catálogo de razas.  'wiki' = título del artículo en en.wikipedia.org
# (se usa para obtener la fotografía vinculada a la etiqueta predicha).
#   peso=(media, desv) kg   altura=(media, desv) cm   edad=(min, max) años
#   actividad = probas [Baja, Media, Alta]   pelaje = {tipo: proba}
RAZAS = {
    # ---------------- PERROS ----------------
    "Chihuahua": dict(tipo="Perro", wiki="Chihuahua_(dog)",
        peso=(2.4, 0.7), altura=(20, 3), edad=(1, 18),
        actividad=[0.15, 0.45, 0.40], pelaje={"Corto": 0.7, "Largo": 0.3}),
    "Beagle": dict(tipo="Perro", wiki="Beagle",
        peso=(11, 2.0), altura=(37, 3), edad=(1, 15),
        actividad=[0.10, 0.30, 0.60], pelaje={"Corto": 1.0}),
    "Labrador Retriever": dict(tipo="Perro", wiki="Labrador_Retriever",
        peso=(30, 4.0), altura=(57, 4), edad=(1, 14),
        actividad=[0.10, 0.35, 0.55], pelaje={"Corto": 1.0}),
    "Pastor Alemán": dict(tipo="Perro", wiki="German_Shepherd",
        peso=(33, 4.5), altura=(61, 4), edad=(1, 13),
        actividad=[0.05, 0.30, 0.65], pelaje={"Corto": 0.6, "Largo": 0.4}),
    "Caniche": dict(tipo="Perro", wiki="Poodle",
        peso=(12, 3.5), altura=(40, 6), edad=(1, 17),
        actividad=[0.15, 0.45, 0.40], pelaje={"Rizado": 1.0}),
    # ---------------- GATOS ----------------
    "Siamés": dict(tipo="Gato", wiki="Siamese_cat",
        peso=(4.0, 0.8), altura=(24, 2.5), edad=(1, 19),
        actividad=[0.10, 0.40, 0.50], pelaje={"Corto": 1.0}),
    "Persa": dict(tipo="Gato", wiki="Persian_cat",
        peso=(4.5, 0.9), altura=(25, 2.5), edad=(1, 17),
        actividad=[0.60, 0.30, 0.10], pelaje={"Largo": 1.0}),
    "Maine Coon": dict(tipo="Gato", wiki="Maine_Coon",
        peso=(7.5, 1.5), altura=(33, 3), edad=(1, 16),
        actividad=[0.25, 0.50, 0.25], pelaje={"Largo": 1.0}),
    "Esfinge": dict(tipo="Gato", wiki="Sphynx_cat",
        peso=(3.8, 0.7), altura=(23, 2.5), edad=(1, 15),
        actividad=[0.15, 0.45, 0.40], pelaje={"Sin pelo": 1.0}),
    "Bengalí": dict(tipo="Gato", wiki="Bengal_cat",
        peso=(5.0, 1.0), altura=(27, 2.5), edad=(1, 16),
        actividad=[0.05, 0.30, 0.65], pelaje={"Corto": 1.0}),
    # ---------------- AVES ----------------
    "Periquito australiano": dict(tipo="Ave", wiki="Budgerigar",
        peso=(0.035, 0.006), altura=(10, 1.2), edad=(0, 10),
        actividad=[0.10, 0.30, 0.60], pelaje={"Plumas": 1.0}),
    "Canario": dict(tipo="Ave", wiki="Domestic_canary",
        peso=(0.022, 0.004), altura=(8, 1.0), edad=(0, 12),
        actividad=[0.15, 0.45, 0.40], pelaje={"Plumas": 1.0}),
    "Ninfa (Cacatúa)": dict(tipo="Ave", wiki="Cockatiel",
        peso=(0.09, 0.015), altura=(17, 2.0), edad=(0, 20),
        actividad=[0.20, 0.50, 0.30], pelaje={"Plumas": 1.0}),
    "Agapornis": dict(tipo="Ave", wiki="Lovebird",
        peso=(0.05, 0.008), altura=(9.5, 1.0), edad=(0, 15),
        actividad=[0.10, 0.40, 0.50], pelaje={"Plumas": 1.0}),
    "Loro gris africano": dict(tipo="Ave", wiki="Grey_parrot",
        peso=(0.45, 0.06), altura=(22, 2.5), edad=(0, 45),
        actividad=[0.30, 0.50, 0.20], pelaje={"Plumas": 1.0}),
    # ---------------- TORTUGAS ----------------
    "Tortuga de orejas rojas": dict(tipo="Tortuga", wiki="Red-eared_slider",
        peso=(1.2, 0.4), altura=(8, 1.5), edad=(0, 35),
        actividad=[0.55, 0.35, 0.10], pelaje={"Caparazón": 1.0}),
    "Tortuga rusa": dict(tipo="Tortuga", wiki="Russian_tortoise",
        peso=(0.9, 0.25), altura=(9, 1.5), edad=(0, 45),
        actividad=[0.65, 0.30, 0.05], pelaje={"Caparazón": 1.0}),
    "Tortuga sulcata": dict(tipo="Tortuga", wiki="African_spurred_tortoise",
        peso=(28, 12), altura=(26, 6), edad=(0, 70),
        actividad=[0.70, 0.25, 0.05], pelaje={"Caparazón": 1.0}),
    "Tortuga leopardo": dict(tipo="Tortuga", wiki="Leopard_tortoise",
        peso=(12, 4.5), altura=(20, 4), edad=(0, 60),
        actividad=[0.70, 0.25, 0.05], pelaje={"Caparazón": 1.0}),
    "Tortuga de caja": dict(tipo="Tortuga", wiki="Box_turtle",
        peso=(0.5, 0.15), altura=(7, 1.2), edad=(0, 50),
        actividad=[0.60, 0.35, 0.05], pelaje={"Caparazón": 1.0}),
}

RAZAS_POR_TIPO = {t: [r for r, d in RAZAS.items() if d["tipo"] == t] for t in TIPOS}

DESCRIPCIONES = {
    "id": "Identificador único y consecutivo del registro.",
    "peso_kg": "Peso corporal de la mascota en kilogramos (2 decimales; aves hasta 3).",
    "altura_cm": "Altura en centímetros: a la cruz (perro/gato), de pie (ave) o del caparazón (tortuga).",
    "edad_anios": "Edad cumplida en años (entero).",
    "nivel_actividad": "Nivel de actividad observado: Baja, Media o Alta (ordinal).",
    "tipo_pelaje": "Cobertura corporal: Corto, Largo, Rizado, Sin pelo, Plumas o Caparazón.",
    "tipo": "Etiqueta 1 — clase de mascota: Perro, Gato, Ave o Tortuga.",
    "raza": "Etiqueta 2 — raza o variedad (20 clases, 5 por tipo).",
}


def slug(texto: str) -> str:
    """'Ninfa (Cacatúa)' -> 'ninfa-cacatua'  (nombre de archivo de la imagen)."""
    import unicodedata, re
    t = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")


def esquema_publico() -> dict:
    """Versión JSON del esquema para el frontend (/api/esquema)."""
    return {
        "predictoras": PREDICTORAS,
        "etiquetas": ETIQUETAS,
        "numericas": NUMERICAS,
        "categoricas": CATEGORICAS,
        "niveles_actividad": NIVELES_ACTIVIDAD,
        "tipos_pelaje": TIPOS_PELAJE,
        "tipos": TIPOS,
        "rangos": RANGOS,
        "pelaje_por_tipo": PELAJE_POR_TIPO,
        "razas_por_tipo": RAZAS_POR_TIPO,
        "razas": {r: {"tipo": d["tipo"], "wiki": d["wiki"], "slug": slug(r)}
                  for r, d in RAZAS.items()},
        "descripciones": DESCRIPCIONES,
    }
