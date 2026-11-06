"""
Generador del dataset original PetIA Shop  (Integrante 1).

Crea data/mascotas.csv con 500,000 mascotas, 5 predictoras y 2 etiquetas
(tipo y raza).  Las distribuciones de cada raza se definen en
modulos/esquema.py a partir de rangos reales aproximados de cada raza y se
traslapan a propósito, de modo que los clasificadores no sean perfectos.

Además genera:
    data/diccionario_datos.md   -> diccionario de datos
    data/reporte_calidad.json   -> controles de calidad del CSV generado

Uso (desde la carpeta del proyecto):
    python data/generar_dataset.py              # 500,000 registros
    python data/generar_dataset.py 20000        # cantidad personalizada
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from modulos import esquema as E                    # noqa: E402
from modulos.validacion import validar_dataframe     # noqa: E402

SEMILLA = 2026
N = int(sys.argv[1]) if len(sys.argv) > 1 else 500_000
CARPETA = Path(__file__).resolve().parent
SALIDA_CSV = CARPETA / "mascotas.csv"

# Proporción de cada tipo (una tienda tiene más perros y gatos que tortugas)
PROP_TIPO = {"Perro": 0.36, "Gato": 0.30, "Ave": 0.20, "Tortuga": 0.14}


def generar(n: int, semilla: int = SEMILLA) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)

    # probabilidad de cada raza = prob. del tipo repartida con ligera variación
    razas, probs = [], []
    for tipo, p_tipo in PROP_TIPO.items():
        lista = E.RAZAS_POR_TIPO[tipo]
        w = rng.uniform(0.8, 1.2, len(lista))
        w = w / w.sum() * p_tipo
        razas += lista
        probs += list(w)
    probs = np.array(probs) / np.sum(probs)
    columna_raza = rng.choice(razas, size=n, p=probs)

    partes = []
    for raza in razas:
        k = int((columna_raza == raza).sum())
        d = E.RAZAS[raza]
        peso = rng.normal(*d["peso"], k)
        altura = rng.normal(*d["altura"], k)
        e_min, e_max = d["edad"]
        # edades sesgadas a jóvenes (más común en tiendas)
        edad = np.round(e_min + rng.beta(1.4, 2.6, k) * (e_max - e_min)).astype(int)
        actividad = rng.choice(E.NIVELES_ACTIVIDAD, size=k, p=d["actividad"])
        pel, pp = zip(*d["pelaje"].items())
        pelaje = rng.choice(pel, size=k, p=pp)
        partes.append(pd.DataFrame({
            "peso_kg": peso, "altura_cm": altura, "edad_anios": edad,
            "nivel_actividad": actividad, "tipo_pelaje": pelaje,
            "tipo": d["tipo"], "raza": raza,
        }))

    df = pd.concat(partes, ignore_index=True)

    # Límites físicos y redondeo (aves conservan 3 decimales de peso)
    lo, hi = E.RANGOS["peso_kg"]
    piso = np.where(df["tipo"] == "Ave", 0.012, 0.3)
    df["peso_kg"] = np.maximum(df["peso_kg"], piso).clip(lo, hi)
    df["peso_kg"] = np.where(df["tipo"] == "Ave",
                             df["peso_kg"].round(3), df["peso_kg"].round(2))
    lo, hi = E.RANGOS["altura_cm"]
    df["altura_cm"] = df["altura_cm"].clip(max(lo, 4.0), hi).round(1)

    df = df.sample(frac=1, random_state=semilla).reset_index(drop=True)
    df.insert(0, "id", np.arange(1, n + 1))
    return df[E.COLUMNAS]


def escribir_diccionario(df: pd.DataFrame) -> None:
    filas = []
    for col in E.COLUMNAS:
        s = df[col]
        if col in E.NUMERICAS or col == "id":
            dom = f"{s.min():g} – {s.max():g}"
            tipo = "Entero" if s.dtype.kind == "i" else "Decimal"
        else:
            vals = E.NIVELES_ACTIVIDAD if col == "nivel_actividad" else sorted(s.unique())
            dom = ", ".join(vals) if len(vals) <= 6 else f"{len(vals)} categorías"
            tipo = "Categórica ordinal" if col == "nivel_actividad" else "Categórica"
        rol = ("Identificador" if col == "id" else
               "Etiqueta" if col in E.ETIQUETAS else "Predictora")
        filas.append(f"| `{col}` | {rol} | {tipo} | {dom} | {E.DESCRIPCIONES[col]} |")

    conteo = df["raza"].value_counts()
    razas_md = "\n".join(
        f"| {E.RAZAS[r]['tipo']} | {r} | {conteo.get(r, 0):,} | "
        f"{df.loc[df.raza == r, 'peso_kg'].median():g} | "
        f"{df.loc[df.raza == r, 'altura_cm'].median():g} |"
        for t in E.TIPOS for r in E.RAZAS_POR_TIPO[t])

    texto = f"""# Diccionario de datos — PetIA Shop

**Archivo:** `data/mascotas.csv` · **Registros:** {len(df):,} · **Columnas:** {len(E.COLUMNAS)}
(1 identificador, 5 predictoras, 2 etiquetas) · **Codificación:** UTF-8 · **Separador:** coma

## Variables

| Columna | Rol | Tipo de dato | Dominio | Descripción |
|---|---|---|---|---|
{chr(10).join(filas)}

## Origen del dataset

Dataset **original y sintético**, generado con `data/generar_dataset.py`
(semilla {SEMILLA}, reproducible). Para cada raza se definieron distribuciones
basadas en rangos reales aproximados de peso, altura, longevidad, nivel de
actividad y tipo de cobertura corporal. Los rangos de razas parecidas se
traslapan a propósito para que la clasificación no sea trivial.

## Reglas de calidad aplicadas

1. Sin valores faltantes en ninguna columna.
2. `id` único y consecutivo.
3. Numéricas dentro de rango físico: peso {E.RANGOS['peso_kg']}, altura {E.RANGOS['altura_cm']}, edad {E.RANGOS['edad_anios']}.
4. Categóricas solo con valores del catálogo.
5. Coherencia raza → tipo (cada raza pertenece a un único tipo).
6. Coherencia tipo → pelaje (p. ej. un Ave solo tiene Plumas; una Tortuga, Caparazón).

## Razas / variedades (20)

| Tipo | Raza | Registros | Peso mediano (kg) | Altura mediana (cm) |
|---|---|---|---|---|
{razas_md}
"""
    (CARPETA / "diccionario_datos.md").write_text(texto, encoding="utf-8")


if __name__ == "__main__":
    print(f"Generando {N:,} registros…")
    datos = generar(N)
    datos.to_csv(SALIDA_CSV, index=False, encoding="utf-8")

    informe = validar_dataframe(pd.read_csv(SALIDA_CSV))
    resumen = {k: v for k, v in informe.items() if k != "df_valido"}
    (CARPETA / "reporte_calidad.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    escribir_diccionario(datos)

    print(f"OK  {SALIDA_CSV.name}: {len(datos):,} filas · "
          f"válidas {informe['filas_validas']:,} · rechazadas {informe['filas_rechazadas']:,}")
    print(datos["tipo"].value_counts().to_string())
