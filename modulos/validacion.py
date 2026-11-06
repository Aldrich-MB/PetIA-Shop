"""
Validación de datos PetIA Shop  (Integrante 1).

validar_dataframe(df)  -> valida un CSV completo (carga masiva).
validar_registro(dict) -> valida un registro del formulario "Añadir mascota".

Controles:
  1. Columnas obligatorias presentes.
  2. Valores faltantes (vacíos / NaN).
  3. Tipo de dato: numéricas convertibles a número; edad entera.
  4. Rangos físicos permitidos.
  5. Categorías dentro del catálogo (se normalizan mayúsculas y acentos).
  6. Coherencia raza -> tipo y tipo -> pelaje.
  7. Identificadores duplicados.
"""
import unicodedata

import numpy as np
import pandas as pd

from . import esquema as E

MAX_ERRORES_DETALLE = 200
OBLIGATORIAS = E.PREDICTORAS + E.ETIQUETAS


def _clave(texto) -> str:
    t = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return " ".join(t.lower().split())


def _mapa(valores):
    return {_clave(v): v for v in valores}


MAPA_CATEGORIAS = {
    "nivel_actividad": _mapa(E.NIVELES_ACTIVIDAD),
    "tipo_pelaje": _mapa(E.TIPOS_PELAJE),
    "tipo": _mapa(E.TIPOS),
    "raza": _mapa(E.RAZAS),
}
NOMBRES = {
    "peso_kg": "Peso (kg)", "altura_cm": "Altura (cm)", "edad_anios": "Edad (años)",
    "nivel_actividad": "Nivel de actividad", "tipo_pelaje": "Tipo de pelaje",
    "tipo": "Tipo de mascota", "raza": "Raza / variedad",
}


def validar_dataframe(df: pd.DataFrame) -> dict:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    filas = len(df)

    faltan = [c for c in OBLIGATORIAS if c not in df.columns]
    extra = [c for c in df.columns if c not in E.COLUMNAS]
    if faltan:
        return {
            "estado": "invalido",
            "mensaje": "Faltan columnas obligatorias: " + ", ".join(faltan),
            "columnas_faltantes": faltan, "columnas_extra": extra,
            "filas_leidas": filas, "filas_validas": 0, "filas_rechazadas": filas,
            "errores": [], "resumen_errores": {}, "faltantes_por_columna": {},
            "normalizados": 0, "df_valido": None,
        }

    malo = pd.Series(False, index=df.index)
    errores = []
    resumen = {}
    faltantes = {}
    normalizados = 0

    def marcar(mascara, columna, motivo):
        nonlocal malo
        n = int(mascara.sum())
        if not n:
            return
        resumen[motivo] = resumen.get(motivo, 0) + n
        if len(errores) < MAX_ERRORES_DETALLE:
            for i in df.index[mascara][: MAX_ERRORES_DETALLE - len(errores)]:
                v = df.at[i, columna] if columna in df.columns else ""
                errores.append({
                    "fila": int(i) + 2,              # +1 encabezado, +1 base 1
                    "columna": columna,
                    "valor": "" if pd.isna(v) else str(v),
                    "motivo": motivo,
                })
        malo = malo | mascara

    # 2. faltantes
    for c in OBLIGATORIAS:
        vacio = df[c].isna() | (df[c].astype(str).str.strip() == "")
        faltantes[c] = int(vacio.sum())
        marcar(vacio, c, f"Valor faltante en {c}")

    # 3-4. numéricas
    for c in E.NUMERICAS:
        texto = df[c].astype(str).str.strip().str.replace(",", ".", regex=False)
        num = pd.to_numeric(texto, errors="coerce")
        presente = ~(df[c].isna() | (df[c].astype(str).str.strip() == ""))
        marcar(presente & num.isna(), c, f"{c} no es numérico")
        lo, hi = E.RANGOS[c]
        marcar(num.notna() & ((num < lo) | (num > hi)), c,
               f"{c} fuera de rango [{lo}, {hi}]")
        if c == "edad_anios":
            marcar(num.notna() & (num % 1 != 0), c, "edad_anios debe ser entero")
        df[c] = num

    # 5. categóricas
    for c, mapa in MAPA_CATEGORIAS.items():
        original = df[c].astype(str).str.strip()
        # normalizar solo los valores únicos (rápido con 500,000 filas)
        traduccion = {u: mapa.get(_clave(u)) for u in original.unique()}
        canon = original.map(traduccion)
        presente = ~(df[c].isna() | (original == ""))
        marcar(presente & canon.isna(), c, f"{c} no pertenece al catálogo")
        normalizados += int((canon.notna() & (canon != original)).sum())
        df[c] = canon.where(canon.notna(), df[c])

    # 6. coherencia
    tipo_esperado = df["raza"].map({r: d["tipo"] for r, d in E.RAZAS.items()})
    marcar(tipo_esperado.notna() & df["tipo"].isin(E.TIPOS) & (tipo_esperado != df["tipo"]),
           "raza", "La raza no corresponde al tipo")
    pelaje_ok = pd.Series(
        [p in E.PELAJE_POR_TIPO.get(t, []) for t, p in zip(df["tipo"], df["tipo_pelaje"])],
        index=df.index)
    marcar(df["tipo"].isin(E.TIPOS) & df["tipo_pelaje"].isin(E.TIPOS_PELAJE) & ~pelaje_ok,
           "tipo_pelaje", "Pelaje incompatible con el tipo")

    # 7. ids duplicados
    if "id" in df.columns:
        ids = pd.to_numeric(df["id"], errors="coerce")
        marcar(ids.duplicated(keep="first") & ids.notna(), "id", "id duplicado")

    valido = df.loc[~malo, [c for c in E.COLUMNAS if c in df.columns]].copy()
    if len(valido):
        valido["edad_anios"] = valido["edad_anios"].astype(int)

    rech = int(malo.sum())
    estado = "ok" if rech == 0 else ("invalido" if rech == filas else "con_errores")
    return {
        "estado": estado,
        "mensaje": ("Archivo válido" if estado == "ok" else
                    f"{rech:,} filas rechazadas de {filas:,}"),
        "columnas_faltantes": [], "columnas_extra": extra,
        "filas_leidas": filas, "filas_validas": int(len(valido)),
        "filas_rechazadas": rech, "errores": sorted(errores, key=lambda e: e["fila"]),
        "resumen_errores": dict(sorted(resumen.items(), key=lambda x: -x[1])),
        "faltantes_por_columna": faltantes, "normalizados": normalizados,
        "df_valido": valido,
    }


def validar_registro(datos: dict) -> tuple:
    """Valida un registro del formulario.  Devuelve (ok, errores_por_campo, registro)."""
    errores = {}
    reg = {}

    for c in E.NUMERICAS:
        v = datos.get(c, "")
        if v is None or str(v).strip() == "":
            errores[c] = f"{NOMBRES[c]} es obligatorio."
            continue
        try:
            num = float(str(v).replace(",", "."))
        except ValueError:
            errores[c] = f"{NOMBRES[c]} debe ser un número."
            continue
        if not np.isfinite(num):
            errores[c] = f"{NOMBRES[c]} debe ser un número."
            continue
        lo, hi = E.RANGOS[c]
        if not lo <= num <= hi:
            errores[c] = f"{NOMBRES[c]} debe estar entre {lo:g} y {hi:g}."
            continue
        if c == "edad_anios" and num % 1:
            errores[c] = "La edad debe ser un número entero."
            continue
        reg[c] = int(num) if c == "edad_anios" else round(num, 3)

    for c, mapa in MAPA_CATEGORIAS.items():
        v = datos.get(c, "")
        if v is None or str(v).strip() == "":
            errores[c] = f"{NOMBRES[c]} es obligatorio."
            continue
        canon = mapa.get(_clave(v))
        if canon is None:
            errores[c] = f"'{v}' no es un valor válido de {NOMBRES[c]}."
        else:
            reg[c] = canon

    if "raza" in reg and "tipo" in reg and E.RAZAS[reg["raza"]]["tipo"] != reg["tipo"]:
        errores["raza"] = f"{reg['raza']} no es una raza de {reg['tipo']}."
    if "tipo" in reg and "tipo_pelaje" in reg and \
            reg["tipo_pelaje"] not in E.PELAJE_POR_TIPO[reg["tipo"]]:
        errores["tipo_pelaje"] = (f"Un(a) {reg['tipo']} no puede tener pelaje "
                                  f"'{reg['tipo_pelaje']}'. Opciones: "
                                  + ", ".join(E.PELAJE_POR_TIPO[reg["tipo"]]) + ".")

    return (not errores, errores, reg)
