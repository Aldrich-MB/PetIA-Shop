"""
Administración del dataset PetIA Shop  (Integrante 1).

GestorDataset mantiene el DataFrame en memoria y lo sincroniza con
data/mascotas.csv.  Los demás módulos (entrenamiento, K-means, Apriori)
deben pedir los datos con   gestor.obtener()   para trabajar siempre con
la versión actualizada (incluye registros añadidos desde el formulario).
"""
import io
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import esquema as E
from .validacion import validar_dataframe, validar_registro


class GestorDataset:
    def __init__(self, ruta_csv: Path):
        self.ruta = Path(ruta_csv)
        self._lock = threading.RLock()
        self.df = pd.DataFrame(columns=E.COLUMNAS)
        self.version = 0                 # sube cada vez que cambian los datos
        self.actualizado = None
        self.ultimo_informe = None
        self._cache_stats = None
        if self.ruta.exists():
            self.cargar_desde_disco()

    # ------------------------------------------------------------ lectura
    def cargar_desde_disco(self):
        with self._lock:
            informe = validar_dataframe(pd.read_csv(self.ruta, encoding="utf-8"))
            self.df = informe["df_valido"] if informe["df_valido"] is not None \
                else pd.DataFrame(columns=E.COLUMNAS)
            self._tocar()
            return informe

    def obtener(self) -> pd.DataFrame:
        """Copia del dataset actual (para entrenamiento, clustering, Apriori)."""
        with self._lock:
            return self.df.copy()

    def _tocar(self):
        self.version += 1
        self.actualizado = datetime.now().isoformat(timespec="seconds")
        self._cache_stats = None

    # ----------------------------------------------------- carga de CSV
    def importar_csv(self, contenido: bytes, modo: str = "anexar") -> dict:
        """modo='anexar' agrega las filas válidas; modo='reemplazar' sustituye el dataset."""
        try:
            texto = contenido.decode("utf-8-sig")
        except UnicodeDecodeError:
            texto = contenido.decode("latin-1")
        try:
            crudo = pd.read_csv(io.StringIO(texto), dtype=str, keep_default_na=False)
        except Exception as exc:  # noqa: BLE001
            return {"estado": "invalido", "mensaje": f"No se pudo leer el CSV: {exc}",
                    "filas_leidas": 0, "filas_validas": 0, "filas_rechazadas": 0,
                    "errores": [], "resumen_errores": {}, "faltantes_por_columna": {},
                    "columnas_faltantes": [], "columnas_extra": [], "normalizados": 0}
        crudo = crudo.replace("", pd.NA)
        informe = validar_dataframe(crudo)
        validos = informe.pop("df_valido")

        with self._lock:
            if validos is not None and len(validos):
                validos = validos[E.PREDICTORAS + E.ETIQUETAS].copy()
                if modo == "reemplazar":
                    validos.insert(0, "id", range(1, len(validos) + 1))
                    self.df = validos.reset_index(drop=True)
                else:
                    inicio = self._siguiente_id()
                    validos.insert(0, "id", range(inicio, inicio + len(validos)))
                    self.df = pd.concat([self.df, validos], ignore_index=True)
                self._guardar()
                self._tocar()
            informe["modo"] = modo
            informe["total_registros"] = int(len(self.df))
            self.ultimo_informe = informe
        return informe

    # --------------------------------------------------- alta individual
    def agregar_registro(self, datos: dict) -> tuple:
        ok, errores, reg = validar_registro(datos)
        if not ok:
            return False, errores, None
        with self._lock:
            reg["id"] = self._siguiente_id()
            fila = pd.DataFrame([reg])[E.COLUMNAS]
            self.df = pd.concat([self.df, fila], ignore_index=True)
            # anexar al CSV sin reescribir los 500,000 registros
            fila.to_csv(self.ruta, mode="a", header=not self.ruta.exists(),
                        index=False, encoding="utf-8")
            self._tocar()
        return True, {}, reg

    def _siguiente_id(self) -> int:
        return int(self.df["id"].max()) + 1 if len(self.df) else 1

    def _guardar(self):
        self.df.to_csv(self.ruta, index=False, encoding="utf-8")

    # ------------------------------------------------------- consultas
    def resumen(self) -> dict:
        with self._lock:
            if self._cache_stats is not None:
                return self._cache_stats
            df = self.df
            por_tipo = df["tipo"].value_counts().reindex(E.TIPOS, fill_value=0)
            por_raza = df["raza"].value_counts()
            numericas = {}
            for c in E.NUMERICAS:
                s = df[c].astype(float)
                numericas[c] = {k: round(float(v), 3) for k, v in {
                    "min": s.min(), "max": s.max(), "media": s.mean(),
                    "mediana": s.median(), "desv": s.std()}.items()} if len(s) else {}
            categoricas = {c: df[c].value_counts().to_dict() for c in E.CATEGORICAS}
            medias_tipo = (df.groupby("tipo")[E.NUMERICAS].mean().round(2)
                           .reindex(E.TIPOS).fillna(0).to_dict(orient="index")
                           if len(df) else {})
            self._cache_stats = {
                "total": int(len(df)),
                "n_tipos": int((por_tipo > 0).sum()),
                "n_razas": int(df["raza"].nunique()),
                "n_predictoras": len(E.PREDICTORAS),
                "por_tipo": {k: int(v) for k, v in por_tipo.items()},
                "por_raza": {r: int(por_raza.get(r, 0)) for t in E.TIPOS
                             for r in E.RAZAS_POR_TIPO[t]},
                "numericas": numericas,
                "categoricas": {c: {k: int(v) for k, v in d.items()}
                                for c, d in categoricas.items()},
                "medias_por_tipo": medias_tipo,
                "faltantes": {c: int(df[c].isna().sum()) for c in E.COLUMNAS},
                "version": self.version,
                "actualizado": self.actualizado,
            }
            return self._cache_stats

    def registros(self, pagina=1, por_pagina=15, tipo=None, raza=None,
                  buscar=None, orden="id", desc=True) -> dict:
        with self._lock:
            df = self.df
            if tipo:
                df = df[df["tipo"] == tipo]
            if raza:
                df = df[df["raza"] == raza]
            if buscar:
                b = str(buscar).strip()
                if b.isdigit():
                    df = df[df["id"] == int(b)]
                else:
                    df = df[df["raza"].str.contains(b, case=False, regex=False)]
            if orden in E.COLUMNAS:
                df = df.sort_values(orden, ascending=not desc, kind="stable")
            total = int(len(df))
            por_pagina = max(5, min(int(por_pagina), 100))
            paginas = max(1, -(-total // por_pagina))
            pagina = max(1, min(int(pagina), paginas))
            trozo = df.iloc[(pagina - 1) * por_pagina: pagina * por_pagina]
            return {"total": total, "pagina": pagina, "paginas": paginas,
                    "por_pagina": por_pagina,
                    "filas": trozo.astype(object).where(trozo.notna(), None)
                                  .to_dict(orient="records")}

    def perfil_raza(self, raza: str) -> dict:
        """Valores típicos de una raza (para autocompletar el formulario)."""
        with self._lock:
            sub = self.df[self.df["raza"] == raza]
            if not len(sub):
                return {}
            return {
                "raza": raza, "tipo": E.RAZAS[raza]["tipo"], "n": int(len(sub)),
                "peso_kg": round(float(sub["peso_kg"].median()), 3),
                "altura_cm": round(float(sub["altura_cm"].median()), 1),
                "edad_anios": int(sub["edad_anios"].median()),
                "nivel_actividad": sub["nivel_actividad"].mode().iat[0],
                "tipo_pelaje": sub["tipo_pelaje"].mode().iat[0],
                "rangos": {c: [round(float(sub[c].quantile(.05)), 3),
                               round(float(sub[c].quantile(.95)), 3)] for c in E.NUMERICAS},
            }
