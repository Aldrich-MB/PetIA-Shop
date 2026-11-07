from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Importaciones de Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
# Algoritmos de aprendizaje no supervisado (Integrante 3)
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.base import clone

# IMPORTAMOS LOS MÓDULOS DEL INTEGRANTE 1
from modulos import esquema as E
from modulos.datos import GestorDataset

app = Flask(__name__)

# Inicializar el gestor del dataset (asegúrate de que la ruta sea correcta)
RUTA_DATASET = Path("dataset/mascotas.csv")
gestor = GestorDataset(RUTA_DATASET)

# Variables globales para guardar modelos en memoria
modelos_entrenados = {}

# Número máximo de puntos que se envían a la gráfica de K-means.
# El modelo se entrena con TODOS los registros; este límite solo afecta la
# visualización para no saturar el navegador con 500 000 puntos.
MAX_PUNTOS_GRAFICA = 5000

# ==========================================
# RUTAS BÁSICAS Y DE DATASET (INTEGRANTE 1)
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/esquema', methods=['GET'])
def api_esquema():
    return jsonify(E.esquema_publico())

@app.route('/api/dataset/resumen', methods=['GET'])
def api_resumen():
    return jsonify(gestor.resumen())

@app.route('/api/dataset/registros', methods=['GET'])
def api_registros():
    pagina = int(request.args.get('pagina', 1))
    por_pagina = int(request.args.get('por_pagina', 15))
    tipo = request.args.get('tipo', '')
    raza = request.args.get('raza', '')
    buscar = request.args.get('buscar', '')
    orden = request.args.get('orden', 'id')
    desc = request.args.get('desc', '1') == '1'

    return jsonify(gestor.registros(pagina, por_pagina, tipo, raza, buscar, orden, desc))

@app.route('/api/dataset/perfil/<raza>', methods=['GET'])
def api_perfil(raza):
    return jsonify(gestor.perfil_raza(raza))

@app.route('/api/dataset/registro', methods=['POST'])
def api_agregar_registro():
    datos = request.json
    ok, errores, reg = gestor.agregar_registro(datos)
    if not ok:
        return jsonify({"mensaje": "Error de validación", "errores": errores}), 400
    return jsonify({"mensaje": "Mascota añadida exitosamente", "registro": reg})

# ==========================================
# LÓGICA DE ENTRENAMIENTO (INTEGRANTE 2)
# ==========================================
@app.route('/api/entrenar', methods=['POST'])
def api_entrenar():
    try:
        # 1. Obtener los datos más recientes del Gestor (en memoria)
        df = gestor.obtener()

        if len(df) < 100:
            return jsonify({"mensaje": "No hay suficientes datos para entrenar. Carga un CSV."}), 400

        # 2. Preparar Variables (X) y etiquetas (y)
        features = E.PREDICTORAS
        X = df[features].copy()

        y_tipo = df["tipo"].copy()
        y_raza = df["raza"].copy()

        # Codificar variables categóricas
        le_actividad = LabelEncoder()
        le_pelaje = LabelEncoder()
        le_tipo = LabelEncoder()
        le_raza = LabelEncoder()

        X['nivel_actividad'] = le_actividad.fit_transform(X['nivel_actividad'])
        X['tipo_pelaje'] = le_pelaje.fit_transform(X['tipo_pelaje'])

        y_tipo_encoded = le_tipo.fit_transform(y_tipo)
        y_raza_encoded = le_raza.fit_transform(y_raza)

        # 3. División 75% Entrenamiento, 15% Validación, 10% Prueba
        # Primero sacamos el 10% para prueba
        X_temp, X_test, y_tipo_temp, y_tipo_test, y_raza_temp, y_raza_test = train_test_split(
            X,
            y_tipo_encoded,
            y_raza_encoded,
            test_size=0.10,
            random_state=42,
            stratify=y_tipo_encoded
        )

        X_train, X_val, y_tipo_train, y_tipo_val, y_raza_train, y_raza_val = train_test_split(
            X_temp,
            y_tipo_temp,
            y_raza_temp,
            test_size=0.1666667,
            random_state=42,
            stratify=y_tipo_temp
        )

        # Escalamiento (SIN FUGAS: fit solo con Train)
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc = scaler.transform(X_val)
        X_test_sc = scaler.transform(X_test)

        # 4. Entrenar los 3 clasificadores para Tipo y Raza
        modelos_base = {
            'GaussianNB': GaussianNB(),
            'Árbol de Decisión': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(
                random_state=42,
                n_estimators=100
            )
        }

        resultados_modelos = []
        mejores_modelos = {}
        mejores_predicciones = {}

        # Entrenar primero la clasificación de Tipo y después la de Raza
        for objetivo, y_train_obj, y_val_obj, y_test_obj in [
            ("tipo", y_tipo_train, y_tipo_val, y_tipo_test),
            ("raza", y_raza_train, y_raza_val, y_raza_test)
        ]:

            # [CORRECCIÓN 1] Etiquetas completas del objetivo para métricas macro
            le_objetivo = le_tipo if objetivo == "tipo" else le_raza
            etiquetas = np.arange(len(le_objetivo.classes_))

            mejor_f1_val = -1
            mejor_nombre = None
            mejor_modelo = None
            mejor_pred_test = None

            for nombre, modelo_base in modelos_base.items():

                modelo = clone(modelo_base)
                modelo.fit(X_train_sc, y_train_obj)

                pred_val = modelo.predict(X_val_sc)
                pred_test = modelo.predict(X_test_sc)

                f1_val = f1_score(
                    y_val_obj,
                    pred_val,
                    average="macro"
                )

                acc_test = accuracy_score(y_test_obj, pred_test)
                error_total = 1.0 - acc_test

                cm = confusion_matrix(
                    y_test_obj,
                    pred_test,
                    labels=etiquetas
                )

                f1_test = f1_score(
                    y_test_obj,
                    pred_test,
                    labels=etiquetas,
                    average="macro",
                    zero_division=0
                )

                fpr_por_clase = []
                fnr_por_clase = []
                total = np.sum(cm)

                for i in range(len(etiquetas)):
                    vp = cm[i, i]
                    fp = np.sum(cm[:, i]) - vp
                    fn = np.sum(cm[i, :]) - vp
                    vn = total - vp - fp - fn

                    fpr = fp / (fp + vn) if (fp + vn) > 0 else 0.0
                    fnr = fn / (fn + vp) if (fn + vp) > 0 else 0.0

                    fpr_por_clase.append(fpr)
                    fnr_por_clase.append(fnr)

                fpr_macro = float(np.mean(fpr_por_clase)) if fpr_por_clase else 0.0
                fnr_macro = float(np.mean(fnr_por_clase)) if fnr_por_clase else 0.0

                resultados_modelos.append({
                    "nombre": nombre,
                    "objetivo": objetivo,
                    "accuracy": float(acc_test),
                    "error_total": float(error_total),
                    "fpr_macro": fpr_macro,
                    "fnr_macro": fnr_macro,
                    "f1_macro": float(f1_test),
                    "f1_validacion": float(f1_val)
                })

                if f1_val > mejor_f1_val:
                    mejor_f1_val = f1_val
                    mejor_nombre = nombre
                    mejor_modelo = modelo
                    mejor_pred_test = pred_test

            mejores_modelos[objetivo] = {
                "nombre": mejor_nombre,
                "modelo": mejor_modelo
            }

            mejores_predicciones[objetivo] = mejor_pred_test
            modelos_entrenados[objetivo] = mejor_modelo

        # Guardar transformadores y modelos en disco
        os.makedirs('models', exist_ok=True)

        joblib.dump(scaler, 'models/scaler.pkl')
        joblib.dump(le_actividad, 'models/le_actividad.pkl')
        joblib.dump(le_pelaje, 'models/le_pelaje.pkl')
        joblib.dump(le_tipo, 'models/le_tipo.pkl')
        joblib.dump(le_raza, 'models/le_raza.pkl')

        joblib.dump(
            mejores_modelos["tipo"]["modelo"],
            'models/modelo_tipo.pkl'
        )
        joblib.dump(
            mejores_modelos["raza"]["modelo"],
            'models/modelo_raza.pkl'
        )

        # Generar las matrices de confusión (ya usan labels completos)
        cm_tipo = confusion_matrix(
            y_tipo_test,
            mejores_predicciones["tipo"],
            labels=np.arange(len(le_tipo.classes_))
        )

        cm_raza = confusion_matrix(
            y_raza_test,
            mejores_predicciones["raza"],
            labels=np.arange(len(le_raza.classes_))
        )

        matrices_front = {
            "tipo": {
                "etiquetas": le_tipo.classes_.tolist(),
                "valores": cm_tipo.tolist()
            },
            "raza": {
                "etiquetas": le_raza.classes_.tolist(),
                "valores": cm_raza.tolist()
            }
        }

        metricas_tipo = next(
            m for m in resultados_modelos
            if m["objetivo"] == "tipo"
            and m["nombre"] == mejores_modelos["tipo"]["nombre"]
        )

        metricas_raza = next(
            m for m in resultados_modelos
            if m["objetivo"] == "raza"
            and m["nombre"] == mejores_modelos["raza"]["nombre"]
        )

        return jsonify({
            "status": "success",
            "mejor": mejores_modelos["tipo"]["nombre"],
            "mejor_raza": mejores_modelos["raza"]["nombre"],
            "division": {
                "train": len(X_train),
                "val": len(X_val),
                "test": len(X_test)
            },
            "modelos": [
                m for m in resultados_modelos
                if m["objetivo"] == "tipo"
            ],
            "modelos_raza": [
                m for m in resultados_modelos
                if m["objetivo"] == "raza"
            ],
            "test": metricas_tipo,
            "test_raza": metricas_raza,
            "matrices": matrices_front,
            "interpretacion": (
                f"Tipo: {mejores_modelos['tipo']['nombre']} fue elegido "
                f"por su mejor F1 en validación. "
                f"Raza: {mejores_modelos['raza']['nombre']} fue elegido "
                f"por su mejor F1 en validación. "
                "Ambos se evaluaron en el conjunto de prueba."
            )
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"mensaje": f"Error interno al entrenar: {str(e)}"}), 500


# ==========================================
# LÓGICA DE CLASIFICACIÓN (INTEGRANTE 2)
# ==========================================

@app.route('/api/clasificar', methods=['POST'])
def api_clasificar():
    try:
        if not modelos_entrenados:
            return jsonify({
                "mensaje": "Primero debes reentrenar los modelos en la pestaña de Entrenamiento."
            }), 400

        datos = request.json

        scaler = joblib.load('models/scaler.pkl')
        le_actividad = joblib.load('models/le_actividad.pkl')
        le_pelaje = joblib.load('models/le_pelaje.pkl')
        le_tipo = joblib.load('models/le_tipo.pkl')
        le_raza = joblib.load('models/le_raza.pkl')

        peso = float(datos['peso_kg'])
        altura = float(datos['altura_cm'])
        edad = float(datos['edad_anios'])

        actividad = le_actividad.transform([datos['nivel_actividad']])[0]
        pelaje = le_pelaje.transform([datos['tipo_pelaje']])[0]

        X_nuevo = np.array([[peso, altura, edad, actividad, pelaje]])
        X_nuevo_sc = scaler.transform(X_nuevo)

        modelo_tipo = modelos_entrenados.get("tipo")
        modelo_raza = modelos_entrenados.get("raza")

        if modelo_tipo is None or modelo_raza is None:
            return jsonify({
                "mensaje": "Los clasificadores no están disponibles. Ejecuta nuevamente el entrenamiento."
            }), 400

        tipo_encoded = modelo_tipo.predict(X_nuevo_sc)[0]
        tipo_predicho = le_tipo.inverse_transform([tipo_encoded])[0]

        probabilidades_tipo = modelo_tipo.predict_proba(X_nuevo_sc)[0]
        prob_tipo = {
            clase: float(prob)
            for clase, prob in zip(modelo_tipo.classes_, probabilidades_tipo)
        }

        raza_encoded = modelo_raza.predict(X_nuevo_sc)[0]
        raza_predicha = le_raza.inverse_transform([raza_encoded])[0]

        probabilidades_raza = modelo_raza.predict_proba(X_nuevo_sc)[0]
        prob_raza = {
            le_raza.inverse_transform([clase])[0]: float(prob)
            for clase, prob in zip(modelo_raza.classes_, probabilidades_raza)
        }

        imagen_url = f"/static/img/razas/{str(raza_predicha).lower()}.jpg"

        return jsonify({
            "tipo": tipo_predicho,
            "raza": raza_predicha,
            "prob_tipo": prob_tipo,
            "prob_raza": prob_raza,
            "modelo": "Clasificadores seleccionados por F1 de validación",
            "imagen": imagen_url
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"mensaje": f"Error al clasificar: {str(e)}"}), 500


# ==========================================
# RUTAS DE ANALISIS NO SUPERVISADO (INTEGRANTE 3)
# ==========================================

@app.route('/api/agrupar', methods=['POST'])
def api_agrupar():
    try:
        df = gestor.obtener()

        if df.empty:
            return jsonify({"mensaje": "El dataset está vacío."}), 400

        datos = request.get_json(silent=True) or {}
        k = int(datos.get("k", 3))

        if k < 2:
            return jsonify({"mensaje": "El número de grupos debe ser al menos 2."}), 400

        columnas = E.PREDICTORAS
        X = df[columnas].copy()

        for col in E.NUMERICAS:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        X = X.dropna(subset=columnas)

        if len(X) < k:
            return jsonify({
                "mensaje": "No hay suficientes registros válidos para formar los grupos."
            }), 400

        # Codificar variables categóricas
        X_codificado = pd.DataFrame(index=X.index)

        for col in E.NUMERICAS:
            X_codificado[col] = X[col]

        mapa_actividad = {
            nivel: i for i, nivel in enumerate(E.NIVELES_ACTIVIDAD)
        }

        X_codificado["nivel_actividad"] = (
            X["nivel_actividad"].map(mapa_actividad)
        )

        pelaje = pd.get_dummies(
            X["tipo_pelaje"],
            prefix="pelaje",
            dtype=float
        )

        X_codificado = pd.concat([X_codificado, pelaje], axis=1)

        if X_codificado.isna().any().any():
            return jsonify({
                "mensaje": "Existen categorías no reconocidas en las variables categóricas."
            }), 400

        # [CORRECCIÓN 2] Estandarizar TODAS las variables utilizadas por K-means
        scaler = StandardScaler()
        X_codificado = pd.DataFrame(
            scaler.fit_transform(X_codificado),
            columns=X_codificado.columns,
            index=X_codificado.index
        )

        # Aplicar K-Means (entrenamiento con TODOS los registros)
        modelo = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        grupos = modelo.fit_predict(X_codificado)

        # Preparar los puntos para el frontend
        # (proyección sobre peso y altura, no las dimensiones completas del modelo)
        puntos = []
        for posicion, grupo in enumerate(grupos):
            puntos.append([
                round(float(X.iloc[posicion]["peso_kg"]), 3),
                round(float(X.iloc[posicion]["altura_cm"]), 3),
                int(grupo)
            ])

        # [RECOMENDACIÓN 4] Submuestreo SOLO para la gráfica.
        # El modelo se entrenó con todos los registros; aquí reducimos lo que
        # viaja al navegador para evitar respuestas JSON gigantes y gráficas lentas.
        n_total = len(puntos)
        if n_total > MAX_PUNTOS_GRAFICA:
            rng = np.random.default_rng(42)
            indices = rng.choice(n_total, size=MAX_PUNTOS_GRAFICA, replace=False)
            puntos_grafica = [puntos[i] for i in indices]
        else:
            puntos_grafica = puntos

        # [RECOMENDACIÓN 3] Centroides reales del modelo (no promedios por grupo).
        # Se deshace la estandarización para devolverlos en unidades originales.
        centros_originales = scaler.inverse_transform(
            modelo.cluster_centers_.copy()
        )

        columnas_codificadas = list(X_codificado.columns)
        indice_peso = columnas_codificadas.index("peso_kg")
        indice_altura = columnas_codificadas.index("altura_cm")

        centroides = []
        for grupo in range(k):
            centroides.append({
                "grupo": int(grupo),
                "x": round(float(centros_originales[grupo, indice_peso]), 3),
                "y": round(float(centros_originales[grupo, indice_altura]), 3)
            })

        # Interpretación: se calcula con TODOS los registros (no con la muestra),
        # porque describe los grupos reales del modelo.
        X_resultados = X.copy()
        X_resultados["grupo"] = grupos

        interpretacion = []
        for grupo in range(k):
            registros_grupo = X_resultados[X_resultados["grupo"] == grupo]

            if len(registros_grupo) == 0:
                interpretacion.append(
                    f"Grupo {grupo}: no contiene registros en esta ejecución."
                )
                continue

            peso = registros_grupo["peso_kg"].mean()
            altura = registros_grupo["altura_cm"].mean()
            edad = registros_grupo["edad_anios"].mean()

            actividad = registros_grupo["nivel_actividad"].mode().iloc[0]
            pelaje = registros_grupo["tipo_pelaje"].mode().iloc[0]

            cantidad = len(registros_grupo)

            interpretacion.append(
                f"Grupo {grupo}: contiene {cantidad} mascotas. "
                f"Peso promedio: {peso:.2f} kg, "
                f"altura promedio: {altura:.2f} cm, "
                f"edad promedio: {edad:.1f} años. "
                f"Actividad predominante: {actividad}. "
                f"Pelaje predominante: {pelaje}."
            )

        return jsonify({
            "k": int(k),
            "ejes": {"x": "peso_kg", "y": "altura_cm"},
            "puntos": puntos_grafica,
            "n_total": n_total,
            "n_mostrados": len(puntos_grafica),
            "centroides": centroides,
            "interpretacion": interpretacion
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"mensaje": f"Error al ejecutar K-Means: {str(e)}"}), 500


@app.route('/api/asociar', methods=['POST'])
def api_asociar():
    try:
        df = gestor.obtener()

        if df.empty:
            return jsonify({"mensaje": "El dataset está vacío."}), 400

        datos = request.get_json(silent=True) or {}
        min_support = float(datos.get("min_support", 0.05))
        min_confidence = float(datos.get("min_confidence", 0.5))

        if not 0 < min_support <= 1:
            return jsonify({"mensaje": "El soporte mínimo debe estar entre 0 y 1."}), 400

        if not 0 < min_confidence <= 1:
            return jsonify({"mensaje": "La confianza mínima debe estar entre 0 y 1."}), 400

        columnas = E.PREDICTORAS + ["tipo"]
        X = df[columnas].copy()

        for col in E.NUMERICAS:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        X = X.dropna(subset=columnas)

        if len(X) == 0:
            return jsonify({
                "mensaje": "No existen registros válidos para analizar."
            }), 400

        # Discretizar variables numéricas usando la mediana
        transacciones = pd.DataFrame(index=X.index)

        peso_mediana = X["peso_kg"].median()
        altura_mediana = X["altura_cm"].median()
        edad_mediana = X["edad_anios"].median()

        transacciones["peso"] = np.where(
            X["peso_kg"] >= peso_mediana, "Peso_Alto", "Peso_Bajo"
        )
        transacciones["altura"] = np.where(
            X["altura_cm"] >= altura_mediana, "Altura_Alta", "Altura_Baja"
        )
        transacciones["edad"] = np.where(
            X["edad_anios"] >= edad_mediana, "Edad_Alta", "Edad_Baja"
        )

        transacciones["actividad"] = (
            "Actividad_" + X["nivel_actividad"].astype(str)
        )
        transacciones["pelaje"] = (
            "Pelaje_" + X["tipo_pelaje"].astype(str)
        )
        transacciones["tipo"] = (
            "Tipo_" + X["tipo"].astype(str)
        )

        matriz = pd.get_dummies(transacciones, dtype=bool)

        conjuntos_frecuentes = apriori(
            matriz,
            min_support=min_support,
            use_colnames=True,
            max_len=3
        )

        # [CORRECCIÓN 3] Proteger Apriori cuando no existan itemsets de ≥ 2 elementos
        if conjuntos_frecuentes.empty or not any(
            len(conjunto) >= 2
            for conjunto in conjuntos_frecuentes["itemsets"]
        ):
            reglas = []
            reglas_df = pd.DataFrame()
        else:
            reglas_df = association_rules(
                conjuntos_frecuentes,
                metric="confidence",
                min_threshold=min_confidence
            )

            if not reglas_df.empty:
                reglas_df = reglas_df.sort_values(
                    by=["confidence", "lift"],
                    ascending=False
                )
                reglas_df = reglas_df.head(100)

            reglas = []
            for _, fila in reglas_df.iterrows():
                reglas.append({
                    "antecedente": sorted(list(fila["antecedents"])),
                    "consecuente": sorted(list(fila["consequents"])),
                    "soporte": round(float(fila["support"]), 4),
                    "confianza": round(float(fila["confidence"]), 4),
                    "lift": round(float(fila["lift"]), 4)
                })

        if reglas:
            mejor = reglas[0]
            interpretacion = (
                f"Se analizaron {len(matriz)} transacciones y "
                f"se encontraron {len(reglas)} reglas de asociación. "
                f"La regla con mayor confianza relaciona "
                f"{', '.join(mejor['antecedente'])} con "
                f"{', '.join(mejor['consecuente'])}. "
                f"Presenta un soporte de {mejor['soporte']:.2%}, "
                f"una confianza de {mejor['confianza']:.2%} "
                f"y un lift de {mejor['lift']:.2f}."
            )
        else:
            interpretacion = (
                "Se analizaron las transacciones, pero no se encontraron "
                "reglas que cumplan los umbrales de soporte y confianza "
                "establecidos. Puedes reducir min_support o min_confidence."
            )

        return jsonify({
            "n_transacciones": int(len(matriz)),
            "reglas": reglas,
            "interpretacion": interpretacion
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"mensaje": f"Error al ejecutar Apriori: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)