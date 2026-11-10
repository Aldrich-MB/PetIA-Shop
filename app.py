from flask import Flask, render_template, request, jsonify, send_file
import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import io

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

# Inicializar el gestor del dataset
RUTA_DATASET = Path("dataset/mascotas.csv")
gestor = GestorDataset(RUTA_DATASET)

# Variables globales para guardar modelos en memoria
modelos_entrenados = {}
MAX_PUNTOS_GRAFICA = 5000

# ==========================================
# RUTAS BÁSICAS Y DE DATASET (INTEGRANTE 1 Y 2)
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

# --- RUTAS FALTANTES PARA LOS BOTONES DE DESCARGA ---
@app.route('/api/dataset/descargar', methods=['GET'])
def api_descargar_dataset():
    try:
        ruta = Path("dataset/mascotas.csv")
        if ruta.exists():
            return send_file(ruta, as_attachment=True, download_name="mascotas.csv")
        return jsonify({"mensaje": "El archivo no existe en el servidor"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/dataset/plantilla', methods=['GET'])
def api_descargar_plantilla():
    try:
        columnas = ["id", "peso_kg", "altura_cm", "edad_anios", "nivel_actividad", "tipo_pelaje", "tipo", "raza"]
        datos_ejemplo = [
            [1, 2.5, 20.0, 2, "Alta", "Corto", "Perro", "Chihuahua"],
            [2, 4.0, 25.0, 3, "Media", "Largo", "Gato", "Persa"]
        ]
        df_plantilla = pd.DataFrame(datos_ejemplo, columns=columnas)
        
        # Guardar en memoria (buffer) en lugar de en disco para ser más limpio
        output = io.BytesIO()
        df_plantilla.to_csv(output, index=False, encoding='utf-8')
        output.seek(0)
        
        return send_file(output, mimetype='text/csv', as_attachment=True, download_name="plantilla_mascotas.csv")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- RUTA FALTANTE PARA EL BOTÓN "VALIDAR Y CARGAR" ---
@app.route('/api/dataset/cargar', methods=['POST'])
def api_cargar_csv():
    if 'archivo' not in request.files:
        return jsonify({"estado": "invalido", "mensaje": "No se envió ningún archivo"}), 400
    
    archivo = request.files['archivo']
    modo = request.form.get('modo', 'anexar')
    
    if archivo.filename == '':
        return jsonify({"estado": "invalido", "mensaje": "Archivo vacío"}), 400

    try:
        contenido = archivo.read()
        # Usar la lógica del GestorDataset del Integrante 1
        informe = gestor.importar_csv(contenido, modo=modo)
        
        # El frontend del I1 espera estos campos en el informe
        if informe["estado"] != "invalido" and informe["filas_validas"] > 0:
            return jsonify(informe), 200
        else:
            return jsonify(informe), 400
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"estado": "invalido", "mensaje": f"Error interno: {str(e)}"}), 500


# ==========================================
# LÓGICA DE ENTRENAMIENTO (INTEGRANTE 2)
# ==========================================
@app.route('/api/entrenar', methods=['POST'])
def api_entrenar():
    try:
        df = gestor.obtener()
        if len(df) < 100:
            return jsonify({"mensaje": "No hay suficientes datos para entrenar."}), 400

        features = E.PREDICTORAS
        X = df[features].copy()
        y_tipo = df["tipo"].copy()
        y_raza = df["raza"].copy()

        le_actividad = LabelEncoder()
        le_pelaje = LabelEncoder()
        le_tipo = LabelEncoder()
        le_raza = LabelEncoder()

        X['nivel_actividad'] = le_actividad.fit_transform(X['nivel_actividad'])
        X['tipo_pelaje'] = le_pelaje.fit_transform(X['tipo_pelaje'])
        y_tipo_encoded = le_tipo.fit_transform(y_tipo)
        y_raza_encoded = le_raza.fit_transform(y_raza)

        X_temp, X_test, y_tipo_temp, y_tipo_test, y_raza_temp, y_raza_test = train_test_split(
            X, y_tipo_encoded, y_raza_encoded, test_size=0.10, random_state=42, stratify=y_tipo_encoded
        )

        X_train, X_val, y_tipo_train, y_tipo_val, y_raza_train, y_raza_val = train_test_split(
            X_temp, y_tipo_temp, y_raza_temp, test_size=0.1666667, random_state=42, stratify=y_tipo_temp
        )

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc = scaler.transform(X_val)
        X_test_sc = scaler.transform(X_test)

        modelos_base = {
            'GaussianNB': GaussianNB(),
            'Árbol de Decisión': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }

        resultados_modelos = []
        mejores_modelos = {}
        mejores_predicciones = {}

        for objetivo, y_train_obj, y_val_obj, y_test_obj in [
            ("tipo", y_tipo_train, y_tipo_val, y_tipo_test),
            ("raza", y_raza_train, y_raza_val, y_raza_test)
        ]:
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

                f1_val = f1_score(y_val_obj, pred_val, average="macro")
                acc_test = accuracy_score(y_test_obj, pred_test)
                error_total = 1.0 - acc_test

                cm = confusion_matrix(y_test_obj, pred_test, labels=etiquetas)
                f1_test = f1_score(y_test_obj, pred_test, labels=etiquetas, average="macro", zero_division=0)

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

            mejores_modelos[objetivo] = {"nombre": mejor_nombre, "modelo": mejor_modelo}
            mejores_predicciones[objetivo] = mejor_pred_test
            modelos_entrenados[objetivo] = mejor_modelo

        os.makedirs('models', exist_ok=True)
        joblib.dump(scaler, 'models/scaler.pkl')
        joblib.dump(le_actividad, 'models/le_actividad.pkl')
        joblib.dump(le_pelaje, 'models/le_pelaje.pkl')
        joblib.dump(le_tipo, 'models/le_tipo.pkl')
        joblib.dump(le_raza, 'models/le_raza.pkl')
        joblib.dump(mejores_modelos["tipo"]["modelo"], 'models/modelo_tipo.pkl')
        joblib.dump(mejores_modelos["raza"]["modelo"], 'models/modelo_raza.pkl')

        cm_tipo = confusion_matrix(y_tipo_test, mejores_predicciones["tipo"], labels=np.arange(len(le_tipo.classes_)))
        cm_raza = confusion_matrix(y_raza_test, mejores_predicciones["raza"], labels=np.arange(len(le_raza.classes_)))

        matrices_front = {
            "tipo": {"etiquetas": le_tipo.classes_.tolist(), "valores": cm_tipo.tolist()},
            "raza": {"etiquetas": le_raza.classes_.tolist(), "valores": cm_raza.tolist()}
        }

        metricas_tipo = next(m for m in resultados_modelos if m["objetivo"] == "tipo" and m["nombre"] == mejores_modelos["tipo"]["nombre"])
        metricas_raza = next(m for m in resultados_modelos if m["objetivo"] == "raza" and m["nombre"] == mejores_modelos["raza"]["nombre"])

        return jsonify({
            "status": "success",
            "mejor": mejores_modelos["tipo"]["nombre"],
            "mejor_raza": mejores_modelos["raza"]["nombre"],
            "division": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
            "modelos": [m for m in resultados_modelos if m["objetivo"] == "tipo"],
            "modelos_raza": [m for m in resultados_modelos if m["objetivo"] == "raza"],
            "test": metricas_tipo,
            "test_raza": metricas_raza,
            "matrices": matrices_front,
            "interpretacion": f"Tipo: {mejores_modelos['tipo']['nombre']} fue elegido por su mejor F1. Raza: {mejores_modelos['raza']['nombre']} fue elegido por su mejor F1."
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
            return jsonify({"mensaje": "Primero debes reentrenar los modelos."}), 400

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

        tipo_encoded = modelo_tipo.predict(X_nuevo_sc)[0]
        tipo_predicho = le_tipo.inverse_transform([tipo_encoded])[0]

        probabilidades_tipo = modelo_tipo.predict_proba(X_nuevo_sc)[0]
        prob_tipo = {clase: float(prob) for clase, prob in zip(modelo_tipo.classes_, probabilidades_tipo)}

        raza_encoded = modelo_raza.predict(X_nuevo_sc)[0]
        raza_predicha = le_raza.inverse_transform([raza_encoded])[0]

        probabilidades_raza = modelo_raza.predict_proba(X_nuevo_sc)[0]
        prob_raza = {le_raza.inverse_transform([clase])[0]: float(prob) for clase, prob in zip(modelo_raza.classes_, probabilidades_raza)}

        imagen_url = f"/static/img/razas/{str(raza_predicha).lower()}.jpg"

        return jsonify({
            "tipo": tipo_predicho,
            "raza": raza_predicha,
            "prob_tipo": prob_tipo,
            "prob_raza": prob_raza,
            "modelo": "Clasificadores seleccionados",
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
            return jsonify({"mensaje": "No hay suficientes registros."}), 400

        X_codificado = pd.DataFrame(index=X.index)
        for col in E.NUMERICAS:
            X_codificado[col] = X[col]

        mapa_actividad = {nivel: i for i, nivel in enumerate(E.NIVELES_ACTIVIDAD)}
        X_codificado["nivel_actividad"] = X["nivel_actividad"].map(mapa_actividad)

        pelaje = pd.get_dummies(X["tipo_pelaje"], prefix="pelaje", dtype=float)
        X_codificado = pd.concat([X_codificado, pelaje], axis=1)

        scaler = StandardScaler()
        X_codificado = pd.DataFrame(scaler.fit_transform(X_codificado), columns=X_codificado.columns, index=X_codificado.index)

        modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
        grupos = modelo.fit_predict(X_codificado)

        puntos = []
        for posicion, grupo in enumerate(grupos):
            puntos.append([round(float(X.iloc[posicion]["peso_kg"]), 3), round(float(X.iloc[posicion]["altura_cm"]), 3), int(grupo)])

        n_total = len(puntos)
        if n_total > MAX_PUNTOS_GRAFICA:
            rng = np.random.default_rng(42)
            indices = rng.choice(n_total, size=MAX_PUNTOS_GRAFICA, replace=False)
            puntos_grafica = [puntos[i] for i in indices]
        else:
            puntos_grafica = puntos

        centros_originales = scaler.inverse_transform(modelo.cluster_centers_.copy())
        columnas_codificadas = list(X_codificado.columns)
        indice_peso = columnas_codificadas.index("peso_kg")
        indice_altura = columnas_codificadas.index("altura_cm")

        centroides = []
        for grupo in range(k):
            centroides.append({"grupo": int(grupo), "x": round(float(centros_originales[grupo, indice_peso]), 3), "y": round(float(centros_originales[grupo, indice_altura]), 3)})

        X_resultados = X.copy()
        X_resultados["grupo"] = grupos

        interpretacion = []
        for grupo in range(k):
            regs = X_resultados[X_resultados["grupo"] == grupo]
            if len(regs) == 0:
                interpretacion.append(f"Grupo {grupo}: vacío.")
                continue
            interpretacion.append(f"Grupo {grupo}: {len(regs)} mascotas. Peso: {regs['peso_kg'].mean():.2f}kg, Altura: {regs['altura_cm'].mean():.2f}cm, Edad: {regs['edad_anios'].mean():.1f}años.")

        return jsonify({"k": int(k), "ejes": {"x": "peso_kg", "y": "altura_cm"}, "puntos": puntos_grafica, "n_total": n_total, "n_mostrados": len(puntos_grafica), "centroides": centroides, "interpretacion": interpretacion})

    except Exception as e:
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

        columnas = E.PREDICTORAS + ["tipo"]
        X = df[columnas].copy()
        for col in E.NUMERICAS:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        X = X.dropna(subset=columnas)

        transacciones = pd.DataFrame(index=X.index)
        transacciones["peso"] = np.where(X["peso_kg"] >= X["peso_kg"].median(), "Peso_Alto", "Peso_Bajo")
        transacciones["altura"] = np.where(X["altura_cm"] >= X["altura_cm"].median(), "Altura_Alta", "Altura_Baja")
        transacciones["edad"] = np.where(X["edad_anios"] >= X["edad_anios"].median(), "Edad_Alta", "Edad_Baja")
        transacciones["actividad"] = "Actividad_" + X["nivel_actividad"].astype(str)
        transacciones["pelaje"] = "Pelaje_" + X["tipo_pelaje"].astype(str)
        transacciones["tipo"] = "Tipo_" + X["tipo"].astype(str)

        matriz = pd.get_dummies(transacciones, dtype=bool)

        conjuntos_frecuentes = apriori(matriz, min_support=min_support, use_colnames=True, max_len=3)

        if conjuntos_frecuentes.empty or not any(len(c) >= 2 for c in conjuntos_frecuentes["itemsets"]):
            reglas = []
        else:
            reglas_df = association_rules(conjuntos_frecuentes, metric="confidence", min_threshold=min_confidence)
            if not reglas_df.empty:
                reglas_df = reglas_df.sort_values(by=["confidence", "lift"], ascending=False).head(100)
            reglas = []
            for _, fila in reglas_df.iterrows():
                reglas.append({
                    "antecedente": sorted(list(fila["antecedents"])),
                    "consecuente": sorted(list(fila["consequents"])),
                    "soporte": round(float(fila["support"]), 4),
                    "confianza": round(float(fila["confidence"]), 4),
                    "lift": round(float(fila["lift"]), 4)
                })

        interpretacion = f"Se encontraron {len(reglas)} reglas." if reglas else "No hay reglas con esos umbrales."

        return jsonify({"n_transacciones": int(len(matriz)), "reglas": reglas, "interpretacion": interpretacion})

    except Exception as e:
        return jsonify({"mensaje": f"Error al ejecutar Apriori: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)