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

# IMPORTAMOS LOS MÓDULOS DEL INTEGRANTE 1
from modulos import esquema as E
from modulos.datos import GestorDataset

app = Flask(__name__)

# Inicializar el gestor del dataset (asegúrate de que la ruta sea correcta)
RUTA_DATASET = Path("dataset/mascotas.csv")
gestor = GestorDataset(RUTA_DATASET)

# Variables globales para guardar modelos en memoria
modelos_entrenados = {}

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
        target = 'tipo'
        
        X = df[features].copy()
        y = df[target].copy()

        # Codificar variables categóricas
        le_actividad = LabelEncoder()
        le_pelaje = LabelEncoder()
        le_tipo = LabelEncoder()

        X['nivel_actividad'] = le_actividad.fit_transform(X['nivel_actividad'])
        X['tipo_pelaje'] = le_pelaje.fit_transform(X['tipo_pelaje'])
        y_encoded = le_tipo.fit_transform(y)

        # 3. División 75% Entrenamiento, 15% Validación, 10% Prueba
        # Primero sacamos el 10% para prueba
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_encoded, test_size=0.10, random_state=42, stratify=y_encoded
        )
        
        # Del 90% restante, sacamos el 16.66% para validación (equivale al 15% del total)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.1666, random_state=42, stratify=y_temp
        )

        # Escalamiento (SIN FUGAS: fit solo con Train)
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc = scaler.transform(X_val)
        X_test_sc = scaler.transform(X_test)

        # 4. Entrenar los 3 clasificadores
        modelos = {
            'GaussianNB': GaussianNB(),
            'Árbol de Decisión': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }

        resultados_modelos = []
        mejor_modelo_nombre = None
        mejor_f1_val = -1
        mejor_modelo_obj = None

        # Para la matriz de confusión del mejor modelo
        mejor_pred_test = None

        for nombre, modelo in modelos.items():
            modelo.fit(X_train_sc, y_train)
            
            pred_val = modelo.predict(X_val_sc)
            pred_test = modelo.predict(X_test_sc)

            f1_val = f1_score(y_val, pred_val, average='macro')
            
            # Métricas en Test (Rúbrica)
            acc_test = accuracy_score(y_test, pred_test)
            error_total = 1.0 - acc_test
            f1_test = f1_score(y_test, pred_test, average='macro')
            fnr_macro = 1.0 - recall_score(y_test, pred_test, average='macro')
            
            cm = confusion_matrix(y_test, pred_test)
            fpr_macro = 1.0 - (np.sum(cm.diagonal()) / np.sum(cm, axis=0).mean())

            resultados_modelos.append({
                "nombre": nombre,
                "accuracy": float(acc_test),
                "error_total": float(error_total),
                "fpr_macro": float(fpr_macro),
                "fnr_macro": float(fnr_macro),
                "f1_macro": float(f1_test)
            })

            modelos_entrenados[nombre] = modelo
            
            if f1_val > mejor_f1_val:
                mejor_f1_val = f1_val
                mejor_modelo_nombre = nombre
                mejor_modelo_obj = modelo
                mejor_pred_test = pred_test

        # Guardar transformadores en disco
        os.makedirs('models', exist_ok=True)
        joblib.dump(scaler, 'models/scaler.pkl')
        joblib.dump(le_actividad, 'models/le_actividad.pkl')
        joblib.dump(le_pelaje, 'models/le_pelaje.pkl')
        joblib.dump(le_tipo, 'models/le_tipo.pkl')
        
        # Generar matriz final para el frontend
        cm_final = confusion_matrix(y_test, mejor_pred_test)
        etiquetas_clases = le_tipo.classes_.tolist()
        
        matrices_front = {
            "tipo": {
                "etiquetas": etiquetas_clases,
                "valores": cm_final.tolist()
            }
        }

        # Obtener las métricas del mejor modelo para el resumen
        metricas_mejor = next(m for m in resultados_modelos if m["nombre"] == mejor_modelo_nombre)

        return jsonify({
            "status": "success",
            "mejor": mejor_modelo_nombre,
            "division": {
                "train": len(X_train),
                "val": len(X_val),
                "test": len(X_test)
            },
            "modelos": resultados_modelos,
            "test": metricas_mejor,
            "matrices": matrices_front,
            "interpretacion": f"El modelo {mejor_modelo_nombre} fue elegido por su mejor F1 en validación. Se evaluó finalmente en el conjunto de prueba."
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
            return jsonify({"mensaje": "Primero debes reentrenar los modelos en la pestaña de Entrenamiento."}), 400

        datos = request.json
        
        scaler = joblib.load('models/scaler.pkl')
        le_actividad = joblib.load('models/le_actividad.pkl')
        le_pelaje = joblib.load('models/le_pelaje.pkl')
        le_tipo = joblib.load('models/le_tipo.pkl')

        peso = float(datos['peso_kg'])
        altura = float(datos['altura_cm'])
        edad = float(datos['edad_anios'])
        actividad = le_actividad.transform([datos['nivel_actividad']])[0]
        pelaje = le_pelaje.transform([datos['tipo_pelaje']])[0]

        X_nuevo = np.array([[peso, altura, edad, actividad, pelaje]])
        X_nuevo_sc = scaler.transform(X_nuevo)

        # Usamos Random Forest por defecto
        modelo = modelos_entrenados.get('Random Forest') 
        
        clase_predicha_encoded = modelo.predict(X_nuevo_sc)[0]
        probabilidades = modelo.predict_proba(X_nuevo_sc)[0]
        
        clase_predicha = le_tipo.inverse_transform([clase_predicha_encoded])[0]

        prob_tipo = {clase: float(prob) for clase, prob in zip(le_tipo.classes_, probabilidades)}

        imagen_url = f"/static/img/razas/{clase_predicha.lower()}.jpg"

        return jsonify({
            "tipo": clase_predicha,
            "raza": "Por definir (Clasificador de Tipo)",
            "prob_tipo": prob_tipo,
            "prob_raza": {},
            "modelo": "Random Forest",
            "imagen": imagen_url
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"mensaje": f"Error al clasificar: {str(e)}"}), 500

# ==========================================
# RUTAS PREPARADAS PARA EL INTEGRANTE 3
# ==========================================
@app.route('/api/agrupar', methods=['POST'])
def api_agrupar():
    return jsonify({"pendiente": True, "mensaje": "Módulo K-means en construcción (Integrante 3)"}), 501

@app.route('/api/asociar', methods=['POST'])
def api_asociar():
    return jsonify({"pendiente": True, "mensaje": "Módulo Apriori en construcción (Integrante 3)"}), 501


if __name__ == '__main__':
    app.run(debug=True, port=5000)