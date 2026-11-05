from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

# Ruta principal
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para entrenar (Tu parte - Integrante 2)
@app.route('/api/train', methods=['POST'])
def train_models():
    return jsonify({"status": "success", "message": "Modelos entrenados (En construcción)"})

# Ruta para predecir (Tu parte - Integrante 2)
@app.route('/api/predict', methods=['POST'])
def predict():
    return jsonify({"status": "success", "message": "Predicción realizada (En construcción)"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)