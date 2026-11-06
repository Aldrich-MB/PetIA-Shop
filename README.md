# PetIA Shop - Dashboard web inteligente de mascotas

Aprendizaje Automático - Unidad III - Ingeniería en Sistemas Computacionales - TESJo

Aplicación web dinámica desarrollada con Flask que administra un dataset original de 500,000 mascotas (perro, gato, ave y tortuga). El sistema integra técnicas de aprendizaje supervisado (clasificación) y no supervisado (agrupación y asociación), permitiendo la interacción en tiempo real a través de un dashboard.

## Tecnologías Utilizadas

* Backend: Python 3, Flask, Pandas, Scikit-Learn, Joblib
* Frontend: HTML5, CSS3, JavaScript (Vanilla), Chart.js
* Control de Versiones: Git y GitHub

## Estructura del Proyecto

PetIA-Shop/
│
├── app.py                  # Servidor Flask, rutas de API y lógica de ML
├── requirements.txt        # Dependencias del proyecto
├── README.md               # Documentación general
├── .gitignore              # Archivos ignorados por Git
│
├── dataset/                # Datos originales
│   ├── mascotas.csv        # Dataset original de 500,000 registros
│   └── generar_dataset.py  # Script generador de datos
│
├── models/                 # Modelos entrenados y transformadores (.pkl)
│
├── modulos/                # Lógica de validación del Integrante 1
│   ├── __init__.py
│   ├── datos.py            # Gestor del dataset en memoria
│   ├── esquema.py          # Definición de columnas, rangos y razas
│   └── validacion.py       # Validaciones de CSV y formularios
│
├── static/                 # Recursos estáticos del frontend
│   ├── css/                # Hojas de estilo (I1)
│   ├── js/                 # Lógica del cliente y gráficas (I1)
│   ├── img/                # Imágenes de las razas
│   └── vendor/             # Librerías de terceros (Chart.js, Iconos)
│
└── templates/              # Plantillas HTML
    └── index.html          # Dashboard de una sola página (I1)


## Características Implementadas

* Administración del Dataset (I1): Carga y validación del CSV, contador dinámico de registros, tabla paginada con filtros y gráficas exploratorias.
* Entrenamiento (I2): División estricta de datos (75% train, 15% validación, 10% test) sin fugas de información. Comparación automática de tres clasificadores: GaussianNB, Árbol de Decisión y Random Forest.
* Clasificación (I2): Formulario con las 5 características (peso, altura, edad, actividad, pelaje) para predecir el tipo de mascota y obtener las probabilidades de cada clase.
* Métricas de Evaluación (I2): Cálculo de error total (1 - accuracy), FPR macro, FNR macro, F1-score y matriz de confusión sobre el conjunto de prueba.
* Agrupamiento K-means (I3): Módulo preparado para descubrir perfiles semejantes sin usar etiquetas.
* Asociación Apriori (I3): Módulo preparado para descubrir reglas entre características con soporte, confianza y lift.

## Instalación y Ejecución

1. Clonar el repositorio:
   git clone https://github.com/TU_USUARIO/PetIA-Shop.git
   cd PetIA-Shop

2. Crear y activar un entorno virtual:
   python -m venv venv
   # En Windows:
   venv\Scripts\activate
   # En macOS/Linux:
   source venv/bin/activate

3. Instalar las dependencias:
   pip install -r requirements.txt

4. Ejecutar la aplicación:
   python app.py

5. Abrir en el navegador:
   Visita http://127.0.0.1:5000 para ver el dashboard en funcionamiento.

## Equipo de Desarrollo

* Integrante 1: Diseño UI/UX, estructura del frontend, administración del dataset y módulos de validación.
* Integrante 2: Backend, integración de Flask, entrenamiento de clasificadores y predicciones.
* Integrante 3: Implementación de K-means, Apriori y generación de gráficas analíticas.
* Integrante 4: Documentación técnica, pruebas de calidad (QA) y demostración final.