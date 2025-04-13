# Red Neuronal para Procesamiento de Documentos

Este proyecto implementa una red neuronal para procesar y aprender de la información contenida en documentos de texto.

## Requisitos

- Python 3.7 o superior
- TensorFlow 2.0 o superior
- NumPy

## Instalación

1. Clonar o descargar este repositorio
2. Instalar las dependencias:
   ```
   pip install -r requirements.txt
   ```

## Uso

1. Coloca tu documento de texto con nombre `documento.txt` en la carpeta raíz del proyecto
2. Ejecuta el script principal:
   ```
   python text_neural_network.py
   ```

## Estructura del Proyecto

- `text_neural_network.py`: Script principal con la implementación de la red neuronal
- `requirements.txt`: Lista de dependencias del proyecto
- `documento.txt`: Archivo de texto para el entrenamiento (debes proporcionarlo)
- `modelo_texto.h5`: Modelo entrenado (se genera después de la ejecución)

## Funcionalidades

- Procesamiento de texto mediante tokenización
- Red neuronal con capas LSTM para aprendizaje secuencial
- Guardado automático del modelo entrenado