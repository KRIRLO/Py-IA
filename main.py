import torch
import torch_directml as dml
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import os

# Configurar el dispositivo DirectML
device = dml.device()
print(f"Usando dispositivo: {device}")

# Cargar modelo preentrenado (no requiere entrenamiento adicional)
model_name = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Mover el modelo al dispositivo DirectML
model = model.to(device)

# Función para cargar documentos
def cargar_documento(ruta_archivo):
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        return f.read()

# Función para responder preguntas
def responder_pregunta(pregunta, contexto):
    inputs = tokenizer(pregunta, contexto, return_tensors="pt")
    # Mover los tensores al dispositivo DirectML
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    
    answer_tokens = inputs.input_ids[0][answer_start:answer_end]
    answer = tokenizer.decode(answer_tokens)
    
    return answer

# Ejemplo de uso
ruta_documento = os.path.join(os.getcwd(), "documentos", "DEV.FSD.ELIOT v3.0.02 with changelog.txt")  # Get absolute path using current working directory
try:
    documento = cargar_documento(ruta_documento)
    print(f"Documento cargado. Longitud: {len(documento)} caracteres")
    
    # Para documentos grandes, podríamos necesitar dividirlos
    max_length = 512  # Limitación del modelo
    if len(tokenizer.tokenize(documento)) > max_length:
        print("Documento demasiado largo, se procesará por fragmentos")
        # Implementación simple de división por párrafos
        parrafos = documento.split('\n\n')
        
        while True:
            pregunta = input("\nHaz una pregunta sobre el documento (o escribe 'salir' para terminar): ")
            if pregunta.lower() == 'salir':
                break
            
            mejores_respuestas = []
            for i, parrafo in enumerate(parrafos):
                if len(parrafo.strip()) > 10:  # Ignorar párrafos muy cortos
                    try:
                        respuesta = responder_pregunta(pregunta, parrafo)
                        # Calcular una puntuación simple (longitud de la respuesta)
                        puntuacion = len(respuesta.strip())
                        if puntuacion > 3:  # Ignorar respuestas muy cortas
                            mejores_respuestas.append((respuesta, puntuacion))
                    except Exception as e:
                        print(f"Error procesando párrafo {i}: {e}")
            
            if mejores_respuestas:
                # Ordenar por puntuación (simple)
                mejores_respuestas.sort(key=lambda x: x[1], reverse=True)
                print(f"Respuesta: {mejores_respuestas[0][0]}")
            else:
                print("No se encontró una respuesta en el documento.")
    else:
        # Documento corto, procesamiento directo
        while True:
            pregunta = input("\nHaz una pregunta sobre el documento (o escribe 'salir' para terminar): ")
            if pregunta.lower() == 'salir':
                break
            
            respuesta = responder_pregunta(pregunta, documento)
            print(f"Respuesta: {respuesta}")
            
except FileNotFoundError:
    print(f"No se encontró el archivo: {ruta_documento}")
except Exception as e:
    print(f"Error: {e}")