import torch
import torch_directml as dml
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import os
import PyPDF2
from typing import List, Tuple
from nltk.tokenize import sent_tokenize
import nltk
from docx import Document

# Descargar recursos necesarios de NLTK
def inicializar_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Descargando recursos de NLTK necesarios...")
        nltk.download('punkt', quiet=True)
        print("Recursos descargados exitosamente.")

# Inicializar NLTK al inicio
inicializar_nltk()

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
def cargar_documento(ruta_archivo: str) -> str:
    extension = os.path.splitext(ruta_archivo)[1].lower()
    
    if extension == '.pdf':
        try:
            with open(ruta_archivo, 'rb') as archivo:
                lector_pdf = PyPDF2.PdfReader(archivo)
                texto = ''
                for pagina in lector_pdf.pages:
                    texto += pagina.extract_text() + '\n'
                return texto
        except Exception as e:
            raise Exception(f"Error al leer el archivo PDF: {e}")
    elif extension == '.docx':
        try:
            doc = Document(ruta_archivo)
            texto = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            return texto
        except Exception as e:
            raise Exception(f"Error al leer el archivo DOCX: {e}")
    else:
        try:
            with open(ruta_archivo, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(ruta_archivo, 'r', encoding='latin-1') as f:
                return f.read()

# Función para responder preguntas
def dividir_texto(texto: str, max_tokens: int) -> List[str]:
    # Dividir el texto en oraciones
    oraciones = sent_tokenize(texto)
    fragmentos = []
    fragmento_actual = ""
    
    for oracion in oraciones:
        # Verificar si agregar la oración excedería el límite de tokens
        temp_fragmento = fragmento_actual + " " + oracion if fragmento_actual else oracion
        if len(tokenizer.tokenize(temp_fragmento)) <= max_tokens:
            fragmento_actual = temp_fragmento
        else:
            if fragmento_actual:
                fragmentos.append(fragmento_actual)
            fragmento_actual = oracion
    
    if fragmento_actual:
        fragmentos.append(fragmento_actual)
    
    return fragmentos

def calcular_puntuacion_respuesta(respuesta: str, pregunta: str, contexto: str) -> float:
    # Puntuación basada en múltiples criterios
    puntuacion = 0.0
    
    # 1. Longitud de la respuesta (penalizar respuestas muy cortas o muy largas)
    long_resp = len(respuesta.split())
    if 3 <= long_resp <= 50:
        puntuacion += 2.0
    elif long_resp > 50:
        puntuacion += 1.0
    
    # 2. Presencia de palabras clave de la pregunta en la respuesta
    palabras_pregunta = set(pregunta.lower().split())
    palabras_respuesta = set(respuesta.lower().split())
    palabras_comunes = palabras_pregunta.intersection(palabras_respuesta)
    puntuacion += len(palabras_comunes) * 0.5
    
    # 3. Contexto cercano
    if respuesta.lower() in contexto.lower():
        puntuacion += 1.5
    
    return puntuacion

def responder_pregunta(pregunta: str, contexto: str) -> Tuple[str, float]:
    inputs = tokenizer(pregunta, contexto, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Obtener los mejores índices de inicio y fin
    start_logits = outputs.start_logits[0].cpu().numpy()
    end_logits = outputs.end_logits[0].cpu().numpy()
    
    # Encontrar la mejor combinación de inicio y fin
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    
    answer_tokens = inputs.input_ids[0][answer_start:answer_end]
    respuesta = tokenizer.decode(answer_tokens)
    
    # Calcular puntuación de la respuesta
    puntuacion = calcular_puntuacion_respuesta(respuesta, pregunta, contexto)
    
    return respuesta, puntuacion

def main():
    # Permitir al usuario seleccionar el archivo
    archivos_disponibles = [f for f in os.listdir(os.path.join(os.getcwd(), "documentos")) 
                          if f.endswith(('.txt', '.pdf', '.docx'))]
    
    print("\nArchivos disponibles:")
    for i, archivo in enumerate(archivos_disponibles, 1):
        print(f"{i}. {archivo}")
    
    try:
        seleccion = int(input("\nSeleccione el número del archivo a procesar: ")) - 1
        if not 0 <= seleccion < len(archivos_disponibles):
            raise ValueError("Selección inválida")
        
        nombre_archivo = archivos_disponibles[seleccion]
        ruta_documento = os.path.join(os.getcwd(), "documentos", nombre_archivo)
        
        # Cargar y procesar el documento
        documento = cargar_documento(ruta_documento)
        print(f"\nDocumento cargado: {nombre_archivo}")
        print(f"Longitud: {len(documento)} caracteres")
        
        # Dividir el documento en fragmentos manejables
        max_tokens = 450  # Dejamos margen para la pregunta
        fragmentos = dividir_texto(documento, max_tokens)
        print(f"Documento dividido en {len(fragmentos)} fragmentos para procesamiento")
        
        while True:
            pregunta = input("\nHaz una pregunta sobre el documento (o escribe 'salir' para terminar): ")
            if pregunta.lower() == 'salir':
                break
            
            mejores_respuestas = []
            for i, fragmento in enumerate(fragmentos):
                try:
                    respuesta, puntuacion = responder_pregunta(pregunta, fragmento)
                    if puntuacion > 2.0:  # Umbral de calidad mínima
                        mejores_respuestas.append((respuesta, puntuacion, fragmento))
                except Exception as e:
                    print(f"Error procesando fragmento {i}: {e}")
            
            if mejores_respuestas:
                # Ordenar por puntuación
                mejores_respuestas.sort(key=lambda x: x[1], reverse=True)
                mejor_respuesta = mejores_respuestas[0]
                
                print(f"\nRespuesta: {mejor_respuesta[0]}")
                print(f"Confianza: {mejor_respuesta[1]:.2f}")
            else:
                print("\nNo se encontró una respuesta confiable en el documento.")
    
    except FileNotFoundError:
        print(f"No se encontró el archivo: {ruta_documento}")
    except ValueError as e:
        print(f"Error de entrada: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()