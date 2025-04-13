import torch
import torch_directml as dml
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import os
import re
import json

# Configurar el dispositivo DirectML
device = dml.device()
print(f"Usando dispositivo: {device}")

# Función para cargar y preparar tus datos
def cargar_documentos(ruta_carpeta):
    documentos = []
    for archivo in os.listdir(ruta_carpeta):
        if archivo.endswith('.txt'):
            with open(os.path.join(ruta_carpeta, archivo), 'r', encoding='utf-8') as f:
                contenido = f.read()
                documentos.append({
                    'id': archivo,
                    'contenido': contenido
                })
    return documentos

# Función para crear un conjunto de datos de ejemplo (pregunta-respuesta)
def crear_dataset_entrenamiento(documentos, preguntas_respuestas):
    dataset = []
    for doc in documentos:
        for qa in preguntas_respuestas:
            if qa['doc_id'] == doc['id']:
                dataset.append({
                    'context': doc['contenido'],
                    'question': qa['pregunta'],
                    'answer': {
                        'text': [qa['respuesta']],
                        'answer_start': [doc['contenido'].find(qa['respuesta'])]
                    }
                })
    return Dataset.from_pandas(pd.DataFrame(dataset))

# Preprocesamiento para el modelo
def preprocesar_datos(ejemplos, tokenizer, max_length=384, stride=128):
    preguntas = [q.strip() for q in ejemplos["question"]]
    contextos = [c.strip() for c in ejemplos["context"]]
    
    inputs = tokenizer(
        preguntas,
        contextos,
        max_length=max_length,
        stride=stride,
        truncation="only_second",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    
    start_positions = []
    end_positions = []
    
    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        respuesta = ejemplos["answer"][sample_idx]
        start_char = respuesta["answer_start"][0]
        end_char = start_char + len(respuesta["text"][0])
        
        sequence_ids = inputs.sequence_ids(i)
        
        # Encontrar el inicio y fin del contexto
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        
        while idx < len(sequence_ids) and sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1
        
        # Si la respuesta no está completamente dentro del contexto, marcarla como imposible
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
            continue
            
        # De lo contrario, encontrar la posición de inicio y fin
        token_start = token_end = context_start
        
        while token_start <= context_end and offset[token_start][0] <= start_char:
            token_start += 1
        start_positions.append(token_start - 1)
        
        while token_end <= context_end and offset[token_end][1] <= end_char:
            token_end += 1
        end_positions.append(token_end - 1)
    
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# Carga de modelo y tokenizer
model_name = "distilbert-base-cased-distilled-squad"  # Modelo más ligero para QA
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Mover modelo al dispositivo DirectML
model = model.to(device)

# Ejemplo de datos de entrenamiento
# Deberías reemplazar esto con tus propios datos
documentos = cargar_documentos("ruta/a/tus/documentos")
preguntas_respuestas = [
    {"doc_id": "documento1.txt", "pregunta": "¿Cuál es el tema principal?", "respuesta": "El tema principal es..."},
    {"doc_id": "documento1.txt", "pregunta": "¿Quién es el autor?", "respuesta": "El autor es..."},
    # Agrega más ejemplos según sea necesario
]

# Crear dataset
dataset = crear_dataset_entrenamiento(documentos, preguntas_respuestas)

# Preprocesamiento
dataset_tokenizado = dataset.map(
    lambda ejemplos: preprocesar_datos(ejemplos, tokenizer),
    batched=True,
    remove_columns=dataset.column_names
)

# Configuración de entrenamiento
args_entrenamiento = TrainingArguments(
    output_dir="./resultados",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_strategy="epoch",
    save_total_limit=2,
    learning_rate=5e-5,
    warmup_steps=500,
    weight_decay=0.01,
)

# Crear el entrenador
trainer = Trainer(
    model=model,
    args=args_entrenamiento,
    train_dataset=dataset_tokenizado,
    tokenizer=tokenizer,
)

# Entrenar el modelo
trainer.train()

# Guardar el modelo entrenado
model.save_pretrained("./modelo_qa_personalizado")
tokenizer.save_pretrained("./modelo_qa_personalizado")

# Función para usar el modelo entrenado
def responder_pregunta(pregunta, contexto):
    inputs = tokenizer(pregunta, contexto, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    
    answer = tokenizer.decode(inputs.input_ids[0][answer_start:answer_end])
    
    return answer

# Ejemplo de uso
documento_ejemplo = "Contenido del documento al que quieres hacer preguntas..."
pregunta_ejemplo = "¿Qué información específica quieres saber?"

respuesta = responder_pregunta(pregunta_ejemplo, documento_ejemplo)
print(f"Respuesta: {respuesta}")