import torch

# Configurar PyTorch para usar la GPU AMD
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Verificar la disponibilidad de la GPU
print("¿GPU disponible?:", torch.cuda.is_available())
print("Dispositivo actual:", device)
print("Nombre de la GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

# Mostrar información adicional de la GPU
if torch.cuda.is_available():
    print("\nInformación detallada de la GPU:")
    print(f"Número de GPUs disponibles: {torch.cuda.device_count()}")
    print(f"Memoria total de la GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Capacidad de cómputo: {torch.cuda.get_device_capability(0)}")
    
    # Crear un tensor de prueba en la GPU
    tensor_prueba = torch.ones(1).to(device)
    print("\nPrueba de tensor en GPU:")
    print(f"Ubicación del tensor de prueba: {tensor_prueba.device}")