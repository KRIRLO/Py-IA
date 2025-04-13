import torch
import torch_directml

# Configurar PyTorch para usar DirectML
dml = torch_directml.device()
device = dml if torch_directml.is_available() else torch.device("cpu")

# Verificar la disponibilidad de DirectML
print("¿DirectML disponible?:", torch_directml.is_available())
print("Dispositivo actual:", device)

# Mostrar información del dispositivo
if torch_directml.is_available():
    print("\nInformación del dispositivo DirectML:")
    print(f"Nombre del dispositivo: {dml.name}")
    
    # Crear un tensor de prueba en el dispositivo DirectML
    tensor_prueba = torch.ones(1).to(device)
    print("\nPrueba de tensor en dispositivo:")
    print(f"Ubicación del tensor de prueba: {tensor_prueba.device}")
else:
    print("DirectML no está disponible. Usando CPU.")