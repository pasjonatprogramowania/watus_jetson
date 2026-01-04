"""
Moduł klasyfikacji dominującego koloru obrazu.

Wykorzystuje algorytm K-Means na GPU do znalezienia dominującego koloru
w obrazie (np. koloru ubrania). Optymalizowany pod kątem przetwarzania 
w czasie rzeczywistym z wykorzystaniem CUDA.

Hierarchia wywołań:
    warstwa_wizji/main.py -> CVAgent._process_person()
        -> process_clothes_detection() -> findDominantColor()
"""

import os
import numpy as np
import torch
from kmeans_gpu import KMeans


# Konfiguracja
os.environ["OMP_NUM_THREADS"] = "8"

# Domyślne parametry K-Means
DEFAULT_NUM_CLUSTERS = 16
MAX_ITERATIONS = 100
CONVERGENCE_TOLERANCE = 1e-2


def findDominantColor(
    input_image: np.ndarray, 
    n_clusters: int = DEFAULT_NUM_CLUSTERS
) -> tuple[float, float, float]:
    """
    Znajduje dominujący kolor w obrazie za pomocą algorytmu K-Means.
    
    Funkcja grupuje piksele obrazu w klastry o podobnych kolorach
    i zwraca kolor klastra z największą liczbą pikseli.
    Wykorzystuje GPU (CUDA) dla przyspieszenia obliczeń.
    
    Argumenty:
        input_image (np.ndarray): Obraz wejściowy w formacie:
                                  - (H, W, 3) - standardowy obraz RGB/BGR
                                  - (N, 3) - lista pikseli RGB
        n_clusters (int): Liczba klastrów K-Means (domyślnie 16).
                          Więcej = dokładniejszy wynik, ale wolniejsze.
        
    Zwraca:
        tuple[float, float, float]: Dominujący kolor jako (R, G, B) w zakresie 0-255.
        
    Wyjątki:
        ValueError: Gdy obraz ma mniej niż 3 kanały lub jest pusty.
        
    Hierarchia wywołań:
        warstwa_wizji/src/cv_utils/detection_processors.py 
            -> process_clothes_detection() -> findDominantColor()
    """
    img_array = np.asarray(input_image)
    
    # Określ urządzenie (GPU jeśli dostępne)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Przygotuj piksele do klastrowania
    if img_array.ndim == 3:
        h, w, c = img_array.shape
        if c < 3:
            raise ValueError(f"Oczekiwano ≥3 kanałów (RGB), a otrzymano {c}.")
        pixels = img_array[..., :3].reshape(-1, 3)
    elif img_array.ndim == 2:
        if img_array.shape[1] < 3:
            raise ValueError(f"Oczekiwano ≥3 kanałów (RGB), a otrzymano {img_array.shape[1]}.")
        pixels = img_array[:, :3]
    else:
        raise ValueError("Oczekiwano obrazu (H, W, 3) lub listy pikseli (N, 3).")
    
    if pixels.size == 0:
        raise ValueError("Obraz nie zawiera żadnych pikseli.")
    
    # Ogranicz liczbę klastrów do liczby pikseli
    n_clusters_eff = int(min(n_clusters, pixels.shape[0]))
    
    # Konwertuj do tensora PyTorch
    pixels_tensor = torch.from_numpy(pixels.astype(np.float32)).to(device)
    
    # Przygotuj dane do K-Means (wymaga formatu (batch, points, features))
    points = pixels_tensor.unsqueeze(0)   # (1, N, 3)
    features = points.permute(0, 2, 1)     # (1, 3, N)
    
    # Uruchom K-Means
    kmeans = KMeans(
        n_clusters=n_clusters_eff,
        max_iter=MAX_ITERATIONS,
        tolerance=CONVERGENCE_TOLERANCE,
        distance="euclidean",
        sub_sampling=None,
        max_neighbors=5,
    ).to(device)
    
    centroids_tensor, _ = kmeans(points, features)
    centroids = centroids_tensor[0].detach().cpu().numpy()  # (K, 3)
    
    # Przypisz każdy piksel do najbliższego centroidu
    diff = pixels[:, None, :] - centroids[None, :, :]   # (N, K, 3)
    dist2 = np.sum(diff * diff, axis=2)                  # (N, K)
    labels = np.argmin(dist2, axis=1)                    # (N,)
    
    # Znajdź klaster z największą liczbą pikseli
    counts = np.bincount(labels, minlength=n_clusters_eff)
    dominant_label = int(np.argmax(counts))
    
    # Pobierz kolor dominującego klastra
    dominant_color = centroids[dominant_label]
    dominant_color = np.clip(dominant_color, 0, 255).astype(np.uint8)
    
    return tuple(dominant_color)