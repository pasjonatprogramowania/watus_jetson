"""
Moduł klasyfikatora obiektów wojskowych (eksperymentalny).

Przykładowy kod do ekstrakcji cech z modelu DINOv3.
Przeznaczony do przyszłej rozbudowy o klasyfikację pojazdów wojskowych.

Uwaga:
    Ten moduł jest w fazie eksperymentalnej i nie jest jeszcze
    zintegrowany z głównym pipeline'em wizyjnym.
    
Hierarchia wywołań:
    Moduł autonomiczny - uruchamiany jako skrypt testowy.
"""

import torch
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image


# Stałe konfiguracyjne
EXAMPLE_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
PRETRAINED_MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"


def load_feature_extraction_model():
    """
    Ładuje model DINOv3 do ekstrakcji cech z obrazów.
    
    Model DINOv3 jest modelem self-supervised vision transformer,
    który generuje reprezentacje wektorowe obrazów przydatne
    do klasyfikacji i wyszukiwania podobnych obrazów.
    
    Zwraca:
        tuple: (processor, model) - procesor obrazu i model DINOv3.
        
    Hierarchia wywołań:
        Używane wewnętrznie do ekstrakcji cech z obrazów.
    """
    processor = AutoImageProcessor.from_pretrained(PRETRAINED_MODEL_NAME)
    model = AutoModel.from_pretrained(
        PRETRAINED_MODEL_NAME,
        device_map="auto",
    )
    return processor, model


def extract_image_features(image, processor, model):
    """
    Ekstrahuje wektor cech z obrazu za pomocą modelu DINOv3.
    
    Argumenty:
        image: Obraz wejściowy (PIL.Image lub URL).
        processor: Załadowany procesor obrazu.
        model: Załadowany model DINOv3.
        
    Zwraca:
        torch.Tensor: Wektor cech obrazu (pooler_output).
        
    Hierarchia wywołań:
        Używane do ekstrakcji cech z obrazów obiektów wojskowych.
    """
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    
    with torch.inference_mode():
        outputs = model(**inputs)
    
    feature_vector = outputs.pooler_output
    return feature_vector


if __name__ == "__main__":
    """Test modułu - ekstrakcja cech z przykładowego obrazu."""
    test_image = load_image(EXAMPLE_IMAGE_URL)
    
    processor, model = load_feature_extraction_model()
    
    inputs = processor(images=test_image, return_tensors="pt").to(model.device)
    
    with torch.inference_mode():
        outputs = model(**inputs)
    
    pooled_output = outputs.pooler_output
    print(f"Wymiar wektora cech (pooled output): {pooled_output.shape}")
    print(f"Indeks maksymalnej wartości: {pooled_output.argmax(dim=-1).item()}")
