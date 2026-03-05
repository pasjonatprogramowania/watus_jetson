"""
Moduł klasyfikatorów obrazów.

Zawiera klasyfikatory do analizy osób na obrazach:
- Klasyfikator emocji (twarzy)
- Klasyfikator płci
- Klasyfikator wieku
- Klasyfikatory ubrań (typ, wzór)

Wszystkie klasyfikatory wykorzystują modele HuggingFace (transformers).

Hierarchia wywołań:
    warstwa_wizji/main.py -> CVAgent.__init__() -> getClassifiers()
    warstwa_wizji/main.py -> CVAgent.__init__() -> getClothesClassifiers()
"""

import json
import os
import time

import numpy as np
import torch
from transformers import (
    AutoImageProcessor, 
    SiglipForImageClassification, 
    ViTImageProcessor,
    ViTForImageClassification
)

from .color_classifier import findDominantColor
from .utils import retrieve_img

# Katalog z plikami ONNX i metadanymi
_MODELS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "models")
)

# Mapowanie klasy klasyfikatora → nazwa pliku ONNX (bez rozszerzenia)
ONNX_NAMES = {
    "EmotionClassifier": "emotion",
    "GenderClassifier": "gender",
    "AgeClassifier": "age",
    "ClothesClassifier": "clothes_type",
    "ClothesPatternClassifier": "clothes_pattern",
}

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False


class ImageClassifier:
    """
    Bazowa klasa klasyfikatora obrazów.
    
    Zapewnia wspólny interfejs do ładowania modeli HuggingFace 
    i przetwarzania obrazów. Klasy pochodne definiują konkretne modele
    i mapowania klas.
    
    Atrybuty:
        id (str): Identyfikator modelu na HuggingFace Hub.
        model: Załadowany model sieci neuronowej (np. Siglip, ViT).
        processor: Załadowany procesor przygotowujący dane wejściowe.
        id2label (dict): Słownik mapujący ID klasy na etykietę tekstową.
        
    Hierarchia wywołań:
        Używana jako klasa bazowa dla konkretnych klasyfikatorów.
    """
    
    def __init__(self, idx: str):
        """
        Inicjalizuje bazowy klasyfikator.
        
        Argumenty:
            idx (str): ID modelu na HuggingFace Hub.
        """
        self.id = idx
        self.model = None
        self.processor = None
        self.id2label = {}
        self.onnx_session = None

    def load_models(self, use_onnx: bool = False):
        """
        Ładuje model — z pliku ONNX jeśli dostępny, inaczej z HuggingFace Hub.
        
        Argumenty:
            use_onnx: Jeśli True, próbuje załadować plik .onnx z src/models/.
                      Gdy plik nie istnieje lub brak onnxruntime — fallback na PyTorch.
        """
        if use_onnx and self._try_load_onnx():
            return
        self.processor = AutoImageProcessor.from_pretrained(self.id, use_fast=True)
        self.model = SiglipForImageClassification.from_pretrained(self.id)

    def _try_load_onnx(self) -> bool:
        """Próbuje załadować model ONNX i metadata. Zwraca True jeśli sukces."""
        if not _ORT_AVAILABLE:
            print(f"[ONNX] onnxruntime niedostępny — fallback na PyTorch")
            return False

        cls_name = type(self).__name__
        onnx_basename = ONNX_NAMES.get(cls_name)
        if onnx_basename is None:
            return False

        onnx_path = os.path.join(_MODELS_DIR, f"{onnx_basename}.onnx")
        meta_path = os.path.join(_MODELS_DIR, f"{onnx_basename}_meta.json")

        if not os.path.isfile(onnx_path):
            print(f"[ONNX] Brak pliku {onnx_path} — fallback na PyTorch")
            return False

        # Ładowanie sesji ONNX
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.onnx_session = ort.InferenceSession(str(onnx_path), providers=providers)
        print(f"[ONNX] Załadowano sesję: {onnx_path}")

        # Ładowanie procesora z metadata lub z HF Hub
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            # id2label z meta (nadpisuje domyślny, jeśli istnieje)
            if "id2label" in meta:
                self.id2label = meta["id2label"]

        # Procesor nadal potrzebny do preprocesingu obrazu
        arch = None
        if os.path.isfile(meta_path):
            arch = meta.get("arch")
        if arch == "vit":
            self.processor = ViTImageProcessor.from_pretrained(self.id, use_fast=True)
        else:
            self.processor = AutoImageProcessor.from_pretrained(self.id, use_fast=True)

        return True

    @property
    def is_onnx(self) -> bool:
        """Czy model działa w trybie ONNX."""
        return self.onnx_session is not None

    def process(self, img) -> dict:
        """
        Przetwarza obraz i zwraca prawdopodobieństwa dla każdej klasy.
        Automatycznie używa ONNX Runtime jeśli sesja jest aktywna.
        """
        if self.onnx_session is not None:
            return self._process_onnx(img)
        return self._process_pytorch(img)

    def _process_pytorch(self, img) -> dict:
        """Inferencja przez PyTorch."""
        img = img.convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

        prediction = {
            self.id2label[str(i)]: round(probs[i], 3) 
            for i in range(len(probs))
        }
        return prediction

    def _process_onnx(self, img) -> dict:
        """Inferencja przez ONNX Runtime."""
        img = img.convert("RGB")
        inputs = self.processor(images=img, return_tensors="np")
        pixel_values = inputs["pixel_values"].astype(np.float32)

        ort_inputs = {"pixel_values": pixel_values}
        logits = self.onnx_session.run(["logits"], ort_inputs)[0]

        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = (exp_logits / exp_logits.sum(axis=1, keepdims=True)).squeeze().tolist()

        prediction = {
            self.id2label[str(i)]: round(probs[i], 3)
            for i in range(len(probs))
        }
        return prediction


class EmotionClassifier(ImageClassifier):
    """
    Klasyfikator emocji na twarzy.
    
    Model: abhilash88/face-emotion-detection
    Obsługiwane klasy: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
    """
    
    def __init__(self):
        """Inicjalizuje klasyfikator emocji z predefiniowanym modelem."""
        super().__init__('abhilash88/face-emotion-detection')
        self.id2label = {
            "0": "Angry",
            "1": "Disgust",
            "2": "Fear",
            "3": "Happy",
            "4": "Sad",
            "5": "Surprise",
            "6": "Neutral",
        }

    def load_models(self, use_onnx: bool = False):
        """Ładuje model ViT lub ONNX dla rozpoznawania emocji."""
        if use_onnx and self._try_load_onnx():
            return
        self.processor = ViTImageProcessor.from_pretrained(self.id, use_fast=True)
        self.model = ViTForImageClassification.from_pretrained(self.id)


class GenderClassifier(ImageClassifier):
    """
    Klasyfikator płci osoby na obrazie.
    
    Model: prithivMLmods/Realistic-Gender-Classification
    Obsługiwane klasy: female, male
    """
    
    def __init__(self):
        """Inicjalizuje klasyfikator płci z predefiniowanym modelem."""
        super().__init__("prithivMLmods/Realistic-Gender-Classification")
        self.id2label = {
            "0": "female",
            "1": "male"
        }


class AgeClassifier(ImageClassifier):
    """
    Klasyfikator grupy wiekowej osoby na obrazie.
    
    Model: prithivMLmods/open-age-detection
    Obsługiwane klasy: Child (0-12), Teenager (13-20), Adult (21-44), 
                       Middle Age (45-64), Aged (65+)
    """
    
    def __init__(self):
        """Inicjalizuje klasyfikator wieku z predefiniowanym modelem."""
        super().__init__("prithivMLmods/open-age-detection")
        self.id2label = {
            "0": "Child 0-12",
            "1": "Teenager 13-20",
            "2": "Adult 21-44",
            "3": "Middle Age 45-64",
            "4": "Aged 65+"
        }


class ClothesClassifier(ImageClassifier):
    """
    Klasyfikator typu części garderoby.
    
    Model: samokosik/finetuned-clothes
    Obsługiwane klasy: Hat, Longsleeve, Outwear, Pants, Shoes, Shorts, Shortsleeve
    """
    
    def __init__(self):
        """Inicjalizuje klasyfikator typu ubrania z predefiniowanym modelem."""
        super().__init__("samokosik/finetuned-clothes")
        self.id2label = {
            "0": "Hat",
            "1": "Longsleeve",
            "2": "Outwear",
            "3": "Pants",
            "4": "Shoes",
            "5": "Shorts",
            "6": "Shortsleeve"
        }

    def load_models(self, use_onnx: bool = False):
        """Ładuje model ViT lub ONNX dla klasyfikacji ubrań."""
        if use_onnx and self._try_load_onnx():
            return
        self.processor = ViTImageProcessor.from_pretrained(self.id, use_fast=True)
        self.model = ViTForImageClassification.from_pretrained(self.id)


class ClothesPatternClassifier(ImageClassifier):
    """
    Klasyfikator wzoru/deseniu na ubraniu.
    
    Model: IrshadG/Clothes_Pattern_Classification_v2
    Obsługiwane klasy: Argyle, Check, Dot, Stripe, Solid, Denim, Lace, itd.
    """
    
    def __init__(self):
        """Inicjalizuje klasyfikator wzoru ubrania z predefiniowanym modelem."""
        super().__init__("IrshadG/Clothes_Pattern_Classification_v2")
        self.id2label = {
            "0": "Argyle",
            "1": "Check",
            "2": "Colour blocking",
            "3": "Denim",
            "4": "Dot",
            "5": "Embroidery",
            "6": "Lace",
            "7": "Metallic",
            "8": "Patterns",
            "9": "Placement print",
            "10": "Sequin",
            "11": "Solid",
            "12": "Stripe",
            "13": "Transparent",
        }

    def load_models(self, use_onnx: bool = False):
        """Ładuje model ViT lub ONNX dla klasyfikacji wzorów."""
        if use_onnx and self._try_load_onnx():
            return
        self.processor = ViTImageProcessor.from_pretrained(self.id, use_fast=True)
        self.model = ViTForImageClassification.from_pretrained(self.id)


def getClothesClassifiers(use_onnx: bool = False):
    """
    Tworzy i zwraca załadowane klasyfikatory do analizy ubrań.
    
    Argumenty:
        use_onnx: Jeśli True, próbuje załadować modele z plików ONNX.
    
    Zwraca:
        tuple: (ClothesClassifier, ClothesPatternClassifier, findDominantColor)
    """
    clothes_classifier = ClothesClassifier()
    clothes_pattern_classifier = ClothesPatternClassifier()
    color_classifier = findDominantColor
    
    clothes_classifier.load_models(use_onnx=use_onnx)
    clothes_pattern_classifier.load_models(use_onnx=use_onnx)
    
    return clothes_classifier, clothes_pattern_classifier, color_classifier


def getClassifiers(use_onnx: bool = False):
    """
    Tworzy i zwraca załadowane klasyfikatory do analizy osób.
    
    Argumenty:
        use_onnx: Jeśli True, próbuje załadować modele z plików ONNX.
    
    Zwraca:
        tuple: (EmotionClassifier, GenderClassifier, AgeClassifier)
    """
    emotion_classifier = EmotionClassifier()
    gender_classifier = GenderClassifier()
    age_classifier = AgeClassifier()
    
    emotion_classifier.load_models(use_onnx=use_onnx)
    gender_classifier.load_models(use_onnx=use_onnx)
    age_classifier.load_models(use_onnx=use_onnx)
    
    return emotion_classifier, gender_classifier, age_classifier


if __name__ == "__main__":
    """Test modułu - przykładowe klasyfikacje."""
    img = retrieve_img("https://example.com/test_image.jpg")
    
    emotion_classifier = EmotionClassifier()
    gender_classifier = GenderClassifier()
    age_classifier = AgeClassifier()
    
    emotion_classifier.load_models()
    gender_classifier.load_models()
    age_classifier.load_models()

    start = time.time()
    print("Obraz załadowany!")
    
    color = findDominantColor(np.asarray(img))
    print(f"Kolor: {color}, Czas: {time.time() - start:.2f}s")

    emotion = emotion_classifier.process(img)
    print(f"Emocja: {emotion}, Czas: {time.time() - start:.2f}s")

    gender = gender_classifier.process(img)
    print(f"Płeć: {gender}, Czas: {time.time() - start:.2f}s")

    age = age_classifier.process(img)
    print(f"Wiek: {age}, Czas: {time.time() - start:.2f}s")
