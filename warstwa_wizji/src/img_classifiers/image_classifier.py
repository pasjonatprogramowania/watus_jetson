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

import time

import numpy as np
import torch
from transformers import (
    AutoImageProcessor, 
    SiglipForImageClassification, 
    ViTImageProcessor,
    ViTForImageClassification
)

from src.img_classifiers import findDominantColor
from src.img_classifiers.utils import retrieve_img


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

    def load_models(self):
        """
        Ładuje model i procesor z HuggingFace Hub.
        
        Domyślnie ładuje modele Siglip. Klasy pochodne mogą nadpisać
        tę metodę dla innych architektur (np. ViT).
        
        Hierarchia wywołań:
            warstwa_wizji/main.py -> CVAgent.__init__() -> load_models()
        """
        self.processor = AutoImageProcessor.from_pretrained(self.id, use_fast=True)
        self.model = SiglipForImageClassification.from_pretrained(self.id)

    def process(self, img) -> dict:
        """
        Przetwarza obraz i zwraca prawdopodobieństwa dla każdej klasy.
        
        Argumenty:
            img (PIL.Image): Obraz wejściowy.
            
        Zwraca:
            dict: Słownik {nazwa_klasy: prawdopodobieństwo}.
                  Prawdopodobieństwa sumują się do 1.0.
                  
        Hierarchia wywołań:
            warstwa_wizji/main.py -> CVAgent._process_person() -> process()
        """
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

    def load_models(self):
        """Ładuje model ViT specyficzny dla rozpoznawania emocji."""
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

    def load_models(self):
        """Ładuje model ViT specyficzny dla klasyfikacji ubrań."""
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

    def load_models(self):
        """Ładuje model ViT specyficzny dla klasyfikacji wzorów."""
        self.processor = ViTImageProcessor.from_pretrained(self.id, use_fast=True)
        self.model = ViTForImageClassification.from_pretrained(self.id)


def getClothesClassifiers():
    """
    Tworzy i zwraca załadowane klasyfikatory do analizy ubrań.
    
    Funkcja inicjalizuje trzy klasyfikatory:
    - Typ ubrania (koszulka, spodnie, buty, itp.)
    - Wzór ubrania (paski, kratka, jednolity, itp.)
    - Dominujący kolor (funkcja K-Means)
    
    Zwraca:
        tuple: (ClothesClassifier, ClothesPatternClassifier, findDominantColor)
        
    Uwaga:
        Ładowanie modeli może zająć kilka sekund przy pierwszym uruchomieniu.
        
    Hierarchia wywołań:
        warstwa_wizji/main.py -> CVAgent.__init__() -> getClothesClassifiers()
    """
    clothes_classifier = ClothesClassifier()
    clothes_pattern_classifier = ClothesPatternClassifier()
    color_classifier = findDominantColor
    
    clothes_classifier.load_models()
    clothes_pattern_classifier.load_models()
    
    return clothes_classifier, clothes_pattern_classifier, color_classifier


def getClassifiers():
    """
    Tworzy i zwraca załadowane klasyfikatory do analizy osób.
    
    Funkcja inicjalizuje trzy klasyfikatory:
    - Emocje twarzy (szczęśliwy, smutny, zły, itp.)
    - Płeć (kobieta, mężczyzna)
    - Grupa wiekowa (dziecko, nastolatek, dorosły, itp.)
    
    Zwraca:
        tuple: (EmotionClassifier, GenderClassifier, AgeClassifier)
        
    Uwaga:
        Ładowanie modeli może zająć kilka sekund przy pierwszym uruchomieniu.
        
    Hierarchia wywołań:
        warstwa_wizji/main.py -> CVAgent.__init__() -> getClassifiers()
    """
    emotion_classifier = EmotionClassifier()
    gender_classifier = GenderClassifier()
    age_classifier = AgeClassifier()
    
    emotion_classifier.load_models()
    gender_classifier.load_models()
    age_classifier.load_models()
    
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
