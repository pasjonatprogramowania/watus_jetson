import time

import numpy as np
import torch
from transformers import AutoImageProcessor, SiglipForImageClassification, ViTImageProcessor, \
    ViTForImageClassification

from src.img_classifiers import  findDominantColor
from src.img_classifiers.utils import retrieve_img


class ImageClassifier:
    """
    Bazowa klasa klasyfikatora obrazów.
    
    Zapewnia interfejs do ładowania modeli HuggingFace i przetwarzania obrazów.
    
    Atrybuty:
        id (str): Identyfikator modelu HuggingFace.
        model: Załadowany model (np. Siglip, ViT).
        processor: Załadowany procesor obrazu.
        id2label (dict): Mapowanie ID klasy na etykietę tekstową.
    
    Hierarchia wywołań:
        Używana jako klasa bazowa dla konkretnych klasyfikatorów.
    """
    def __init__(self, idx: str):
        self.id = idx
        self.model = None
        self.processor = None
        self.id2label = {}

    def load_models(self):
        """
        Ładuje model i procesor z HuggingFace.
        
        Hierarchia wywołań:
            warstwa_wizji/main.py -> CVAgent.warm_up() -> load_models()
            image_classifier.py -> getClassifiers() -> load_models()
        """
        self.processor = AutoImageProcessor.from_pretrained(self.id, use_fast=True)
        self.model = SiglipForImageClassification.from_pretrained(self.id)

    def process(self, img) -> dict:
        """
        Przetwarza obraz i zwraca prawdopodobieństwa klas.
        
        Argumenty:
            img (PIL.Image): Obraz wejściowy.
            
        Zwraca:
            dict: Słownik {etykieta: prawdopodobieństwo}.
            
        Hierarchia wywołań:
            warstwa_wizji/main.py -> CVAgent.process_camera() -> process()
        """
        img = img.convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

        prediction = {
            self.id2label[str(i)]: round(probs[i], 3) for i in range(len(probs))
        }
        return prediction

class EmotionClassifier(ImageClassifier):
    """
    Klasyfikator emocji na twarzy.
    
    Model: abhilash88/face-emotion-detection
    Klasy: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
    """
    def __init__(self):
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
        """Ładuje specyficzny model ViT dla emocji."""
        self.processor = ViTImageProcessor.from_pretrained(self.id, use_fast=True)
        self.model = ViTForImageClassification.from_pretrained(self.id)

class GenderClassifier(ImageClassifier):
    """
    Klasyfikator płci.
    
    Model: prithivMLmods/Realistic-Gender-Classification
    Klasy: female, male
    """
    def __init__(self):
        super().__init__("prithivMLmods/Realistic-Gender-Classification")
        self.id2label = {
            "0": "female",
            "1": "male"
        }

class AgeClassifier(ImageClassifier):
    """
    Klasyfikator wieku.
    
    Model: prithivMLmods/open-age-detection
    Klasy: Child, Teenager, Adult, Middle Age, Aged
    """
    def __init__(self):
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
    Klasyfikator części garderoby.
    
    Model: samokosik/finetuned-clothes
    Klasy: Hat, Longsleeve, Outwear, Pants, Shoes, Shorts, Shortsleeve
    """
    def __init__(self):
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
        """Ładuje specyficzny model ViT dla ubrań."""
        self.processor = ViTImageProcessor.from_pretrained(self.id, use_fast=True)
        self.model = ViTForImageClassification.from_pretrained(self.id)

class ClothesPatternClassifier(ImageClassifier):
    """
    Klasyfikator wzorów na ubraniach.
    
    Model: IrshadG/Clothes_Pattern_Classification_v2
    Klasy: Argyle, Check, Dot, Stripe, Solid, itd.
    """
    def __init__(self):
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
        """Ładuje specyficzny model ViT dla wzorów."""
        self.processor = ViTImageProcessor.from_pretrained(self.id, use_fast=True)
        self.model = ViTForImageClassification.from_pretrained(self.id)


def getClothesClassifiers():
    """
    Zwraca instancje klasyfikatorów ubrań, wzorów i kolorów (załadowane).
    
    Zwraca:
        tuple: (ClothesClassifier, ClothesPatternClassifier, findDominantColor)
        
    Hierarchia wywołań:
        warstwa_wizji/main.py -> CVAgent.warm_up() -> getClothesClassifiers()
    """
    clothes_classifier = ClothesClassifier()
    clothes_pattern_classifier = ClothesPatternClassifier()
    color_classifier = findDominantColor
    clothes_classifier.load_models()
    clothes_pattern_classifier.load_models()
    return clothes_classifier, clothes_pattern_classifier, color_classifier

def getClassifiers():
    """
    Zwraca instancje klasyfikatorów emocji, płci i wieku (załadowane).
    
    Zwraca:
        tuple: (EmotionClassifier, GenderClassifier, AgeClassifier)
        
    Hierarchia wywołań:
        warstwa_wizji/main.py -> CVAgent.warm_up() -> getClassifiers()
    """
    emotion_classifier = EmotionClassifier()
    gender_classifier = GenderClassifier()
    age_classifier = AgeClassifier()
    emotion_classifier.load_models()
    gender_classifier.load_models()
    age_classifier.load_models()
    return emotion_classifier, gender_classifier, age_classifier

if __name__ == "__main__":
    img = retrieve_img("https://wassyl.pl/hpeciai/849585a6d57bcc5edbf17f94ccec1d35/pol_pl_Viralowa-Dresowa-bawelniana-BLUZA-z-kapturem-oversize-szeroka-baby-pink-E644-14377_1.jpg")
    emotion_classifier = EmotionClassifier()
    gender_classifier = GenderClassifier()
    age_classifier = AgeClassifier()
    emotion_classifier.load_models()
    gender_classifier.load_models()
    age_classifier.load_models()

    start = time.time()
    print("Uploaded!")
    color = findDominantColor(np.asarray(img))
    print(color, time.time() - start)

    emotion = emotion_classifier.process(img)
    print(emotion, time.time() - start)

    gender = gender_classifier.process(img)
    print(gender, time.time() - start)

    age = age_classifier.process(img)
    print(age, time.time() - start)


