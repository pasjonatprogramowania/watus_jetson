"""
Moduł ważonego zbioru danych YOLO.

Implementuje niestandardowy Dataset z ważeniem próbek bazującym na częstości klas.
Pomaga zbalansować trening na zbiorach danych z nierównomiernym rozkładem klas.

Hierarchia wywołań:
    warstwa_wizji/src/model_trainer/yolo_train.py -> YOLOWeightedDataset()
"""

import numpy as np
from ultralytics.data import dataset


class YOLOWeightedDataset(dataset.YOLODataset):
    """
    Dataset YOLO z ważonym próbkowaniem klas.
    
    Rozszerza standardowy YOLODataset o mechanizm ważenia próbek,
    gdzie rzadsze klasy mają większe prawdopodobieństwo wylosowania.
    Pomaga w treningu na niezbalansowanych zbiorach danych.
    
    Atrybuty:
        counts (np.ndarray): Liczba wystąpień każdej klasy.
        train_mode (bool): Czy jest to tryb treningowy.
        class_weights (np.ndarray): Wagi dla każdej klasy (odwrotność częstości).
        weights (list): Zagregowane wagi dla każdej etykiety.
        probabilities (list): Prawdopodobieństwa próbkowania.
        agg_func: Funkcja agregująca wagi (domyślnie np.mean).
        
    Hierarchia wywołań:
        yolo_train.py -> dataset.YOLODataset = YOLOWeightedDataset
    """
    
    def __init__(self, *args, mode="train", **kwargs):
        """
        Inicjalizuje ważony dataset YOLO.
        
        Oblicza wagi klas na podstawie odwrotności ich częstości występowania.
        Klasy rzadsze otrzymują wyższe wagi.
        
        Argumenty:
            *args: Argumenty pozycyjne przekazywane do rodzica.
            mode (str): Tryb działania ("train" lub "val").
            **kwargs: Argumenty słownikowe przekazywane do rodzica.
        """
        super(YOLOWeightedDataset, self).__init__(*args, **kwargs)

        self.counts = None
        self.train_mode = "train" in self.prefix

        # Zlicz wystąpienia klas
        self.count_instances()
        
        # Oblicz wagi jako odwrotność częstości
        class_weights = np.sum(self.counts) / self.counts

        # Funkcja agregująca dla etykiet z wieloma obiektami
        self.agg_func = np.mean

        self.class_weights = np.array(class_weights)
        self.weights = self.calculate_weights()
        self.probabilities = self.calculate_probabilities()

    def count_instances(self):
        """
        Zlicza liczbę wystąpień każdej klasy w zbiorze danych.
        
        Iteruje przez wszystkie etykiety i sumuje wystąpienia każdej klasy.
        Klasy bez wystąpień otrzymują minimalną wartość 1.
        
        Hierarchia wywołań:
            YOLOWeightedDataset.__init__() -> count_instances()
        """
        self.counts = [0 for i in range(len(self.data["names"]))]
        
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)
            for idx in cls:
                self.counts[idx] += 1

        self.counts = np.array(self.counts)
        # Zamień zera na jedynki żeby uniknąć dzielenia przez zero
        self.counts = np.where(self.counts == 0, 1, self.counts)
        print(f"Counted instances: {self.counts}")

    def calculate_weights(self):
        """
        Oblicza zagregowaną wagę dla każdej etykiety.
        
        Dla obrazów z wieloma obiektami agreguje wagi klas
        wszystkich obiektów używając funkcji agregującej (domyślnie średnia).
        
        Zwraca:
            list: Lista wag odpowiadających każdej etykiecie w zbiorze.
            
        Hierarchia wywołań:
            YOLOWeightedDataset.__init__() -> calculate_weights()
        """
        weights = []
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)

            # Domyślna waga dla pustych etykiet (tło)
            if cls.size == 0:
                weights.append(1)
                continue

            # Zagreguj wagi obiektów na obrazie
            weight = self.agg_func(self.class_weights[cls])
            weights.append(weight)
        return weights

    def calculate_probabilities(self):
        """
        Oblicza prawdopodobieństwa próbkowania na podstawie wag.
        
        Normalizuje wagi do zakresu [0, 1] tak, aby suma wynosiła 1.
        
        Zwraca:
            list: Lista prawdopodobieństw próbkowania dla każdej etykiety.
            
        Hierarchia wywołań:
            YOLOWeightedDataset.__init__() -> calculate_probabilities()
        """
        total_weight = sum(self.weights)
        probabilities = [w / total_weight for w in self.weights]
        return probabilities

    def __getitem__(self, index):
        """
        Zwraca próbkę ze zbioru danych.
        
        W trybie treningowym losuje próbkę zgodnie z prawdopodobieństwami ważonymi.
        W trybie walidacyjnym zwraca próbkę pod podanym indeksem.
        
        Argumenty:
            index (int): Indeks próbki (używany tylko w trybie walidacji).
            
        Zwraca:
            dict: Przetransformowana próbka (obraz + etykiety).
            
        Hierarchia wywołań:
            DataLoader -> YOLOWeightedDataset.__getitem__()
        """
        if not self.train_mode:
            # Walidacja - zwróć normalnie
            return self.transforms(self.get_image_and_label(index))
        else:
            # Trening - losuj z wagami
            index = np.random.choice(len(self.labels), p=self.probabilities)
            return self.transforms(self.get_image_and_label(index))