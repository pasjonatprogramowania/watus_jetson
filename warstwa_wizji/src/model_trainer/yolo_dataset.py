import numpy as np
from ultralytics.data import dataset


class YOLOWeightedDataset(dataset.YOLODataset):
    def __init__(self, *args, mode="train", **kwargs):
        """
        Inicjalizuje WeightedDataset.

        Argumenty:
            class_weights (list or numpy array): Lista lub tablica wag odpowiadających każdej klasie.
        """

        super(YOLOWeightedDataset, self).__init__(*args, **kwargs)

        self.counts = None
        self.train_mode = "train" in self.prefix

        # Możesz również określić wagi ręcznie
        self.count_instances()
        class_weights = np.sum(self.counts) / self.counts

        # Funkcja agregująca
        self.agg_func = np.mean

        self.class_weights = np.array(class_weights)
        self.weights = self.calculate_weights()
        self.probabilities = self.calculate_probabilities()

    def count_instances(self):
        """
        Zlicza liczbę wystąpień dla każdej klasy.

        Zwraca:
            dict: Słownik zawierający liczniki dla każdej klasy.
        """
        self.counts = [0 for i in range(len(self.data["names"]))]
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)
            for idx in cls:
                self.counts[idx] += 1

        self.counts = np.array(self.counts)
        self.counts = np.where(self.counts == 0, 1, self.counts)
        print(f"Counted instances: {self.counts}")

    def calculate_weights(self):
        """
        Oblicza zagregowaną wagę dla każdej etykiety na podstawie wag klas.

        Zwraca:
            list: Lista zagregowanych wag odpowiadających każdej etykiecie.
        """
        weights = []
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)

            # Daj domyślną wagę dla klasy tła
            if cls.size == 0:
                weights.append(1)
                continue

            # Weź średnią ze wag
            # Możesz zmienić tę funkcję agregacji wag, aby agregować wagi inaczej
            weight = self.agg_func(self.class_weights[cls])
            weights.append(weight)
        return weights

    def calculate_probabilities(self):
        """
        Oblicza i przechowuje prawdopodobieństwa próbkowania na podstawie wag.

        Zwraca:
            list: Lista prawdopodobieństw próbkowania odpowiadających każdej etykiecie.
        """
        total_weight = sum(self.weights)
        probabilities = [w / total_weight for w in self.weights]
        return probabilities

    def __getitem__(self, index):
        """
        Zwraca przetransformowane informacje o etykiecie na podstawie indeksu próbkowania.
        """
        # Nie używaj do walidacji
        if not self.train_mode:
            return self.transforms(self.get_image_and_label(index))
        else:
            index = np.random.choice(len(self.labels), p=self.probabilities)
            return self.transforms(self.get_image_and_label(index))