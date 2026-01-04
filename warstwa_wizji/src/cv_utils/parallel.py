"""
Moduł pomocniczy do przetwarzania równoległego.

Zawiera klasę bazową do tworzenia procesów w tle do równoległego
przetwarzania zadań wizyjnych.

Hierarchia wywołań:
    warstwa_wizji/src/cv_utils/parallel.py -> ParallelProcess (bazowa)
"""

import multiprocessing
import time


class ParallelProcess(multiprocessing.Process):
    """
    Klasa bazowa procesu do przetwarzania równoległego.
    
    Umożliwia uruchamianie funkcji w osobnym procesie systemowym.
    
    Atrybuty:
        process_id (int): Unikalny identyfikator procesu.
        
    Hierarchia wywołań:
        Używane jako klasa bazowa dla procesów przetwarzania wizyjnego.
    """
    
    def __init__(self, process_id: int):
        """
        Inicjalizuje proces równoległy.
        
        Argumenty:
            process_id (int): Unikalny identyfikator procesu.
        """
        super(ParallelProcess, self).__init__()
        self.process_id = process_id

    def run(self, target_function=None, args=None, kwargs=None):
        """
        Uruchamia podaną funkcję w procesie.
        
        Argumenty:
            target_function: Funkcja do wywołania.
            args: Krotka argumentów pozycyjnych.
            kwargs: Słownik argumentów nazwanych.
        """
        if target_function is not None:
            target_function(*args, **kwargs)


# Alias dla kompatybilności wstecznej
Process = ParallelProcess