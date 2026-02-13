from .config import LIDAR_PORT
from .hardware.lidar_driver import initialize_lidar, acquire_complete_scan

# Prosty skrypt testowy
def main() -> None:
    """
    Prosty test połączenia z LiDARem.
    
    Próbuje zainicjować połączenie i pobrać jeden pełny skan (20 pakietów).
    Wypisuje statystyki pobranych danych.
    
    Hierarchia wywołań:
        lidar/src/check_lidar.py -> __main__ -> main()
    """
    print(f"Sprawdzam lidar na porcie: {LIDAR_PORT}")
    try:
        initialize_lidar(LIDAR_PORT)
        print("Połączono.")
        
        # Pobierz próbkę
        r, theta = acquire_complete_scan(num_packets=20)
        print(f"Pobrano skan. Liczba punktów: {len(r)}")
        if len(r) > 0:
            print(f"Przykładowe (r, theta): {r[0]:.2f} m, {theta[0]:.2f} rad")
            
    except Exception as e:
        print(f"Błąd: {e}")

if __name__ == "__main__":
    main()
