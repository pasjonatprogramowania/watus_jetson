class LEDStatusController:
    """
    Kontroler diod LED (lub innego wskaźnika wizualnego) do sygnalizowania stanu systemu.
    Obecnie implementacja jest pusta (dummy), ale przygotowana pod przyszłą rozbudowę.
    
    Hierarchia wywołań:
        watus_main.py -> led_controller = LEDStatusController()
        watus_main.py -> indicate_listen_state() -> led_controller.indicate_listening_state()
        watus_main.py -> indicate_think_state() -> led_controller.indicate_processing_state()
    """
    
    def cleanup(self):
        """
        Sprząta zasoby (np. wyłącza diody) przy zamykaniu aplikacji.
        """
        pass

    def indicate_listening_state(self):
        """
        Sygnalizuje stan nasłuchiwania (np. kolor niebieski lub zielony).
        """
        pass

    def indicate_processing_state(self):
        """
        Sygnalizuje stan przetwarzania lub mówienia (np. pulsujący kolor lub czerwony).
        """
        pass
