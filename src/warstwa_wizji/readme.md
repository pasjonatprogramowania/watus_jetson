Wklep komendę:
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.* torchvision==0.21.* torchaudio==2.6.*

Następnie zainstaluj i wypakuj TensorRT ze strony Nvidia. W katalogu /python z aktywowanym środowiskiem wprowadź komendę
```bash
pip install tensorrt-[WERSJA]-cp[WERSJA_PYTHON]-none-win_amd64.whl
```

Wykonanie testów jednostkowych i integracyjnych

```bash
python -m unittest discover -s tests -v 
```