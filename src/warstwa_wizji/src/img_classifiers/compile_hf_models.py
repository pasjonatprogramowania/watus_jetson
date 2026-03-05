"""
Skrypt kompilacji modeli HuggingFace do formatu ONNX.

Eksportuje klasyfikatory (emocje, płeć, wiek, typ ubrań, wzór ubrań)
z formatu PyTorch do ONNX i zapisuje w katalogu src/models/.

Uruchomienie:
    python -m warstwa_wizji.src.img_classifiers.compile_hf_models
    python -m warstwa_wizji.src.img_classifiers.compile_hf_models --output-dir ./my_models
"""

import os
import json
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from transformers import (
    AutoImageProcessor,
    SiglipForImageClassification,
    ViTImageProcessor,
    ViTForImageClassification,
)


# Domyślny katalog wyjściowy
_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_OUTPUT_DIR = _THIS_DIR.parent.parent.parent / "models"

# Rejestr modeli do eksportu
MODELS_REGISTRY = [
    {
        "name": "emotion",
        "hf_id": "abhilash88/face-emotion-detection",
        "arch": "vit",
        "id2label": {
            "0": "Angry", "1": "Disgust", "2": "Fear", "3": "Happy",
            "4": "Sad", "5": "Surprise", "6": "Neutral",
        },
    },
    {
        "name": "gender",
        "hf_id": "prithivMLmods/Realistic-Gender-Classification",
        "arch": "siglip",
        "id2label": {"0": "female", "1": "male"},
    },
    {
        "name": "age",
        "hf_id": "prithivMLmods/open-age-detection",
        "arch": "siglip",
        "id2label": {
            "0": "Child 0-12", "1": "Teenager 13-20", "2": "Adult 21-44",
            "3": "Middle Age 45-64", "4": "Aged 65+",
        },
    },
    {
        "name": "clothes_type",
        "hf_id": "samokosik/finetuned-clothes",
        "arch": "vit",
        "id2label": {
            "0": "Hat", "1": "Longsleeve", "2": "Outwear", "3": "Pants",
            "4": "Shoes", "5": "Shorts", "6": "Shortsleeve",
        },
    },
    {
        "name": "clothes_pattern",
        "hf_id": "IrshadG/Clothes_Pattern_Classification_v2",
        "arch": "vit",
        "id2label": {
            "0": "Argyle", "1": "Check", "2": "Colour blocking", "3": "Denim",
            "4": "Dot", "5": "Embroidery", "6": "Lace", "7": "Metallic",
            "8": "Patterns", "9": "Placement print", "10": "Sequin",
            "11": "Solid", "12": "Stripe", "13": "Transparent",
        },
    },
]


def _load_model_and_processor(entry: dict):
    """Ładuje model i procesor z HuggingFace Hub."""
    hf_id = entry["hf_id"]
    arch = entry["arch"]

    if arch == "vit":
        processor = ViTImageProcessor.from_pretrained(hf_id, use_fast=True)
        model = ViTForImageClassification.from_pretrained(hf_id)
    else:  # siglip
        processor = AutoImageProcessor.from_pretrained(hf_id, use_fast=True)
        model = SiglipForImageClassification.from_pretrained(hf_id)

    model.eval()
    return model, processor


def export_single_model(
    entry: dict,
    output_dir: Path,
    opset_version: int = 14,
    verify: bool = True,
) -> Path:
    """
    Eksportuje jeden model HuggingFace do ONNX.

    Argumenty:
        entry: Słownik z rejestru modeli (name, hf_id, arch, id2label).
        output_dir: Katalog wyjściowy dla plików .onnx i .json.
        opset_version: Wersja ONNX opset (domyślnie 14).
        verify: Czy zweryfikować wynik ONNX vs PyTorch.

    Zwraca:
        Path: Ścieżka do wygenerowanego pliku .onnx.
    """
    name = entry["name"]
    print(f"\n{'='*60}")
    print(f"[ONNX] Eksport: {name} ({entry['hf_id']})")
    print(f"{'='*60}")

    # 1. Ładowanie modelu
    print(f"[ONNX] Ładowanie modelu z HuggingFace Hub...")
    model, processor = _load_model_and_processor(entry)

    # 2. Tworzenie dummy input
    dummy_img = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    inputs = processor(images=dummy_img, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    print(f"[ONNX] Input shape: {pixel_values.shape}")

    # 3. Eksport ONNX
    onnx_path = output_dir / f"{name}.onnx"
    print(f"[ONNX] Eksportowanie do {onnx_path}...")

    torch.onnx.export(
        model,
        (pixel_values,),
        str(onnx_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"[ONNX] Zapisano: {onnx_path} ({onnx_size_mb:.1f} MB)")

    # 4. Metadata JSON (procesor + id2label)
    meta_path = output_dir / f"{name}_meta.json"
    meta = {
        "hf_id": entry["hf_id"],
        "arch": entry["arch"],
        "id2label": entry["id2label"],
        "input_names": ["pixel_values"],
        "output_names": ["logits"],
        "onnx_file": onnx_path.name,
    }

    # Zapisz parametry preprocesingu z procesora
    if hasattr(processor, "do_resize"):
        meta["do_resize"] = processor.do_resize
    if hasattr(processor, "size"):
        meta["size"] = (
            processor.size
            if isinstance(processor.size, dict)
            else {"height": processor.size, "width": processor.size}
        )
    if hasattr(processor, "do_normalize"):
        meta["do_normalize"] = processor.do_normalize
    if hasattr(processor, "image_mean"):
        mean = processor.image_mean
        meta["image_mean"] = list(mean) if hasattr(mean, "__iter__") else mean
    if hasattr(processor, "image_std"):
        std = processor.image_std
        meta["image_std"] = list(std) if hasattr(std, "__iter__") else std

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[ONNX] Metadata: {meta_path}")

    # 5. Opcjonalna weryfikacja
    if verify:
        _verify_onnx(model, processor, onnx_path, entry)

    return onnx_path


def _verify_onnx(model, processor, onnx_path: Path, entry: dict):
    """Porównuje wyniki PyTorch i ONNX Runtime."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[ONNX] onnxruntime nie zainstalowany — pomijam weryfikację")
        return

    print("[ONNX] Weryfikacja ONNX vs PyTorch...")

    # Inferencja PyTorch
    dummy_img = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    inputs = processor(images=dummy_img, return_tensors="pt")
    pixel_values = inputs["pixel_values"]

    with torch.no_grad():
        pt_logits = model(pixel_values).logits.numpy()

    # Inferencja ONNX
    session = ort.InferenceSession(str(onnx_path))
    ort_inputs = {"pixel_values": pixel_values.numpy()}
    ort_logits = session.run(["logits"], ort_inputs)[0]

    # Porównanie
    max_diff = np.max(np.abs(pt_logits - ort_logits))
    print(f"[ONNX] Max różnica logitów: {max_diff:.6f}")

    if max_diff < 1e-4:
        print(f"[ONNX] ✅ Weryfikacja OK ({entry['name']})")
    else:
        print(f"[ONNX] ⚠️  Różnica > 1e-4, ale model wyeksportowany ({entry['name']})")


def export_all(output_dir: Path, opset_version: int = 14, verify: bool = True):
    """Eksportuje wszystkie modele z rejestru."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ONNX] Katalog wyjściowy: {output_dir}")
    print(f"[ONNX] Modele do eksportu: {len(MODELS_REGISTRY)}")

    exported = []
    for entry in MODELS_REGISTRY:
        try:
            path = export_single_model(entry, output_dir, opset_version, verify)
            exported.append((entry["name"], str(path)))
        except Exception as e:
            print(f"[ONNX] ❌ Błąd eksportu {entry['name']}: {e}")

    print(f"\n{'='*60}")
    print(f"[ONNX] Podsumowanie: {len(exported)}/{len(MODELS_REGISTRY)} wyeksportowano")
    for name, path in exported:
        print(f"  ✅ {name}: {path}")
    print(f"{'='*60}")
    return exported


def parse_args():
    ap = argparse.ArgumentParser(description="Eksport modeli HuggingFace do ONNX")
    ap.add_argument(
        "--output-dir", type=str, default=str(_DEFAULT_OUTPUT_DIR),
        help=f"Katalog wyjściowy (domyślnie: {_DEFAULT_OUTPUT_DIR})",
    )
    ap.add_argument(
        "--opset", type=int, default=14,
        help="Wersja ONNX opset (domyślnie: 14)",
    )
    ap.add_argument(
        "--no-verify", action="store_true",
        help="Pomiń weryfikację ONNX vs PyTorch",
    )
    ap.add_argument(
        "--model", type=str, default=None,
        choices=[m["name"] for m in MODELS_REGISTRY],
        help="Eksportuj tylko wybrany model (domyślnie: wszystkie)",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.output_dir)

    if args.model:
        entry = next(m for m in MODELS_REGISTRY if m["name"] == args.model)
        out_dir.mkdir(parents=True, exist_ok=True)
        export_single_model(entry, out_dir, args.opset, not args.no_verify)
    else:
        export_all(out_dir, args.opset, not args.no_verify)
