import numpy as np

# 1. Sprawdz czy zaleznosci sie importuja
try:
    import torch
    from speechbrain.pretrained import EncoderClassifier
    print(f"[OK] torch={torch.__version__}, CUDA={torch.cuda.is_available()}")
except ImportError as e:
    print(f"[BLAD] Brak zaleznosci: {e}")
    exit(1)

# 2. Zaladuj model (pierwsze uruchomienie = pobieranie ~90MB)
print("[...] Ladowanie modelu ECAPA-TDNN...")
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
)
print("[OK] Model zaladowany!")

# 3. Test z losowym audio (16kHz, 3 sekundy)
fake_audio = torch.randn(1, 48000)  # 3s @ 16kHz
embedding = classifier.encode_batch(fake_audio)
print(f"[OK] Embedding shape: {embedding.shape}")  # Powinno byc [1, 1, 192]

# 4. Test porownania dwoch probek
audio_a = torch.randn(1, 48000)
audio_b = torch.randn(1, 48000)
emb_a = classifier.encode_batch(audio_a).squeeze()
emb_b = classifier.encode_batch(audio_b).squeeze()

similarity = torch.nn.functional.cosine_similarity(emb_a, emb_b, dim=0)
print(f"[OK] Cosine similarity (losowe audio): {similarity.item():.4f}")
print("     (powinno byc niskie, bo to losowy szum)")

print("\n[SUKCES] ECAPA-TDNN dziala poprawnie!")
