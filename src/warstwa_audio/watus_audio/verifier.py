import time
import numpy as np
from collections import deque
from .common import log_message
from . import config

class _NoopVerifier:
    """
    Pusta implementacja weryfikatora, używana gdy weryfikacja jest wyłączona lub brak zależności.
    """
    enabled = True

    def __init__(self): self._enrolled_embedding = None

    @property
    def enrolled(self): return False

    def enroll_voice_samples(self, audio_samples, sample_rate_hz): pass

    def adaptive_update(self, audio_samples, sample_rate_hz, score): pass

    def verify_speaker_identity(self, audio_samples, sample_rate_hz, audio_volume_dbfs): return {"enabled": False}


def create_speaker_verifier():
    """
    Tworzy i zwraca instancję weryfikatora mówcy (Speaker Verification).
    Jeśli biblioteki (torch, speechbrain) nie są dostępne, zwraca wersję Noop.

    Zwraca:
        Object: Instancja _SbVerifier lub _NoopVerifier.

    Hierarchia wywołań:
        watus_main.py -> main() -> create_speaker_verifier()
    """
    if not config.SPEAKER_VERIFY: return _NoopVerifier()
    try:
        import torch  # noqa
        from speechbrain.pretrained import EncoderClassifier  # noqa
    except Exception as e:
        log_message(f"[Watus][SPK] OFF (brak zależności): {e}")
        return _NoopVerifier()

    class _SbVerifier:
        """
        Weryfikator mówcy oparty na SpeechBrain (ECAPA-TDNN).
        Uśrednia embeddingi z wielu próbek głosu dla stabilniejszej weryfikacji.
        """
        enabled = True

        def __init__(self):
            import torch
            self.threshold = config.SPEAKER_THRESHOLD
            self.sticky_threshold = config.SPEAKER_STICKY_THRESHOLD
            self.back_threshold = config.SPEAKER_BACK_THRESHOLD
            self.grace_period = config.SPEAKER_GRACE
            self.sticky_seconds = config.SPEAKER_STICKY_SEC
            self._speaker_encoder_classifier = None
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            # Historia embeddingów do uśredniania
            self._embedding_history = deque(maxlen=config.SPEAKER_EMBEDDING_HISTORY_SIZE)
            self._mean_embedding = None
            self._base_embedding = None  # Używany do generowania stabilnego hash ID
            self._enroll_timestamp = 0.0

        @property
        def enrolled(self):
            """Czy wzorzec głosu lidera jest zarejestrowany?"""
            return self._mean_embedding is not None

        def _ensure_model_loaded(self):
            """
            Ładuje model SpeechBrain jeśli nie jest jeszcze załadowany.
            
            Hierarchia wywołań:
                verifier.py -> _compute_embedding() -> _ensure_model_loaded()
            """
            from speechbrain.pretrained import EncoderClassifier
            if self._speaker_encoder_classifier is None:
                self._speaker_encoder_classifier = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    run_opts={"device": self._device},
                )

        @staticmethod
        def _resample_to_16k(audio_samples: np.ndarray, sample_rate_hz: int) -> np.ndarray:
            """
            Konwertuje próbki audio do 16kHz (wymagane przez model).
            
            Argumenty:
                audio_samples (np.ndarray): Próbki audio.
                sample_rate_hz (int): Oryginalna częstotliwość próbkowania.
                
            Zwraca:
                np.ndarray: Próbki audio 16kHz.
                
            Hierarchia wywołań:
                verifier.py -> _compute_embedding() -> _resample_to_16k()
            """
            if sample_rate_hz == 16000: return audio_samples.astype(np.float32)
            ratio = 16000.0 / sample_rate_hz
            num_output_samples = int(round(len(audio_samples) * ratio))
            output_indices = np.linspace(0, len(audio_samples) - 1, num=num_output_samples, dtype=np.float32)
            input_indices = np.arange(len(audio_samples), dtype=np.float32)
            return np.interp(output_indices, input_indices, audio_samples).astype(np.float32)

        def _compute_embedding(self, audio_samples: np.ndarray, sample_rate_hz: int):
            """
            Oblicza wektor cech (embedding) dla podanych próbek audio.
            
            Argumenty:
                audio_samples (np.ndarray): Próbki audio.
                sample_rate_hz (int): Częstotliwość próbkowania.
                
            Zwraca:
                np.ndarray: Wektor embeddingu.
                
            Hierarchia wywołań:
                verifier.py -> enroll_voice_samples() -> _compute_embedding()
                verifier.py -> verify_speaker_identity() -> _compute_embedding()
            """
            import torch
            self._ensure_model_loaded()
            wav_resampled = self._resample_to_16k(audio_samples, sample_rate_hz)
            tensor_input = torch.tensor(wav_resampled, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                embedding_vector = self._speaker_encoder_classifier.encode_batch(tensor_input).squeeze(0).squeeze(0)
            return embedding_vector.detach().cpu().numpy().astype(np.float32)

        def _update_mean_embedding(self):
            """
            Przelicza średni embedding z historii próbek.
            Uśrednianie stabilizuje wektor referencyjny i zwiększa dokładność weryfikacji.
            """
            if not self._embedding_history:
                self._mean_embedding = None
                return
            stacked = np.stack(list(self._embedding_history), axis=0)
            self._mean_embedding = np.mean(stacked, axis=0).astype(np.float32)

        def enroll_voice_samples(self, audio_samples: np.ndarray, sample_rate_hz: int):
            """
            Rejestruje próbki głosu jako wzorzec lidera.
            Dodaje embedding do historii i przelicza średnią.
            
            Argumenty:
                audio_samples (np.ndarray): Próbki audio (float32).
                sample_rate_hz (int): Częstotliwość próbkowania.
                
            Hierarchia wywołań:
                watus_main.py -> main() -> enroll_voice_samples() (via command "enroll")
                stt.py -> _process_recorded_speech_segment() -> enroll_voice_samples()
            """
            try:
                embedding_vector = self._compute_embedding(audio_samples, sample_rate_hz)
                if self._base_embedding is None:
                    self._base_embedding = embedding_vector
                self._embedding_history.append(embedding_vector)
                self._update_mean_embedding()
                self._enroll_timestamp = time.time()
                log_message(f"[Watus][SPK] Enrolled new leader voice. (history={len(self._embedding_history)})")
            except Exception as e:
                log_message(f"[Watus][SPK] enroll err: {e}")

        def adaptive_update(self, audio_samples: np.ndarray, sample_rate_hz: int, score: float):
            """
            Adaptacyjne uczenie — po udanej weryfikacji dodaje embedding do historii,
            żeby profil głosu stawał się coraz dokładniejszy.
            
            Argumenty:
                audio_samples (np.ndarray): Próbki audio.
                sample_rate_hz (int): Częstotliwość próbkowania.
                score (float): Score z ostatniej weryfikacji.
                
            Hierarchia wywołań:
                stt.py -> _process_recorded_speech_segment() -> adaptive_update()
            """
            if not config.SPEAKER_ADAPTIVE_LEARN:
                return
            if score < config.SPEAKER_ADAPTIVE_MIN_SCORE:
                return
            try:
                embedding_vector = self._compute_embedding(audio_samples, sample_rate_hz)
                self._embedding_history.append(embedding_vector)
                self._update_mean_embedding()
                self._enroll_timestamp = time.time()
                log_message(f"[Watus][SPK] Adaptive update (score={score:.3f}, history={len(self._embedding_history)})")
            except Exception as e:
                log_message(f"[Watus][SPK] adaptive err: {e}")

        def verify_speaker_identity(self, audio_samples: np.ndarray, sample_rate_hz: int, audio_volume_dbfs: float) -> dict:
            """
            Weryfikuje, czy podane próbki audio należą do zarejestrowanego lidera.
            Porównuje z uśrednionym embeddingiem z historii próbek.
            
            Argumenty:
                audio_samples (np.ndarray): Próbki audio.
                sample_rate_hz (int): Częstotliwość próbkowania.
                audio_volume_dbfs (float): Głośność próbki (dBFS).
                
            Zwraca:
                dict: Wynik weryfikacji (score, is_leader).
                
            Hierarchia wywołań:
                stt.py -> SpeechToTextProcessingEngine._process_recorded_speech_segment() -> verify_speaker_identity()
            """
            if self._mean_embedding is None:
                return {"enabled": True, "enrolled": False}
            import torch, torch.nn.functional as F
            current_embedding = self._compute_embedding(audio_samples, sample_rate_hz)
            similarity_score = float(F.cosine_similarity(
                torch.tensor(current_embedding, dtype=torch.float32).flatten(),
                torch.tensor(self._mean_embedding, dtype=torch.float32).flatten(), dim=0, eps=1e-8
            ).detach().cpu().item())
            
            current_time = time.time()
            age_seconds = current_time - self._enroll_timestamp
            is_leader = False
            
            adjusted_threshold = (
                        self.sticky_threshold - self.grace_period) if audio_volume_dbfs > -22.0 else self.sticky_threshold  # emocje → głośniej → trochę łagodniej
            
            # Znacznie łagodniejsze wymagania przez pierwszą minutę rozmowy (sticky_seconds)
            # ECAPA radzi sobie gorzej z bardzo krótkimi wypowiedziami (rzędu 2s).
            if age_seconds <= self.sticky_seconds and similarity_score >= (adjusted_threshold - 0.25):
                is_leader = True
            elif similarity_score >= self.threshold:
                is_leader = True
            elif similarity_score >= self.back_threshold and age_seconds <= self.sticky_seconds:
                is_leader = True
            
            return {"enabled": True, "enrolled": True, "score": similarity_score, "is_leader": bool(is_leader), "sticky_age_s": age_seconds}

    return _SbVerifier()
