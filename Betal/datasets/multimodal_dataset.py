"""
Multimodal Dataset Loader.

Loads and aligns per-patient data from four modalities:
  - Audio  : raw waveform from .wav files via torchaudio
  - Text   : transcript strings tokenised with BertTokenizer
  - MRI    : 3-D NIfTI volumes loaded via nibabel
  - EHR    : tabular features from a metadata CSV

The CSV (metadata_path) must contain the columns:
  patient_id, audio_path, text_path, mri_path, label
  plus any number of columns prefixed with "ehr_" for clinical features.

Usage
-----
train_ds = MultimodalDataset(metadata_path, data_dir, is_train=True)
val_ds   = MultimodalDataset(metadata_path, data_dir, is_train=False,
                             preprocessor=train_ds.preprocessor)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from transformers import BertTokenizer


class MultimodalDataset(Dataset):
    """PyTorch Dataset for four-modality Alzheimer / MCI data."""

    # Fixed tokeniser across all instances (loaded once at class level)
    _tokenizer: Optional[BertTokenizer] = None

    def __init__(
        self,
        metadata_path: str,
        data_dir: str,
        text_model_name: str = "bert-base-uncased",
        max_text_len: int = 128,
        target_sr: int = 16_000,
        mri_target_shape: tuple[int, int, int] = (64, 64, 64),
        is_train: bool = True,
        preprocessor: Optional[ColumnTransformer] = None,
    ) -> None:
        """
        Args:
            metadata_path   : Path to CSV with columns described in the module docstring.
            data_dir        : Root directory; paths in the CSV are relative to this.
            text_model_name : HuggingFace model name for the BertTokenizer.
            max_text_len    : Maximum token sequence length for BERT.
            target_sr       : Target sample rate for audio resampling.
            mri_target_shape: (D, H, W) to which all MRI volumes are resized via slicing.
            is_train        : If True, fit the EHR preprocessor; else use ``preprocessor``.
            preprocessor    : Pre-fitted ColumnTransformer (pass for val/test splits).
        """
        super().__init__()
        np.random.seed(42)  # Reproducibility

        self.metadata = pd.read_csv(metadata_path)
        self.data_dir = Path(data_dir)
        self.max_text_len = max_text_len
        self.target_sr = target_sr
        self.mri_target_shape = mri_target_shape

        # ── EHR Preprocessing (sklearn Pipeline, no leakage) ───────────────
        self.ehr_cols = [c for c in self.metadata.columns if c.startswith("ehr_")]

        if is_train and preprocessor is None:
            # Fit ONLY on training data — transforms val/test via the returned object
            self.preprocessor: ColumnTransformer = ColumnTransformer(
                [("num", StandardScaler(), self.ehr_cols)],
                remainder="passthrough",
            )
            self.metadata[self.ehr_cols] = self.preprocessor.fit_transform(
                self.metadata[self.ehr_cols]
            )
        elif preprocessor is not None:
            self.preprocessor = preprocessor
            self.metadata[self.ehr_cols] = self.preprocessor.transform(
                self.metadata[self.ehr_cols]
            )
        else:
            # Inference mode — no preprocessing fitted
            self.preprocessor = None

        # ── Lazy-load tokenizer once ────────────────────────────────────────
        if MultimodalDataset._tokenizer is None:
            MultimodalDataset._tokenizer = BertTokenizer.from_pretrained(
                text_model_name
            )

    # ── Dunder helpers ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.metadata.iloc[idx]

        audio = self._load_audio(row["audio_path"])
        text_ids, text_mask = self._load_text(row["text_path"])
        mri = self._load_mri(row["mri_path"])
        ehr = self._load_ehr(row)
        label = torch.tensor(int(row["label"]), dtype=torch.long)

        return {
            "audio": audio,          # (1, T)
            "text_ids": text_ids,    # (max_text_len,)
            "text_mask": text_mask,  # (max_text_len,)
            "mri": mri,              # (1, D, H, W)
            "ehr": ehr,              # (n_ehr_features,)
            "label": label,          # scalar
        }

    # ── Private loaders ─────────────────────────────────────────────────────

    def _load_audio(self, rel_path: str) -> torch.Tensor:
        """Load a .wav file, resample to target_sr, return mono (1, T)."""
        full_path = self.data_dir / rel_path
        waveform, sr = torchaudio.load(str(full_path))
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)
        # Convert to mono by averaging channels
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform  # (1, T)

    def _load_text(self, rel_path: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Read a transcript .txt file and tokenise with BERT tokenizer."""
        full_path = self.data_dir / rel_path
        text = full_path.read_text(encoding="utf-8").strip()
        encoding = self._tokenizer(
            text,
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)       # (max_text_len,)
        attention_mask = encoding["attention_mask"].squeeze(0)  # (max_text_len,)
        return input_ids, attention_mask

    def _load_mri(self, rel_path: str) -> torch.Tensor:
        """Load a NIfTI MRI volume, centre-crop/pad to mri_target_shape, return (1, D, H, W)."""
        full_path = self.data_dir / rel_path
        img = nib.load(str(full_path))
        volume: np.ndarray = img.get_fdata(dtype=np.float32)

        # Normalise to [0, 1]
        v_min, v_max = volume.min(), volume.max()
        if v_max > v_min:
            volume = (volume - v_min) / (v_max - v_min)

        # Centre-crop or zero-pad each spatial dimension to target shape
        volume = self._crop_or_pad_volume(volume, self.mri_target_shape)

        tensor = torch.from_numpy(volume).unsqueeze(0)  # (1, D, H, W)
        return tensor

    @staticmethod
    def _crop_or_pad_volume(
        volume: np.ndarray, target: tuple[int, int, int]
    ) -> np.ndarray:
        """Centre-crop or zero-pad a 3-D array to target (D, H, W)."""
        result = np.zeros(target, dtype=np.float32)
        for dim in range(3):
            src_size = volume.shape[dim]
            tgt_size = target[dim]
            if src_size >= tgt_size:
                start = (src_size - tgt_size) // 2
                volume = np.take(volume, range(start, start + tgt_size), axis=dim)
            else:
                pad_total = tgt_size - src_size
                pad_before = pad_total // 2
                pad_after = pad_total - pad_before
                pad_width = [(0, 0)] * 3
                pad_width[dim] = (pad_before, pad_after)
                volume = np.pad(volume, pad_width, mode="constant")
        return volume

    def _load_ehr(self, row: pd.Series) -> torch.Tensor:
        """Return the pre-scaled EHR features as a float32 tensor."""
        values = row[self.ehr_cols].values.astype(np.float32)
        return torch.from_numpy(values)
