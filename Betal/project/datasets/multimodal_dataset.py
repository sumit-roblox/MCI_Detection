"""
Multimodal Dataset Loader.
Loads Audio, Text, MRI, and EHR data and aligns them per patient.
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

class MultimodalDataset(Dataset):
    def __init__(self, metadata_path: str, data_dir: str, is_train: bool = True, preprocessor=None):
        """
        Args:
            metadata_path: Path to CSV mapping patient IDs to paths/labels.
            data_dir: Root directory for data.
            is_train: Boolean to fit preprocessing on training data only.
            preprocessor: Fitted ColumnTransformer for validation/test sets.
        """
        self.metadata = pd.read_csv(metadata_path)
        self.data_dir = Path(data_dir)
        
        # Enforce explicit random seed for reproducibility
        np.random.seed(42)
        
        ehr_cols = [col for col in self.metadata.columns if col.startswith('ehr_')]
        
        if is_train and preprocessor is None:
            # Using ColumnTransformer to prevent data leakage between train/test
            self.preprocessor = ColumnTransformer([
                ('num', StandardScaler(), ehr_cols)
            ], remainder='passthrough')
            self.metadata[ehr_cols] = self.preprocessor.fit_transform(self.metadata[ehr_cols])
        elif preprocessor is not None:
            self.preprocessor = preprocessor
            self.metadata[ehr_cols] = self.preprocessor.transform(self.metadata[ehr_cols])
            
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        ehr_cols = [col for col in self.metadata.columns if col.startswith('ehr_')]
        
        # Load EHR
        ehr_tensor = torch.tensor(row[ehr_cols].values, dtype=torch.float32)
        
        # Load audio (placeholder)
        audio_tensor = torch.randn(1, 16000) # Dummy audio data
        
        # Load text (placeholder)
        text_ids = torch.randint(0, 30000, (128,)) # Dummy token IDs
        text_mask = torch.ones(128)
        
        # Load MRI (placeholder)
        mri_tensor = torch.randn(1, 32, 32, 32) # Dummy 3D MRI
        
        label = torch.tensor(int(row['label']), dtype=torch.long)
        
        return {
            "audio": audio_tensor,
            "text_ids": text_ids,
            "text_mask": text_mask,
            "mri": mri_tensor,
            "ehr": ehr_tensor,
            "label": label
        }
