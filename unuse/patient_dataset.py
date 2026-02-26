# utils/patient_dataset.py
import os
import json
from typing import List, Tuple, Dict, Any

import torch
import dgl


class PatientGraphDataset:
    def __init__(self, patient_graphs_bin: str, patient_graphs_meta: str):
        self.patient_graphs_bin = patient_graphs_bin
        self.patient_graphs_meta = patient_graphs_meta

        self.graphs, _ = dgl.load_graphs(patient_graphs_bin)
        with open(patient_graphs_meta, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.patients = self.meta.get("patients", [])
        assert len(self.graphs) == len(self.patients), "graphs length mismatch with patients list"

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx: int) -> dgl.DGLHeteroGraph:
        return self.graphs[idx]


def load_split(split_pt: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sp = torch.load(split_pt, map_location="cpu")
    return sp["train_idx"].long(), sp["val_idx"].long(), sp["test_idx"].long()
