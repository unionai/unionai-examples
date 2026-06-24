"""
DataAgent — ingests any data source, profiles it, cleans it, and returns
a DataProfile describing the data's characteristics plus a quality score.

Supported sources:
  - CSV / Parquet / JSON (local or URL)
  - HuggingFace datasets (e.g. "hf://username/dataset" or bare "username/dataset")
  - Image folder / zip archive
  - FASTA sequences (bioinformatics)
  - Kaggle datasets (kaggle://owner/dataset)
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import urllib.request
import urllib.parse


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DataProfile:
    """Compact description of an ingested dataset, produced by DataAgent."""

    source: str = ""
    modality: str = "tabular"          # tabular | image | sequence | multimodal
    task_type: str = "classification"  # classification | regression | segmentation | clustering
    num_samples: int = 0
    num_features: int = 0
    num_classes: int = 0
    class_distribution: dict[str, int] = field(default_factory=dict)
    missing_rate: float = 0.0          # fraction of cells that were NaN (tabular only)
    target_column: str = ""
    feature_names: list[str] = field(default_factory=list)
    label_names: list[str] = field(default_factory=list)
    data_size_mb: float = 0.0
    quality_score: float = 1.0         # 0–1; penalised for missingness/imbalance/small N
    recommendations: list[str] = field(default_factory=list)
    local_data_path: str = ""          # where cleaned data lives on disk
    domain: str = "auto"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DataProfile":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# DataAgent
# ---------------------------------------------------------------------------

class DataAgent:
    """Deterministic agent: ingest → profile → clean → score."""

    def __init__(self, work_dir: str = "/tmp/automl_data"):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        dataset_link: str,
        target_column: str,
        domain: str = "auto",
    ) -> DataProfile:
        """
        Ingest the dataset, profile it, clean it, compute quality score.

        Returns a DataProfile with all characteristics and recommendations.
        """
        profile = DataProfile(source=dataset_link, target_column=target_column, domain=domain)

        # ---- 1. Detect domain-based modality hint before ingesting ----
        _IMAGE_KEYWORDS = ("image", "vision", "satellite", "photo", "visual",
                           "mri", "x-ray", "xray", "microscop", "retina", "skin")
        domain_hints_image = any(kw in domain.lower() for kw in _IMAGE_KEYWORDS)

        # ---- 2. Ingest ----
        local_path = self._ingest(dataset_link, force_image=domain_hints_image)
        profile.data_size_mb = self._dir_size_mb(local_path)

        # ---- 3. Detect modality ----
        modality = self._detect_modality(local_path)
        profile.modality = modality

        # ---- 3. Load + profile + clean ----
        if modality == "tabular":
            profile = self._profile_tabular(local_path, profile)
        elif modality == "image":
            profile = self._profile_images(local_path, profile)
        elif modality == "sequence":
            profile = self._profile_sequences(local_path, profile)
        else:
            profile = self._profile_tabular(local_path, profile)

        profile.local_data_path = str(local_path)

        # ---- 4. Detect task type ----
        if modality == "tabular" and target_column:
            profile.task_type = self._infer_task_type(local_path, target_column)
        elif modality == "image":
            profile.task_type = "classification"
        elif modality == "sequence":
            profile.task_type = "classification"

        # ---- 5. Compute quality score + recommendations ----
        profile.quality_score, profile.recommendations = self._quality_score(profile)

        return profile

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def _ingest(self, link: str, force_image: bool = False) -> Path:
        """Download / locate the dataset and return a local Path."""
        dest = self.work_dir / "raw"
        dest.mkdir(parents=True, exist_ok=True)

        if link.startswith("hf://") or self._looks_like_hf(link):
            return self._ingest_hf(link, dest, force_image=force_image)
        elif link.startswith("kaggle://"):
            return self._ingest_kaggle(link, dest)
        elif link.startswith("http://") or link.startswith("https://"):
            return self._ingest_url(link, dest)
        else:
            # Local path
            local = Path(link)
            if not local.exists():
                raise FileNotFoundError(f"Dataset not found: {link}")
            if local.suffix.lower() == ".zip":
                return self._unzip(local, dest)
            return local

    def _looks_like_hf(self, link: str) -> bool:
        """Heuristic: owner/dataset with no slashes beyond the first."""
        stripped = link.replace("hf://", "")
        parts = stripped.split("/")
        # HuggingFace dataset IDs have 1 or 2 parts (e.g. "imdb" or "user/dataset")
        return len(parts) in (1, 2) and not link.startswith("/") and not link.startswith(".")

    def _ingest_hf(self, link: str, dest: Path, force_image: bool = False) -> Path:
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError:
            raise ImportError("Install 'datasets' to use HuggingFace sources: pip install datasets")

        dataset_id = link.replace("hf://", "")
        ds = load_dataset(dataset_id)
        split_name = list(ds.keys())[0]
        split = ds[split_name]

        feat_summary = {c: type(f).__name__ for c, f in split.features.items()}
        print(f"DataAgent: HF dataset={dataset_id}  split={split_name}  features={feat_summary}", flush=True)

        # --- Three-way image column detection ---
        image_col = None
        label_col = None

        # 1. Check HuggingFace feature type names
        for col, feat in split.features.items():
            feat_type = type(feat).__name__
            if "Image" in feat_type:          # "Image", "Image_", etc.
                image_col = col
            if feat_type == "ClassLabel" or col.lower() in ("label", "labels", "category", "class"):
                label_col = col

        # 2. Fallback: check actual sample value — duck-type a PIL Image
        if image_col is None:
            sample0 = split[0]
            for col, val in sample0.items():
                if hasattr(val, "save") and hasattr(val, "mode"):   # PIL Image
                    image_col = col
                    break
                if col.lower() in ("image", "img", "pixel_values") and val is not None:
                    image_col = col
                    break

        # 3. Domain hint override — if user said "satellite imagery" etc., treat as image
        if image_col is None and force_image:
            # Best guess: first non-label column that isn't a scalar
            sample0 = split[0]
            for col, val in sample0.items():
                if col != label_col and not isinstance(val, (int, float, str, bool)):
                    image_col = col
                    print(f"DataAgent: force_image=True, guessing image_col={col!r}", flush=True)
                    break

        print(f"DataAgent: image_col={image_col}  label_col={label_col}", flush=True)

        if image_col is not None:
            import numpy as np
            from PIL import Image as PILImage  # type: ignore

            img_dir = dest / "images"
            label_names = None
            if label_col and hasattr(split.features.get(label_col), "names"):
                label_names = split.features[label_col].names

            print(f"DataAgent: saving {len(split)} images as ImageFolder → {img_dir}", flush=True)
            for i, sample in enumerate(split):
                img_obj = sample[image_col]
                lbl     = sample.get(label_col, 0) if label_col else 0
                cls     = label_names[lbl] if (label_names and isinstance(lbl, int)) else str(lbl)
                cls_dir = img_dir / cls
                cls_dir.mkdir(parents=True, exist_ok=True)

                if not isinstance(img_obj, PILImage.Image):
                    img_obj = PILImage.fromarray(np.array(img_obj))
                img_obj.convert("RGB").save(str(cls_dir / f"{i}.jpg"))

                if (i + 1) % 5000 == 0:
                    print(f"  {i + 1}/{len(split)} images saved", flush=True)

            print(f"DataAgent: ImageFolder complete  classes={list(img_dir.iterdir())[:5]}", flush=True)
            return img_dir

        # No image column — materialise as parquet
        out = dest / "hf_dataset.parquet"
        split.to_parquet(str(out))
        return out

    def _ingest_kaggle(self, link: str, dest: Path) -> Path:
        try:
            import kaggle  # type: ignore
        except ImportError:
            raise ImportError("Install 'kaggle' and set KAGGLE_USERNAME/KAGGLE_KEY env vars.")

        # kaggle://owner/dataset
        parts = link.replace("kaggle://", "").split("/")
        if len(parts) != 2:
            raise ValueError(f"Kaggle link must be kaggle://owner/dataset, got {link}")
        owner, dataset = parts
        kaggle.api.dataset_download_files(f"{owner}/{dataset}", path=str(dest), unzip=True)
        # Return the first CSV/parquet we find
        for ext in ["*.csv", "*.parquet", "*.json"]:
            matches = list(dest.glob(ext))
            if matches:
                return matches[0]
        return dest

    def _ingest_url(self, url: str, dest: Path) -> Path:
        filename = urllib.parse.urlparse(url).path.split("/")[-1] or "download"
        out_path = dest / filename
        urllib.request.urlretrieve(url, str(out_path))
        if out_path.suffix.lower() == ".zip":
            return self._unzip(out_path, dest)
        return out_path

    def _unzip(self, zip_path: Path, dest: Path) -> Path:
        extract_dir = dest / zip_path.stem
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(extract_dir))
        return extract_dir

    # ------------------------------------------------------------------
    # Modality detection
    # ------------------------------------------------------------------

    def _detect_modality(self, path: Path) -> str:
        if path.is_dir():
            exts = {p.suffix.lower() for p in path.rglob("*") if p.is_file()}
            image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
            fasta_exts = {".fasta", ".fa", ".faa", ".fna"}
            if exts & image_exts:
                return "image"
            if exts & fasta_exts:
                return "sequence"
            return "tabular"
        suffix = path.suffix.lower()
        if suffix in {".csv", ".tsv", ".parquet", ".json", ".jsonl"}:
            return "tabular"
        if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
            return "image"
        if suffix in {".fasta", ".fa", ".faa", ".fna"}:
            return "sequence"
        return "tabular"

    # ------------------------------------------------------------------
    # Tabular profiling + cleaning
    # ------------------------------------------------------------------

    def _profile_tabular(self, path: Path, profile: DataProfile) -> DataProfile:
        import pandas as pd

        df = self._load_tabular(path)
        original_shape = df.shape

        # ---- Missing values ----
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = int(df.isna().sum().sum())
        profile.missing_rate = missing_cells / max(total_cells, 1)

        # ---- Clean ----
        df = self._clean_tabular(df, profile.target_column)

        profile.num_samples = len(df)
        profile.num_features = len(df.columns) - (1 if profile.target_column in df.columns else 0)
        profile.feature_names = [c for c in df.columns if c != profile.target_column]

        # ---- Target stats ----
        if profile.target_column and profile.target_column in df.columns:
            target = df[profile.target_column]
            if target.dtype == object or target.nunique() <= 30:
                vc = target.value_counts().to_dict()
                profile.class_distribution = {str(k): int(v) for k, v in vc.items()}
                profile.num_classes = len(vc)
                profile.label_names = [str(k) for k in vc.keys()]
            else:
                profile.num_classes = 0  # continuous → regression

        # ---- Save cleaned data ----
        cleaned_path = self.work_dir / "cleaned" / "data.parquet"
        cleaned_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(str(cleaned_path), index=False)
        profile.local_data_path = str(cleaned_path)

        return profile

    def _load_tabular(self, path: Path):
        import pandas as pd
        if path.is_dir():
            # Try to find a CSV/parquet inside
            for ext, reader in [("*.parquet", pd.read_parquet), ("*.csv", pd.read_csv), ("*.json", pd.read_json)]:
                matches = list(path.rglob(ext))
                if matches:
                    return reader(str(matches[0]))
            raise FileNotFoundError(f"No tabular file found under {path}")
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return pd.read_parquet(str(path))
        elif suffix in {".csv", ".tsv"}:
            sep = "\t" if suffix == ".tsv" else ","
            return pd.read_csv(str(path), sep=sep)
        elif suffix in {".json", ".jsonl"}:
            return pd.read_json(str(path), lines=(suffix == ".jsonl"))
        else:
            # Attempt CSV as fallback
            import pandas as pd
            return pd.read_csv(str(path))

    def _clean_tabular(self, df, target_column: str):
        import pandas as pd
        import numpy as np

        # Drop columns that are entirely empty
        df = df.dropna(axis=1, how="all")

        # For numeric columns: fill missing with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if col != target_column:
                df[col] = df[col].fillna(df[col].median())

        # For categorical columns: fill missing with mode
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        for col in cat_cols:
            if col != target_column and df[col].isna().any():
                mode_val = df[col].mode()
                fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "unknown"
                df[col] = df[col].fillna(fill_val)

        # Drop rows where target is missing
        if target_column and target_column in df.columns:
            df = df.dropna(subset=[target_column])

        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Image profiling
    # ------------------------------------------------------------------

    def _profile_images(self, path: Path, profile: DataProfile) -> DataProfile:
        """
        Expects: folder of class subfolders (ImageFolder layout) or flat image folder.
        """
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        all_images = [p for p in Path(path).rglob("*") if p.suffix.lower() in image_exts]

        profile.num_samples = len(all_images)
        profile.num_features = 0  # raw pixels — no fixed count

        # Check for ImageFolder layout (subdirs = classes)
        subdirs = [d for d in Path(path).iterdir() if d.is_dir()]
        if subdirs:
            class_dist: dict[str, int] = {}
            for subdir in subdirs:
                imgs = [p for p in subdir.rglob("*") if p.suffix.lower() in image_exts]
                if imgs:
                    class_dist[subdir.name] = len(imgs)
            profile.class_distribution = class_dist
            profile.num_classes = len(class_dist)
            profile.label_names = list(class_dist.keys())

        profile.local_data_path = str(path)
        return profile

    # ------------------------------------------------------------------
    # Sequence (FASTA) profiling
    # ------------------------------------------------------------------

    def _profile_sequences(self, path: Path, profile: DataProfile) -> DataProfile:
        fasta_files = list(Path(path).rglob("*.fasta")) + list(Path(path).rglob("*.fa"))
        if not fasta_files and path.suffix.lower() in {".fasta", ".fa"}:
            fasta_files = [path]

        seq_lengths: list[int] = []
        count = 0
        for fasta_path in fasta_files:
            with open(fasta_path, "r") as fh:
                seq = ""
                for line in fh:
                    line = line.strip()
                    if line.startswith(">"):
                        if seq:
                            seq_lengths.append(len(seq))
                            count += 1
                        seq = ""
                    else:
                        seq += line
                if seq:
                    seq_lengths.append(len(seq))
                    count += 1

        profile.num_samples = count
        profile.num_features = int(sum(seq_lengths) / max(count, 1))  # avg seq length
        profile.local_data_path = str(fasta_files[0]) if fasta_files else str(path)
        return profile

    # ------------------------------------------------------------------
    # Task type inference
    # ------------------------------------------------------------------

    def _infer_task_type(self, path: Path, target_column: str) -> str:
        try:
            import pandas as pd
            import numpy as np
            df = self._load_tabular(path)
            if target_column not in df.columns:
                return "classification"
            target = df[target_column].dropna()
            if target.dtype == object:
                return "classification"
            unique_ratio = target.nunique() / max(len(target), 1)
            if unique_ratio < 0.05 or target.nunique() <= 20:
                return "classification"
            return "regression"
        except Exception:
            return "classification"

    # ------------------------------------------------------------------
    # Quality scoring
    # ------------------------------------------------------------------

    def _quality_score(self, profile: DataProfile) -> tuple[float, list[str]]:
        score = 1.0
        recs: list[str] = []

        # Penalise small datasets
        if profile.num_samples < 100:
            score -= 0.4
            recs.append("Dataset is very small (<100 samples). Consider data augmentation or collecting more data.")
        elif profile.num_samples < 1000:
            score -= 0.2
            recs.append("Dataset is small (<1000 samples). Cross-validation and regularisation are recommended.")

        # Penalise high missingness
        if profile.missing_rate > 0.5:
            score -= 0.3
            recs.append(f"Very high missingness ({profile.missing_rate:.1%}). Consider imputation or dropping high-NaN features.")
        elif profile.missing_rate > 0.1:
            score -= 0.1
            recs.append(f"Moderate missingness ({profile.missing_rate:.1%}). Advanced imputation (e.g. IterativeImputer) may improve results.")

        # Penalise class imbalance
        if profile.class_distribution and len(profile.class_distribution) >= 2:
            counts = list(profile.class_distribution.values())
            min_c, max_c = min(counts), max(counts)
            imbalance_ratio = max_c / max(min_c, 1)
            if imbalance_ratio > 10:
                score -= 0.2
                recs.append(f"Severe class imbalance (ratio {imbalance_ratio:.0f}:1). Use class weights, oversampling (SMOTE), or focal loss.")
            elif imbalance_ratio > 3:
                score -= 0.05
                recs.append(f"Moderate class imbalance (ratio {imbalance_ratio:.0f}:1). Consider stratified splits and weighted metrics.")

        # Bonus for large datasets
        if profile.num_samples > 50000:
            score = min(score + 0.05, 1.0)

        score = round(max(0.0, min(1.0, score)), 4)
        return score, recs

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _dir_size_mb(self, path: Path) -> float:
        if path.is_file():
            return path.stat().st_size / (1024 ** 2)
        total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        return round(total / (1024 ** 2), 2)
