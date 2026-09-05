"""
M-2LRF Declarative Configuration Schema (Axolotl-Inspired)
===========================================================
YAML/JSON declarative configuration validator.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import json

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class QuantConfig:
    method: str = "m2lrf_2bit"
    rank: int = 64
    alpha: float = 64.0
    use_hadamard: bool = True
    block_size: int = 64
    group_size: Optional[int] = 128
    loftq_iters: int = 1
    target_avg_bits: Optional[float] = 2.0
    double_quant: bool = False


@dataclass
class DatasetConfig:
    path: str = "tatsu-lab/alpaca"
    type: str = "alpaca"
    split: str = "train"
    sample_packing: bool = True
    max_seq_length: int = 4096


@dataclass
class TrainingArgsConfig:
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_train_epochs: int = 1
    max_steps: Optional[int] = None
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    logging_steps: int = 10
    output_dir: str = "./outputs"


@dataclass
class M2LRFConfig:
    base_model: str
    model_type: Optional[str] = None
    quantization: QuantConfig = field(default_factory=QuantConfig)
    datasets: List[DatasetConfig] = field(default_factory=lambda: [DatasetConfig()])
    training: TrainingArgsConfig = field(default_factory=TrainingArgsConfig)
    export_format: Optional[str] = "hf"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "M2LRFConfig":
        base_model = data["base_model"]
        model_type = data.get("model_type")
        
        # Quantization
        q_data = data.get("quantization", {})
        quant = QuantConfig(**q_data) if isinstance(q_data, dict) else QuantConfig()

        # Datasets
        d_data = data.get("datasets", [{}])
        datasets = [DatasetConfig(**d) if isinstance(d, dict) else DatasetConfig() for d in d_data]

        # Training
        t_data = data.get("training", {})
        training = TrainingArgsConfig(**t_data) if isinstance(t_data, dict) else TrainingArgsConfig()

        return cls(
            base_model=base_model,
            model_type=model_type,
            quantization=quant,
            datasets=datasets,
            training=training,
            export_format=data.get("export_format", "hf")
        )

    @classmethod
    def from_yaml(cls, file_path: str) -> "M2LRFConfig":
        with open(file_path, "r", encoding="utf-8") as f:
            if HAS_YAML:
                data = yaml.safe_load(f)
            else:
                # Fallback to json parser if file is json formatted
                data = json.load(f)
        return cls.from_dict(data)
