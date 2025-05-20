"""Tests for the ollama.utils module."""

import json
import os
import sys
import pytest
from dataclasses import is_dataclass, dataclass, field
from unittest.mock import MagicMock

# Create mock modules
sys.modules['flytekit'] = MagicMock()
sys.modules['mashumaro'] = MagicMock()
sys.modules['mashumaro.mixins'] = MagicMock()
sys.modules['mashumaro.mixins.json'] = MagicMock()
sys.modules['ollama'] = MagicMock()
sys.modules['ollama.constants'] = MagicMock()
sys.modules['ollama.constants'].REGISTRY = "mock-registry"

# Mock implementation of ImageSpec
class MockImageSpec:
    def __init__(self, name, **kwargs):
        self.name = name
        self.__dict__.update(kwargs)
        self._commands = []
        self.apt_packages = kwargs.get('apt_packages', [])
        self.packages = kwargs.get('packages', [])
        self.registry = kwargs.get('registry', '')
        self.requirements = kwargs.get('requirements', '')
        self.python_version = kwargs.get('python_version', '')
    
    def with_commands(self, commands):
        self._commands = commands
        return self

sys.modules['flytekit'].ImageSpec = MockImageSpec

# Mock implementation of DataClassJSONMixin
@dataclass
class DataClassJSONMixin:
    def to_json(self):
        return json.dumps({k: v for k, v in self.__dict__.items() if not k.startswith('_')})
    
    @classmethod
    def from_json(cls, json_str):
        return cls(**json.loads(json_str))

sys.modules['mashumaro.mixins.json'].DataClassJSONMixin = DataClassJSONMixin


# Now import the module under test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from _blogs.ollama.utils import (
    TrainingConfig,
    PEFTConfig,
    image_spec,
    lora_to_gguf_image,
    ollama_image,
)


@pytest.mark.unit
def test_training_config_defaults():
    """Test the TrainingConfig dataclass has the expected default values."""
    config = TrainingConfig()
    
    # Check basic attributes
    assert config.model == "microsoft/Phi-3-mini-4k-instruct"
    assert config.bf16 is True
    assert config.learning_rate == 5.0e-06
    assert config.num_train_epochs == 1
    assert config.per_device_train_batch_size == 4
    assert config.model_max_length == 1024
    
    # Check gradient checkpointing
    assert config.gradient_checkpointing is True
    assert config.gradient_checkpointing_kwargs == {"use_reentrant": False}
    
    # Verify it's a proper dataclass
    assert is_dataclass(config)


@pytest.mark.unit
def test_training_config_json_serialization():
    """Test the TrainingConfig can be serialized to and from JSON."""
    # Create a config with some custom values
    config = TrainingConfig(
        model="custom/model",
        learning_rate=1.0e-05,
        num_train_epochs=3,
        per_device_train_batch_size=8,
    )
    
    # Convert to JSON
    json_str = config.to_json()
    
    # Parse the JSON to verify its structure
    parsed = json.loads(json_str)
    assert parsed["model"] == "custom/model"
    assert parsed["learning_rate"] == 1.0e-05
    assert parsed["num_train_epochs"] == 3
    
    # Convert back from JSON
    config2 = TrainingConfig.from_json(json_str)
    assert config2.model == "custom/model"
    assert config2.learning_rate == 1.0e-05
    assert config2.num_train_epochs == 3


@pytest.mark.unit
def test_peft_config_defaults():
    """Test the PEFTConfig dataclass has the expected default values."""
    config = PEFTConfig()
    
    # Check attributes
    assert config.r == 16
    assert config.lora_alpha == 32
    assert config.lora_dropout == 0.05
    assert config.bias == "none"
    assert config.task_type == "CAUSAL_LM"
    assert config.target_modules == "all-linear"
    assert config.modules_to_save is None
    
    # Verify it's a proper dataclass
    assert is_dataclass(config)


@pytest.mark.unit
def test_image_spec_configuration():
    """Test the ImageSpec objects are configured properly."""
    # Check image_spec
    assert image_spec.name == "phi3-finetune"
    assert "build-essential" in image_spec.apt_packages
    assert image_spec.requirements == "requirements.txt"
    
    # Check lora_to_gguf_image
    assert lora_to_gguf_image.name == "gguf-ollama"
    assert "git" in lora_to_gguf_image.apt_packages
    assert "huggingface_hub" in lora_to_gguf_image.packages
    assert len(lora_to_gguf_image._commands) == 2
    assert "git clone" in lora_to_gguf_image._commands[0]
    assert "pip install" in lora_to_gguf_image._commands[1]
    
    # Check ollama_image
    assert ollama_image.name == "phi3-ollama-serve"
    assert any(pkg.startswith("flytekitplugins-inference") for pkg in ollama_image.packages)