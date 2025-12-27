# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Unit tests for sageLLM accel types."""

import pytest

from sage.common.components.sage_llm.sageLLM.accel.types import (
    AccelConfig,
    QuantizationMethod,
    QuantizationProfile,
    SparsityConfig,
    SparsityPattern,
    SpeculativeConfig,
    SpeculativeMethod,
)


class TestQuantizationProfile:
    """Tests for QuantizationProfile dataclass."""

    def test_default(self) -> None:
        """Test default profile (no quantization)."""
        profile = QuantizationProfile()
        assert profile.method == QuantizationMethod.NONE
        assert profile.bits == 16

    def test_preset_fp16(self) -> None:
        """Test FP16 preset."""
        profile = QuantizationProfile.preset("fp16")
        assert profile.method == QuantizationMethod.NONE
        assert profile.bits == 16

    def test_preset_awq(self) -> None:
        """Test AWQ preset."""
        profile = QuantizationProfile.preset("awq")
        assert profile.method == QuantizationMethod.AWQ
        assert profile.bits == 4
        assert profile.group_size == 128

    def test_preset_gptq(self) -> None:
        """Test GPTQ preset."""
        profile = QuantizationProfile.preset("gptq")
        assert profile.method == QuantizationMethod.GPTQ
        assert profile.bits == 4

    def test_preset_unknown(self) -> None:
        """Test unknown preset raises error."""
        with pytest.raises(ValueError, match="Unknown preset"):
            QuantizationProfile.preset("unknown")

    def test_to_dict(self) -> None:
        """Test serialization."""
        profile = QuantizationProfile(
            method=QuantizationMethod.FP8,
            bits=8,
        )
        data = profile.to_dict()
        assert data["method"] == "fp8"
        assert data["bits"] == 8

    def test_from_dict(self) -> None:
        """Test deserialization."""
        data = {
            "method": "awq",
            "bits": 4,
            "group_size": 64,
        }
        profile = QuantizationProfile.from_dict(data)
        assert profile.method == QuantizationMethod.AWQ
        assert profile.group_size == 64


class TestSparsityConfig:
    """Tests for SparsityConfig dataclass."""

    def test_default(self) -> None:
        """Test default config (no sparsity)."""
        config = SparsityConfig()
        assert config.pattern == SparsityPattern.NONE
        assert config.ratio == 0.0

    def test_structured_sparsity(self) -> None:
        """Test 2:4 structured sparsity."""
        config = SparsityConfig(
            pattern=SparsityPattern.STRUCTURED_2_4,
            ratio=0.5,
        )
        assert config.pattern == SparsityPattern.STRUCTURED_2_4


class TestSpeculativeConfig:
    """Tests for SpeculativeConfig dataclass."""

    def test_default(self) -> None:
        """Test default config (no speculative)."""
        config = SpeculativeConfig()
        assert config.method == SpeculativeMethod.NONE

    def test_draft_model(self) -> None:
        """Test draft model config."""
        config = SpeculativeConfig(
            method=SpeculativeMethod.DRAFT_MODEL,
            draft_model_id="Qwen/Qwen2.5-0.5B-Instruct",
            num_speculative_tokens=5,
        )
        assert config.method == SpeculativeMethod.DRAFT_MODEL
        assert config.draft_model_id == "Qwen/Qwen2.5-0.5B-Instruct"

    def test_to_dict(self) -> None:
        """Test serialization."""
        config = SpeculativeConfig(
            method=SpeculativeMethod.MEDUSA,
            num_speculative_tokens=8,
        )
        data = config.to_dict()
        assert data["method"] == "medusa"


class TestAccelConfig:
    """Tests for AccelConfig dataclass."""

    def test_default(self) -> None:
        """Test default config."""
        config = AccelConfig.default()
        assert config.quantization.method == QuantizationMethod.NONE
        assert config.enable_flash_attention is True

    def test_for_inference(self) -> None:
        """Test inference-optimized config."""
        config = AccelConfig.for_inference(quantization="fp8")
        assert config.quantization.method == QuantizationMethod.FP8
        assert config.enable_cuda_graphs is True

    def test_to_dict(self) -> None:
        """Test serialization."""
        config = AccelConfig(
            enable_flash_attention=True,
            enable_cuda_graphs=True,
        )
        data = config.to_dict()
        assert data["enable_flash_attention"] is True
        assert "quantization" in data
        assert "sparsity" in data
