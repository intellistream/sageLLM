# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SAGE project

"""Tests for RequestClassifier module."""

import sys
from pathlib import Path

import pytest

# Add parent to path
root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))

from control_plane.request_classifier import (  # noqa: E402  # type: ignore[import-not-found]
    RequestClassifier,
    ValidationErrorCode,
    ValidationResult,
    create_classifier,
)
from control_plane.types import (  # noqa: E402  # type: ignore[import-not-found]
    ExecutionInstance,
    ExecutionInstanceType,
    RequestMetadata,
    RequestPriority,
    RequestType,
)


class TestRequestClassifier:
    """Tests for RequestClassifier class."""

    @pytest.fixture
    def classifier(self) -> RequestClassifier:
        """Create a RequestClassifier instance for testing."""
        return RequestClassifier()

    @pytest.fixture
    def strict_classifier(self) -> RequestClassifier:
        """Create a strict RequestClassifier instance for testing."""
        return RequestClassifier(strict_validation=True)


class TestClassify(TestRequestClassifier):
    """Tests for classify() method."""

    def test_classify_explicit_embedding_type(self, classifier: RequestClassifier):
        """Test classification when request_type is explicitly EMBEDDING."""
        request = RequestMetadata(
            request_id="req-1",
            request_type=RequestType.EMBEDDING,
            embedding_texts=["Hello"],
        )
        assert classifier.classify(request) == RequestType.EMBEDDING

    def test_classify_explicit_llm_generate_type(self, classifier: RequestClassifier):
        """Test classification when request_type is explicitly LLM_GENERATE."""
        request = RequestMetadata(
            request_id="req-1",
            prompt="Generate something",
            request_type=RequestType.LLM_GENERATE,
        )
        assert classifier.classify(request) == RequestType.LLM_GENERATE

    def test_classify_explicit_llm_chat_type(self, classifier: RequestClassifier):
        """Test classification when request_type is explicitly LLM_CHAT."""
        request = RequestMetadata(
            request_id="req-1",
            prompt="Hello",
            request_type=RequestType.LLM_CHAT,
        )
        assert classifier.classify(request) == RequestType.LLM_CHAT

    def test_classify_auto_detect_embedding_from_texts(
        self, classifier: RequestClassifier
    ):
        """Test auto-detection of EMBEDDING when embedding_texts is set."""
        request = RequestMetadata(
            request_id="req-1",
            embedding_texts=["Hello", "World"],
            # request_type defaults to LLM_CHAT
        )
        assert classifier.classify(request) == RequestType.EMBEDDING

    def test_classify_auto_detect_embedding_single_text(
        self, classifier: RequestClassifier
    ):
        """Test auto-detection of EMBEDDING with single text."""
        request = RequestMetadata(
            request_id="req-1",
            embedding_texts=["Single text"],
        )
        assert classifier.classify(request) == RequestType.EMBEDDING

    def test_classify_empty_embedding_texts_defaults_to_llm_chat(
        self, classifier: RequestClassifier
    ):
        """Test that empty embedding_texts list defaults to LLM_CHAT."""
        request = RequestMetadata(
            request_id="req-1",
            embedding_texts=[],
            prompt="Hello",
        )
        assert classifier.classify(request) == RequestType.LLM_CHAT

    def test_classify_none_embedding_texts_defaults_to_llm_chat(
        self, classifier: RequestClassifier
    ):
        """Test that None embedding_texts defaults to LLM_CHAT."""
        request = RequestMetadata(
            request_id="req-1",
            embedding_texts=None,
            prompt="Hello",
        )
        assert classifier.classify(request) == RequestType.LLM_CHAT

    def test_classify_prompt_only_defaults_to_llm_chat(
        self, classifier: RequestClassifier
    ):
        """Test that request with only prompt defaults to LLM_CHAT."""
        request = RequestMetadata(
            request_id="req-1",
            prompt="Hello, how are you?",
        )
        assert classifier.classify(request) == RequestType.LLM_CHAT

    def test_classify_empty_request_defaults_to_llm_chat(
        self, classifier: RequestClassifier
    ):
        """Test that completely empty request defaults to LLM_CHAT."""
        request = RequestMetadata(request_id="req-1")
        assert classifier.classify(request) == RequestType.LLM_CHAT

    def test_classify_embedding_with_explicit_type_overrides_prompt(
        self, classifier: RequestClassifier
    ):
        """Test that explicit EMBEDDING type overrides presence of prompt."""
        request = RequestMetadata(
            request_id="req-1",
            prompt="This prompt should be ignored",
            request_type=RequestType.EMBEDDING,
            embedding_texts=["Use this instead"],
        )
        assert classifier.classify(request) == RequestType.EMBEDDING

    def test_classify_embedding_texts_overrides_default_llm_chat(
        self, classifier: RequestClassifier
    ):
        """Test that embedding_texts overrides default LLM_CHAT even with prompt."""
        request = RequestMetadata(
            request_id="req-1",
            prompt="Some prompt",
            embedding_texts=["Text to embed"],
            # request_type defaults to LLM_CHAT, but embedding_texts takes precedence
        )
        assert classifier.classify(request) == RequestType.EMBEDDING


class TestGetCompatibleInstances(TestRequestClassifier):
    """Tests for get_compatible_instances() method."""

    @pytest.fixture
    def sample_instances(self) -> list[ExecutionInstance]:
        """Create sample instances for testing."""
        return [
            ExecutionInstance(
                instance_id="llm-general-1",
                host="localhost",
                port=8000,
                model_name="Qwen/Qwen2.5-7B-Instruct",
                instance_type=ExecutionInstanceType.GENERAL,
                is_available=True,
                is_healthy=True,
                current_load=0.3,
            ),
            ExecutionInstance(
                instance_id="llm-prefill-1",
                host="localhost",
                port=8001,
                model_name="Qwen/Qwen2.5-7B-Instruct",
                instance_type=ExecutionInstanceType.PREFILLING,
                is_available=True,
                is_healthy=True,
                current_load=0.5,
            ),
            ExecutionInstance(
                instance_id="llm-decode-1",
                host="localhost",
                port=8002,
                model_name="Qwen/Qwen2.5-7B-Instruct",
                instance_type=ExecutionInstanceType.DECODING,
                is_available=True,
                is_healthy=True,
                current_load=0.2,
            ),
            ExecutionInstance(
                instance_id="embed-1",
                host="localhost",
                port=8090,
                model_name="BAAI/bge-m3",
                instance_type=ExecutionInstanceType.EMBEDDING,
                is_available=True,
                is_healthy=True,
                current_load=0.1,
            ),
            ExecutionInstance(
                instance_id="mixed-1",
                host="localhost",
                port=8003,
                model_name="Qwen/Qwen2.5-7B-Instruct",
                instance_type=ExecutionInstanceType.LLM_EMBEDDING,
                embedding_model_loaded="BAAI/bge-m3",
                is_available=True,
                is_healthy=True,
                current_load=0.4,
            ),
            ExecutionInstance(
                instance_id="unavailable-1",
                host="localhost",
                port=8004,
                model_name="Qwen/Qwen2.5-7B-Instruct",
                instance_type=ExecutionInstanceType.GENERAL,
                is_available=False,
                is_healthy=True,
                current_load=0.0,
            ),
            ExecutionInstance(
                instance_id="unhealthy-1",
                host="localhost",
                port=8005,
                model_name="Qwen/Qwen2.5-7B-Instruct",
                instance_type=ExecutionInstanceType.GENERAL,
                is_available=True,
                is_healthy=False,
                current_load=0.0,
            ),
        ]

    def test_get_compatible_instances_for_embedding(
        self, classifier: RequestClassifier, sample_instances: list[ExecutionInstance]
    ):
        """Test getting compatible instances for EMBEDDING request."""
        compatible = classifier.get_compatible_instances(
            RequestType.EMBEDDING, sample_instances
        )
        instance_ids = [i.instance_id for i in compatible]

        # Should include EMBEDDING and LLM_EMBEDDING instances
        assert "embed-1" in instance_ids
        assert "mixed-1" in instance_ids
        # Should NOT include LLM-only instances
        assert "llm-general-1" not in instance_ids
        assert "llm-prefill-1" not in instance_ids
        assert "llm-decode-1" not in instance_ids
        # Should NOT include unavailable/unhealthy instances
        assert "unavailable-1" not in instance_ids
        assert "unhealthy-1" not in instance_ids

    def test_get_compatible_instances_for_llm_chat(
        self, classifier: RequestClassifier, sample_instances: list[ExecutionInstance]
    ):
        """Test getting compatible instances for LLM_CHAT request."""
        compatible = classifier.get_compatible_instances(
            RequestType.LLM_CHAT, sample_instances
        )
        instance_ids = [i.instance_id for i in compatible]

        # Should include LLM instances and LLM_EMBEDDING
        assert "llm-general-1" in instance_ids
        assert "llm-prefill-1" in instance_ids
        assert "llm-decode-1" in instance_ids
        assert "mixed-1" in instance_ids
        # Should NOT include pure EMBEDDING instances
        assert "embed-1" not in instance_ids
        # Should NOT include unavailable/unhealthy instances
        assert "unavailable-1" not in instance_ids
        assert "unhealthy-1" not in instance_ids

    def test_get_compatible_instances_for_llm_generate(
        self, classifier: RequestClassifier, sample_instances: list[ExecutionInstance]
    ):
        """Test getting compatible instances for LLM_GENERATE request."""
        compatible = classifier.get_compatible_instances(
            RequestType.LLM_GENERATE, sample_instances
        )
        instance_ids = [i.instance_id for i in compatible]

        # Should include same instances as LLM_CHAT
        assert "llm-general-1" in instance_ids
        assert "llm-prefill-1" in instance_ids
        assert "llm-decode-1" in instance_ids
        assert "mixed-1" in instance_ids
        assert "embed-1" not in instance_ids

    def test_get_compatible_instances_include_unavailable(
        self, classifier: RequestClassifier, sample_instances: list[ExecutionInstance]
    ):
        """Test including unavailable instances when flag is set."""
        compatible = classifier.get_compatible_instances(
            RequestType.LLM_CHAT,
            sample_instances,
            include_unavailable=True,
        )
        instance_ids = [i.instance_id for i in compatible]

        # Should now include unavailable and unhealthy instances
        assert "unavailable-1" in instance_ids
        assert "unhealthy-1" in instance_ids

    def test_get_compatible_instances_prefer_specialized_embedding(
        self, classifier: RequestClassifier, sample_instances: list[ExecutionInstance]
    ):
        """Test that specialized EMBEDDING instances are preferred."""
        compatible = classifier.get_compatible_instances(
            RequestType.EMBEDDING,
            sample_instances,
            prefer_specialized=True,
        )

        # First instance should be pure EMBEDDING
        assert compatible[0].instance_type == ExecutionInstanceType.EMBEDDING
        # Second should be LLM_EMBEDDING
        if len(compatible) > 1:
            assert compatible[1].instance_type == ExecutionInstanceType.LLM_EMBEDDING

    def test_get_compatible_instances_prefer_specialized_llm(
        self, classifier: RequestClassifier, sample_instances: list[ExecutionInstance]
    ):
        """Test that specialized LLM instances are preferred over LLM_EMBEDDING."""
        compatible = classifier.get_compatible_instances(
            RequestType.LLM_CHAT,
            sample_instances,
            prefer_specialized=True,
        )

        # LLM_EMBEDDING should come after pure LLM instances
        mixed_idx = next(
            (
                i
                for i, inst in enumerate(compatible)
                if inst.instance_type == ExecutionInstanceType.LLM_EMBEDDING
            ),
            len(compatible),
        )
        general_idx = next(
            (
                i
                for i, inst in enumerate(compatible)
                if inst.instance_type == ExecutionInstanceType.GENERAL
            ),
            len(compatible),
        )

        # GENERAL should come before LLM_EMBEDDING
        assert general_idx < mixed_idx

    def test_get_compatible_instances_no_preference_sorting(
        self, classifier: RequestClassifier, sample_instances: list[ExecutionInstance]
    ):
        """Test disabling preference sorting."""
        compatible = classifier.get_compatible_instances(
            RequestType.EMBEDDING,
            sample_instances,
            prefer_specialized=False,
        )

        # Should still return correct instances, just not sorted by preference
        instance_ids = {i.instance_id for i in compatible}
        assert "embed-1" in instance_ids
        assert "mixed-1" in instance_ids

    def test_get_compatible_instances_empty_list(
        self, classifier: RequestClassifier
    ):
        """Test with empty instance list."""
        compatible = classifier.get_compatible_instances(
            RequestType.EMBEDDING, []
        )
        assert compatible == []

    def test_get_compatible_instances_no_compatible(
        self, classifier: RequestClassifier
    ):
        """Test when no instances are compatible."""
        # Only LLM instances
        instances = [
            ExecutionInstance(
                instance_id="llm-1",
                host="localhost",
                port=8000,
                model_name="llama",
                instance_type=ExecutionInstanceType.GENERAL,
                is_available=True,
                is_healthy=True,
            ),
        ]

        # Request EMBEDDING
        compatible = classifier.get_compatible_instances(
            RequestType.EMBEDDING, instances
        )
        assert compatible == []


class TestValidateRequest(TestRequestClassifier):
    """Tests for validate_request() method."""

    def test_validate_valid_embedding_request(self, classifier: RequestClassifier):
        """Test validation of a valid embedding request."""
        request = RequestMetadata(
            request_id="req-1",
            request_type=RequestType.EMBEDDING,
            embedding_texts=["Hello", "World"],
        )
        result = classifier.validate_request(request)

        assert result.is_valid is True
        assert result.error_code is None
        assert result.error_message == ""

    def test_validate_valid_llm_chat_request(self, classifier: RequestClassifier):
        """Test validation of a valid LLM chat request."""
        request = RequestMetadata(
            request_id="req-1",
            prompt="Hello, how are you?",
            request_type=RequestType.LLM_CHAT,
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
        )
        result = classifier.validate_request(request)

        assert result.is_valid is True
        assert result.error_code is None

    def test_validate_missing_request_id(self, classifier: RequestClassifier):
        """Test validation fails with empty request_id."""
        request = RequestMetadata(
            request_id="",
            prompt="Hello",
        )
        result = classifier.validate_request(request)

        assert result.is_valid is False
        assert result.error_code == ValidationErrorCode.MISSING_REQUEST_ID

    def test_validate_whitespace_request_id(self, classifier: RequestClassifier):
        """Test validation fails with whitespace-only request_id."""
        request = RequestMetadata(
            request_id="   ",
            prompt="Hello",
        )
        result = classifier.validate_request(request)

        assert result.is_valid is False
        assert result.error_code == ValidationErrorCode.MISSING_REQUEST_ID

    def test_validate_embedding_missing_texts(self, classifier: RequestClassifier):
        """Test validation fails when embedding request has no texts."""
        request = RequestMetadata(
            request_id="req-1",
            request_type=RequestType.EMBEDDING,
            embedding_texts=None,
        )
        result = classifier.validate_request(request)

        assert result.is_valid is False
        assert result.error_code == ValidationErrorCode.MISSING_EMBEDDING_TEXTS

    def test_validate_embedding_empty_texts(self, classifier: RequestClassifier):
        """Test validation fails when embedding request has empty texts list."""
        request = RequestMetadata(
            request_id="req-1",
            request_type=RequestType.EMBEDDING,
            embedding_texts=[],
        )
        result = classifier.validate_request(request)

        assert result.is_valid is False
        assert result.error_code == ValidationErrorCode.EMPTY_EMBEDDING_TEXTS

    def test_validate_embedding_invalid_batch_size(self, classifier: RequestClassifier):
        """Test validation fails with invalid embedding batch size."""
        request = RequestMetadata(
            request_id="req-1",
            request_type=RequestType.EMBEDDING,
            embedding_texts=["Hello"],
            embedding_batch_size=0,
        )
        result = classifier.validate_request(request)

        assert result.is_valid is False
        assert result.error_code == ValidationErrorCode.INVALID_EMBEDDING_BATCH_SIZE

    def test_validate_llm_invalid_max_tokens(self, classifier: RequestClassifier):
        """Test validation fails with invalid max_tokens."""
        request = RequestMetadata(
            request_id="req-1",
            prompt="Hello",
            max_tokens=-10,
        )
        result = classifier.validate_request(request)

        assert result.is_valid is False
        assert result.error_code == ValidationErrorCode.INVALID_MAX_TOKENS

    def test_validate_llm_zero_max_tokens(self, classifier: RequestClassifier):
        """Test validation fails with zero max_tokens."""
        request = RequestMetadata(
            request_id="req-1",
            prompt="Hello",
            max_tokens=0,
        )
        result = classifier.validate_request(request)

        assert result.is_valid is False
        assert result.error_code == ValidationErrorCode.INVALID_MAX_TOKENS

    def test_validate_llm_none_max_tokens_ok(self, classifier: RequestClassifier):
        """Test validation passes with None max_tokens (no limit)."""
        request = RequestMetadata(
            request_id="req-1",
            prompt="Hello",
            max_tokens=None,
        )
        result = classifier.validate_request(request)

        assert result.is_valid is True

    def test_validate_llm_invalid_temperature(self, classifier: RequestClassifier):
        """Test validation fails with negative temperature."""
        request = RequestMetadata(
            request_id="req-1",
            prompt="Hello",
            temperature=-0.5,
        )
        result = classifier.validate_request(request)

        assert result.is_valid is False
        assert result.error_code == ValidationErrorCode.INVALID_TEMPERATURE

    def test_validate_llm_invalid_top_p_negative(self, classifier: RequestClassifier):
        """Test validation fails with negative top_p."""
        request = RequestMetadata(
            request_id="req-1",
            prompt="Hello",
            top_p=-0.1,
        )
        result = classifier.validate_request(request)

        assert result.is_valid is False
        assert result.error_code == ValidationErrorCode.INVALID_TOP_P

    def test_validate_llm_invalid_top_p_above_one(self, classifier: RequestClassifier):
        """Test validation fails with top_p > 1.0."""
        request = RequestMetadata(
            request_id="req-1",
            prompt="Hello",
            top_p=1.5,
        )
        result = classifier.validate_request(request)

        assert result.is_valid is False
        assert result.error_code == ValidationErrorCode.INVALID_TOP_P

    def test_validate_llm_no_prompt_warning(self, classifier: RequestClassifier):
        """Test validation passes but warns when LLM request has no prompt."""
        request = RequestMetadata(
            request_id="req-1",
            request_type=RequestType.LLM_CHAT,
            prompt=None,
        )
        result = classifier.validate_request(request)

        assert result.is_valid is True
        assert result.warnings is not None
        assert len(result.warnings) > 0
        assert any("prompt" in w.lower() for w in result.warnings)

    def test_validate_strict_conflicting_type(
        self, strict_classifier: RequestClassifier
    ):
        """Test strict validation fails with both prompt and embedding_texts."""
        request = RequestMetadata(
            request_id="req-1",
            prompt="Hello",
            embedding_texts=["World"],
        )
        result = strict_classifier.validate_request(request)

        assert result.is_valid is False
        assert result.error_code == ValidationErrorCode.CONFLICTING_REQUEST_TYPE

    def test_validate_non_strict_allows_conflicting_type(
        self, classifier: RequestClassifier
    ):
        """Test non-strict validation allows both prompt and embedding_texts."""
        request = RequestMetadata(
            request_id="req-1",
            prompt="Hello",
            embedding_texts=["World"],
        )
        result = classifier.validate_request(request)

        # Non-strict mode should allow this (classified as EMBEDDING due to texts)
        assert result.is_valid is True


class TestGetRequestSummary(TestRequestClassifier):
    """Tests for get_request_summary() method."""

    def test_get_summary_embedding_request(self, classifier: RequestClassifier):
        """Test summary for embedding request."""
        request = RequestMetadata(
            request_id="req-1",
            request_type=RequestType.EMBEDDING,  # Explicit type for effective_model_name
            embedding_texts=["Hello", "World"],
            embedding_model="BAAI/bge-m3",
            priority=RequestPriority.HIGH,
        )
        summary = classifier.get_request_summary(request)

        assert summary["request_id"] == "req-1"
        assert summary["classified_type"] == "embedding"
        assert summary["has_embedding_texts"] is True
        assert summary["embedding_text_count"] == 2
        assert summary["is_valid"] is True
        assert summary["effective_model"] == "BAAI/bge-m3"
        assert summary["priority"] == 1  # HIGH = 1

    def test_get_summary_llm_request(self, classifier: RequestClassifier):
        """Test summary for LLM request."""
        request = RequestMetadata(
            request_id="req-2",
            prompt="Hello, world",
            model_name="Qwen/Qwen2.5-7B-Instruct",
            request_type=RequestType.LLM_GENERATE,
        )
        summary = classifier.get_request_summary(request)

        assert summary["request_id"] == "req-2"
        assert summary["classified_type"] == "llm_generate"
        assert summary["explicit_type"] == "llm_generate"
        assert summary["has_prompt"] is True
        assert summary["has_embedding_texts"] is False
        assert summary["is_valid"] is True
        assert summary["effective_model"] == "Qwen/Qwen2.5-7B-Instruct"

    def test_get_summary_invalid_request(self, classifier: RequestClassifier):
        """Test summary for invalid request."""
        request = RequestMetadata(
            request_id="",
            prompt="Hello",
        )
        summary = classifier.get_request_summary(request)

        assert summary["is_valid"] is False
        assert summary["validation_error"] is not None
        assert "request" in summary["validation_error"].lower()


class TestCreateClassifier:
    """Tests for create_classifier factory function."""

    def test_create_classifier_default(self):
        """Test creating classifier with default settings."""
        classifier = create_classifier()
        assert isinstance(classifier, RequestClassifier)
        assert classifier.strict_validation is False

    def test_create_classifier_strict(self):
        """Test creating classifier with strict validation."""
        classifier = create_classifier(strict_validation=True)
        assert isinstance(classifier, RequestClassifier)
        assert classifier.strict_validation is True


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_validation_result_valid(self):
        """Test creating a valid result."""
        result = ValidationResult(is_valid=True)
        assert result.is_valid is True
        assert result.error_code is None
        assert result.error_message == ""
        assert result.warnings == []

    def test_validation_result_invalid(self):
        """Test creating an invalid result."""
        result = ValidationResult(
            is_valid=False,
            error_code=ValidationErrorCode.MISSING_REQUEST_ID,
            error_message="Request ID is required",
        )
        assert result.is_valid is False
        assert result.error_code == ValidationErrorCode.MISSING_REQUEST_ID
        assert "Request ID" in result.error_message

    def test_validation_result_with_warnings(self):
        """Test creating result with warnings."""
        result = ValidationResult(
            is_valid=True,
            warnings=["Warning 1", "Warning 2"],
        )
        assert result.is_valid is True
        assert len(result.warnings) == 2
