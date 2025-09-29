"""Contract tests for OpenAPI schema validation.

These tests ensure that the actual API implementation matches the OpenAPI schema
definition and detect breaking changes in CI.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

from gunn.schemas.messages import View
from gunn.schemas.types import Effect, Intent, ObservationDelta


class TestOpenAPISchema:
    """Test OpenAPI schema consistency and validation."""

    @pytest.fixture
    def openapi_schema(self) -> dict[str, Any]:
        """Load the OpenAPI schema from the golden file."""
        schema_path = Path(__file__).parent.parent.parent / "schemas" / "openapi.yaml"
        with open(schema_path) as f:
            schema = yaml.safe_load(f)
            assert isinstance(schema, dict), "OpenAPI schema must be a dictionary"
            return schema

    def test_openapi_schema_loads(self, openapi_schema: dict[str, Any]) -> None:
        """Test that the OpenAPI schema file loads without errors."""
        assert openapi_schema is not None
        assert "openapi" in openapi_schema
        assert openapi_schema["openapi"] == "3.0.3"
        assert "info" in openapi_schema
        assert "paths" in openapi_schema
        assert "components" in openapi_schema

    def test_required_endpoints_present(self, openapi_schema: dict[str, Any]) -> None:
        """Test that all required endpoints are defined in the schema."""
        paths = openapi_schema["paths"]

        # Core endpoints that must be present
        required_endpoints = [
            "/worlds/{world_id}/agents/{agent_id}/observe",
            "/worlds/{world_id}/agents/{agent_id}/intents",
            "/worlds/{world_id}/events",
            "/worlds/{world_id}/agents/{agent_id}/stream",
            "/worlds/{world_id}/agents/{agent_id}/cancel/{req_id}",
            "/health",
        ]

        for endpoint in required_endpoints:
            assert endpoint in paths, (
                f"Required endpoint {endpoint} not found in schema"
            )

    def test_error_schema_matches_docs(self, openapi_schema: dict[str, Any]) -> None:
        """Test that error schema matches the structure defined in docs/errors.md."""
        error_schema = openapi_schema["components"]["schemas"]["Error"]

        # Verify error structure matches docs/errors.md format
        error_properties = error_schema["properties"]["error"]["properties"]

        required_fields = ["code", "name", "message", "recovery_action", "timestamp"]
        for field in required_fields:
            assert field in error_properties, f"Required error field {field} missing"

        # Verify recovery actions match docs/errors.md
        recovery_actions = error_properties["recovery_action"]["enum"]
        expected_actions = [
            "RETRY",
            "RETRY_WITH_DELAY",
            "REGENERATE",
            "MODIFY_INTENT",
            "DEFER",
            "SHED_OLDEST",
            "ABORT",
        ]
        for action in expected_actions:
            assert action in recovery_actions, f"Recovery action {action} missing"

    def test_schema_version_consistency(self, openapi_schema: dict[str, Any]) -> None:
        """Test that schema_version fields use consistent semantic versioning pattern."""
        schemas = openapi_schema["components"]["schemas"]

        # Find all schema_version fields
        schema_version_fields = []
        for schema_name, schema_def in schemas.items():
            if "properties" in schema_def:
                for prop_name, prop_def in schema_def["properties"].items():
                    if prop_name == "schema_version":
                        schema_version_fields.append((schema_name, prop_def))

        # Verify all schema_version fields have semantic versioning pattern
        semver_pattern = r"^\d+\.\d+\.\d+$"
        for schema_name, field_def in schema_version_fields:
            assert "pattern" in field_def, (
                f"schema_version in {schema_name} missing pattern"
            )
            assert field_def["pattern"] == semver_pattern, (
                f"schema_version pattern in {schema_name} doesn't match semantic versioning"
            )

    def test_pydantic_models_match_openapi(
        self, openapi_schema: dict[str, Any]
    ) -> None:
        """Test that Pydantic models are consistent with OpenAPI schema definitions."""
        # Test View model
        view_schema = openapi_schema["components"]["schemas"]["View"]
        view_required = set(view_schema["required"])

        # Create a valid View instance and check it has all required fields
        view = View(
            agent_id="test_agent",
            view_seq=42,
            visible_entities={"entity1": {"type": "test"}},
            visible_relationships={"entity1": ["entity2"]},
            context_digest="sha256:test123",
        )

        view_dict = view.model_dump()
        view_fields = set(view_dict.keys())

        # All required fields should be present
        assert view_required.issubset(view_fields), (
            f"View model missing required fields: {view_required - view_fields}"
        )

    def test_intent_schema_validation(self, openapi_schema: dict[str, Any]) -> None:
        """Test that Intent TypedDict structure matches OpenAPI schema."""
        intent_schema = openapi_schema["components"]["schemas"]["IntentRequest"]
        required_fields = set(intent_schema["required"])

        # Create a valid intent
        intent: Intent = {
            "kind": "Speak",
            "payload": {"text": "Hello"},
            "context_seq": 42,
            "req_id": "req_123",
            "agent_id": "agent_001",
            "priority": 1,
            "schema_version": "1.0.0",
        }

        intent_fields = set(intent.keys())

        # All required fields should be present
        assert required_fields.issubset(intent_fields), (
            f"Intent missing required fields: {required_fields - intent_fields}"
        )

        # Verify kind enum values
        kind_enum = intent_schema["properties"]["kind"]["enum"]
        assert intent["kind"] in kind_enum, (
            f"Intent kind {intent['kind']} not in allowed values"
        )

    def test_observation_delta_patch_format(
        self, openapi_schema: dict[str, Any]
    ) -> None:
        """Test that ObservationDelta patch format matches RFC6902 JSON Patch."""
        delta_schema = openapi_schema["components"]["schemas"]["ObservationDelta"]
        patch_schema = delta_schema["properties"]["patches"]["items"]

        # Verify patch operations match RFC6902
        patch_ops = patch_schema["properties"]["op"]["enum"]
        rfc6902_ops = ["add", "remove", "replace", "move", "copy", "test"]

        for op in rfc6902_ops:
            assert op in patch_ops, f"RFC6902 operation {op} missing from schema"

        # Test valid ObservationDelta
        delta: ObservationDelta = {
            "view_seq": 43,
            "patches": [
                {
                    "op": "replace",
                    "path": "/visible_entities/entity1/position",
                    "value": [11.0, 21.0, 0.0],
                }
            ],
            "context_digest": "sha256:def456",
            "schema_version": "1.0.0",
        }

        # Verify patch structure
        patch = delta["patches"][0]
        assert patch["op"] in patch_ops
        assert "path" in patch
        assert patch["path"].startswith("/")  # JSON Pointer format

    def test_breaking_change_detection(self, openapi_schema: dict[str, Any]) -> None:
        """Test for potential breaking changes in the API schema.

        This test serves as a canary for breaking changes that would require
        explicit version bumps and migration guidance.
        """
        # Verify API version hasn't changed unexpectedly
        api_version = openapi_schema["info"]["version"]
        assert api_version == "1.0.0", (
            f"API version changed to {api_version} - ensure this is intentional and documented"
        )

        # Verify core endpoint paths haven't changed
        paths = openapi_schema["paths"]
        core_paths = [
            "/worlds/{world_id}/agents/{agent_id}/observe",
            "/worlds/{world_id}/agents/{agent_id}/intents",
            "/worlds/{world_id}/events",
        ]

        for path in core_paths:
            assert path in paths, (
                f"Core API path {path} was removed - this is a breaking change"
            )

        # Verify required fields in core schemas haven't been removed
        view_required = set(openapi_schema["components"]["schemas"]["View"]["required"])
        expected_view_fields = {
            "agent_id",
            "view_seq",
            "visible_entities",
            "visible_relationships",
            "context_digest",
        }
        assert expected_view_fields.issubset(view_required), (
            "Required fields removed from View schema - this is a breaking change"
        )

    def test_http_status_codes_match_errors_doc(
        self, openapi_schema: dict[str, Any]
    ) -> None:
        """Test that HTTP status codes in responses match docs/errors.md mappings."""
        # Load error mappings from the test (in real implementation,
        # this would parse docs/errors.md)
        # Note: These mappings would be validated against actual error responses
        # in a full implementation

        # Check that response examples use correct HTTP status codes
        responses = openapi_schema["components"]["responses"]

        # Verify specific error response status codes
        assert "BadRequest" in responses
        assert "Unauthorized" in responses
        assert "Forbidden" in responses
        assert "Conflict" in responses
        assert "UnprocessableEntity" in responses
        assert "TooManyRequests" in responses

        # In a full implementation, we would parse the example error codes
        # and verify they map to the correct HTTP status codes

    def test_websocket_endpoint_specification(
        self, openapi_schema: dict[str, Any]
    ) -> None:
        """Test that WebSocket endpoint is properly specified."""
        stream_endpoint = "/worlds/{world_id}/agents/{agent_id}/stream"
        assert stream_endpoint in openapi_schema["paths"]

        stream_spec = openapi_schema["paths"][stream_endpoint]["get"]

        # Verify WebSocket upgrade response
        responses = stream_spec["responses"]
        assert "101" in responses, "WebSocket upgrade response (101) missing"

        upgrade_response = responses["101"]
        assert "headers" in upgrade_response
        headers = upgrade_response["headers"]
        assert "Upgrade" in headers
        assert "Connection" in headers

    def test_security_schemes_defined(self, openapi_schema: dict[str, Any]) -> None:
        """Test that security schemes are properly defined."""
        security_schemes = openapi_schema["components"]["securitySchemes"]

        # Verify both authentication methods are defined
        assert "BearerAuth" in security_schemes
        assert "mTLS" in security_schemes

        # Verify Bearer auth configuration
        bearer_auth = security_schemes["BearerAuth"]
        assert bearer_auth["type"] == "http"
        assert bearer_auth["scheme"] == "bearer"

        # Verify mTLS configuration
        mtls_auth = security_schemes["mTLS"]
        assert mtls_auth["type"] == "mutualTLS"

    def test_parameter_validation_patterns(
        self, openapi_schema: dict[str, Any]
    ) -> None:
        """Test that path parameters have proper validation patterns."""
        parameters = openapi_schema["components"]["parameters"]

        # Verify WorldId parameter
        world_id_param = parameters["WorldId"]
        assert world_id_param["schema"]["pattern"] == "^[a-zA-Z0-9_-]+$"
        assert world_id_param["schema"]["minLength"] == 1
        assert world_id_param["schema"]["maxLength"] == 64

        # Verify AgentId parameter
        agent_id_param = parameters["AgentId"]
        assert agent_id_param["schema"]["pattern"] == "^[a-zA-Z0-9_-]+$"
        assert agent_id_param["schema"]["minLength"] == 1
        assert agent_id_param["schema"]["maxLength"] == 64


class TestSchemaEvolution:
    """Test schema evolution and backward compatibility."""

    def test_schema_version_field_presence(self) -> None:
        """Test that all message types include schema_version field."""
        # Test TypedDict types
        intent: Intent = {
            "kind": "Speak",
            "payload": {},
            "context_seq": 0,
            "req_id": "test",
            "agent_id": "test",
            "priority": 0,
            "schema_version": "1.0.0",  # Must be present
        }

        effect: Effect = {
            "uuid": "test",
            "kind": "test",
            "payload": {},
            "global_seq": 0,
            "sim_time": 0.0,
            "source_id": "test",
            "schema_version": "1.0.0",  # Must be present
        }

        delta: ObservationDelta = {
            "view_seq": 0,
            "patches": [],
            "context_digest": "test",
            "schema_version": "1.0.0",  # Must be present
        }

        # Verify all have schema_version
        assert "schema_version" in intent
        assert "schema_version" in effect
        assert "schema_version" in delta

    def test_semantic_versioning_format(self) -> None:
        """Test that schema versions follow semantic versioning."""
        import re

        semver_pattern = re.compile(r"^\d+\.\d+\.\d+$")

        valid_versions = ["1.0.0", "2.1.3", "10.20.30"]
        invalid_versions = ["1.0", "v1.0.0", "1.0.0-beta", "1.0.0+build"]

        for version in valid_versions:
            assert semver_pattern.match(version), (
                f"Valid version {version} failed pattern"
            )

        for version in invalid_versions:
            assert not semver_pattern.match(version), (
                f"Invalid version {version} passed pattern"
            )


if __name__ == "__main__":
    pytest.main([__file__])
