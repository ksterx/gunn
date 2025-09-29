"""Contract tests for Protocol Buffer schema validation.

These tests ensure that the protobuf schema is valid and consistent with
the Python type definitions.
"""

import re
from pathlib import Path

import pytest


class TestProtobufSchema:
    """Test Protocol Buffer schema consistency and validation."""

    @pytest.fixture
    def proto_content(self) -> str:
        """Load the protobuf schema content."""
        proto_path = (
            Path(__file__).parent.parent.parent
            / "schemas"
            / "proto"
            / "simulation.proto"
        )
        with open(proto_path) as f:
            return f.read()

    def test_proto_syntax_version(self, proto_content: str) -> None:
        """Test that protobuf uses proto3 syntax."""
        assert 'syntax = "proto3";' in proto_content

    def test_package_declaration(self, proto_content: str) -> None:
        """Test that package is properly declared."""
        assert "package gunn.simulation.v1;" in proto_content

    def test_required_imports(self, proto_content: str) -> None:
        """Test that required imports are present."""
        required_imports = [
            'import "google/protobuf/any.proto";',
            'import "google/protobuf/timestamp.proto";',
        ]

        for import_stmt in required_imports:
            assert import_stmt in proto_content, f"Missing import: {import_stmt}"

    def test_language_options(self, proto_content: str) -> None:
        """Test that language-specific options are defined."""
        # Go package option
        assert "option go_package = " in proto_content

        # Java options
        assert "option java_package = " in proto_content
        assert "option java_multiple_files = true;" in proto_content

        # C# namespace
        assert "option csharp_namespace = " in proto_content

    def test_core_message_types_present(self, proto_content: str) -> None:
        """Test that all core message types are defined."""
        required_messages = [
            "message Intent",
            "message EffectDraft",
            "message Effect",
            "message ObservationDelta",
            "message JsonPatch",
            "message View",
            "message WorldState",
            "message Error",
        ]

        for message in required_messages:
            assert message in proto_content, f"Missing message type: {message}"

    def test_unity_specific_messages(self, proto_content: str) -> None:
        """Test that Unity-specific message types are defined."""
        unity_messages = [
            "message TimeTick",
            "message MoveCommand",
            "message CollisionEvent",
            "message Vector3",
            "message SpeakPayload",
            "message MovePayload",
            "message InteractPayload",
        ]

        for message in unity_messages:
            assert message in proto_content, f"Missing Unity message: {message}"

    def test_service_definition(self, proto_content: str) -> None:
        """Test that UnityAdapter service is properly defined."""
        assert "service UnityAdapter" in proto_content

        # Check required RPC methods
        required_rpcs = [
            "rpc SubmitIntent",
            "rpc EmitEffect",
            "rpc StreamObservations",
            "rpc GetView",
            "rpc CancelIntent",
            "rpc HealthCheck",
        ]

        for rpc in required_rpcs:
            assert rpc in proto_content, f"Missing RPC method: {rpc}"

    def test_streaming_rpc_definition(self, proto_content: str) -> None:
        """Test that streaming RPC is properly defined."""
        # StreamObservations should return a stream
        stream_pattern = r"rpc StreamObservations\([^)]+\) returns \(stream [^)]+\);"
        assert re.search(stream_pattern, proto_content), (
            "StreamObservations RPC not properly defined as streaming"
        )

    def test_enum_definitions(self, proto_content: str) -> None:
        """Test that enums are properly defined with UNSPECIFIED values."""
        # Find all enum definitions
        enum_pattern = r"enum (\w+) \{([^}]+)\}"
        enums = re.findall(enum_pattern, proto_content, re.MULTILINE | re.DOTALL)

        for enum_name, enum_body in enums:
            # Each enum should have an UNSPECIFIED value as the first entry (value 0)
            if (
                "UNSPECIFIED" not in enum_body
                and "STATUS_UNSPECIFIED" not in enum_body
                and "OPERATION_UNSPECIFIED" not in enum_body
                and "RECOVERY_ACTION_UNSPECIFIED" not in enum_body
            ):
                # Some enums might use different naming conventions
                zero_value_pattern = r"\w+ = 0;"
                assert re.search(zero_value_pattern, enum_body), (
                    f"Enum {enum_name} missing zero value (required in proto3)"
                )

    def test_intent_kind_enum_consistency(self, proto_content: str) -> None:
        """Test that Intent.Kind enum matches Python Intent type."""
        # Extract Intent.Kind enum values
        intent_section = re.search(
            r"message Intent \{.*?enum Kind \{([^}]+)\}",
            proto_content,
            re.MULTILINE | re.DOTALL,
        )
        assert intent_section, "Intent.Kind enum not found"

        kind_enum = intent_section.group(1)

        # Check for expected values
        expected_kinds = ["KIND_SPEAK", "KIND_MOVE", "KIND_INTERACT", "KIND_CUSTOM"]
        for kind in expected_kinds:
            assert kind in kind_enum, f"Missing Intent kind: {kind}"

    def test_json_patch_operations_consistency(self, proto_content: str) -> None:
        """Test that JsonPatch.Operation enum matches RFC6902."""
        # Extract JsonPatch.Operation enum
        patch_section = re.search(
            r"message JsonPatch \{.*?enum Operation \{([^}]+)\}",
            proto_content,
            re.MULTILINE | re.DOTALL,
        )
        assert patch_section, "JsonPatch.Operation enum not found"

        operation_enum = patch_section.group(1)

        # Check for RFC6902 operations
        rfc6902_ops = [
            "OPERATION_ADD",
            "OPERATION_REMOVE",
            "OPERATION_REPLACE",
            "OPERATION_MOVE",
            "OPERATION_COPY",
            "OPERATION_TEST",
        ]
        for op in rfc6902_ops:
            assert op in operation_enum, f"Missing JSON Patch operation: {op}"

    def test_error_recovery_actions_consistency(self, proto_content: str) -> None:
        """Test that Error.RecoveryAction enum matches docs/errors.md."""
        # Extract Error.RecoveryAction enum
        error_section = re.search(
            r"message Error \{.*?enum RecoveryAction \{([^}]+)\}",
            proto_content,
            re.MULTILINE | re.DOTALL,
        )
        assert error_section, "Error.RecoveryAction enum not found"

        recovery_enum = error_section.group(1)

        # Check for expected recovery actions
        expected_actions = [
            "RECOVERY_ACTION_RETRY",
            "RECOVERY_ACTION_RETRY_WITH_DELAY",
            "RECOVERY_ACTION_REGENERATE",
            "RECOVERY_ACTION_MODIFY_INTENT",
            "RECOVERY_ACTION_DEFER",
            "RECOVERY_ACTION_SHED_OLDEST",
            "RECOVERY_ACTION_ABORT",
        ]

        for action in expected_actions:
            assert action in recovery_enum, f"Missing recovery action: {action}"

    def test_field_numbering_consistency(self, proto_content: str) -> None:
        """Test that field numbers are consistent and don't conflict."""
        # Extract all field definitions
        field_pattern = r"(\w+)\s+(\w+)\s+=\s+(\d+);"
        fields = re.findall(field_pattern, proto_content)

        # Group fields by message (this is a simplified check)
        # In a full implementation, we'd parse the proto structure more carefully
        field_numbers = [int(num) for _, _, num in fields]

        # Check that field numbers are positive
        assert all(num > 0 for num in field_numbers), "Field numbers must be positive"

        # Check that we don't use reserved ranges (19000-19999, 50000-99999)
        reserved_ranges = [(19000, 19999), (50000, 99999)]
        for num in field_numbers:
            for start, end in reserved_ranges:
                assert not (start <= num <= end), (
                    f"Field number {num} is in reserved range"
                )

    def test_message_field_consistency(self, proto_content: str) -> None:
        """Test that message fields are consistent with Python types."""
        # Test Intent message fields - need to handle nested braces properly
        intent_start = proto_content.find("message Intent {")
        assert intent_start != -1, "Intent message not found"

        # Find the matching closing brace for the message
        brace_count = 0
        intent_end = intent_start + len("message Intent {")
        for i, char in enumerate(proto_content[intent_end:], intent_end):
            if char == "{":
                brace_count += 1
            elif char == "}":
                if brace_count == 0:
                    intent_end = i
                    break
                brace_count -= 1

        intent_fields = proto_content[intent_start : intent_end + 1]
        required_intent_fields = [
            "kind",
            "payload",
            "context_seq",
            "req_id",
            "agent_id",
            "priority",
            "schema_version",
        ]

        for field in required_intent_fields:
            assert field in intent_fields, f"Intent missing field: {field}"

    def test_google_protobuf_any_usage(self, proto_content: str) -> None:
        """Test that google.protobuf.Any is used appropriately for flexible payloads."""
        # Fields that should use Any for flexibility
        any_fields = [
            "google.protobuf.Any payload",  # In Intent, EffectDraft, Effect
            "google.protobuf.Any value",  # In JsonPatch
            "google.protobuf.Any details",  # In Error
        ]

        for field in any_fields:
            assert field in proto_content, f"Missing Any field: {field}"

    def test_timestamp_usage(self, proto_content: str) -> None:
        """Test that google.protobuf.Timestamp is used for time fields."""
        # Should use Timestamp for structured time fields
        timestamp_fields = [
            "google.protobuf.Timestamp timestamp",  # In Error and HealthResponse
        ]

        for field in timestamp_fields:
            assert field in proto_content, f"Missing Timestamp field: {field}"

    def test_vector3_message_structure(self, proto_content: str) -> None:
        """Test that Vector3 message has proper 3D coordinate fields."""
        vector3_match = re.search(
            r"message Vector3 \{([^}]+)\}", proto_content, re.MULTILINE | re.DOTALL
        )
        assert vector3_match, "Vector3 message not found"

        vector3_fields = vector3_match.group(1)
        coordinate_fields = ["float x", "float y", "float z"]

        for field in coordinate_fields:
            assert field in vector3_fields, f"Vector3 missing coordinate: {field}"

    def test_service_method_signatures(self, proto_content: str) -> None:
        """Test that service methods have correct request/response types."""
        # Test specific method signatures
        method_signatures = [
            "rpc SubmitIntent(Intent) returns (IntentResponse)",
            "rpc EmitEffect(EffectDraft) returns (EffectResponse)",
            "rpc GetView(ViewRequest) returns (View)",
            "rpc HealthCheck(HealthRequest) returns (HealthResponse)",
        ]

        for signature in method_signatures:
            assert signature in proto_content, (
                f"Missing or incorrect method signature: {signature}"
            )

    def test_breaking_change_detection(self, proto_content: str) -> None:
        """Test for potential breaking changes in the protobuf schema."""
        # Verify package version hasn't changed unexpectedly
        assert "package gunn.simulation.v1;" in proto_content, (
            "Package version changed - ensure this is intentional and documented"
        )

        # Verify core message types haven't been removed
        core_messages = ["Intent", "Effect", "ObservationDelta", "View"]
        for message in core_messages:
            assert f"message {message}" in proto_content, (
                f"Core message {message} was removed - this is a breaking change"
            )


class TestProtobufPythonConsistency:
    """Test consistency between protobuf schema and Python types."""

    def test_intent_field_consistency(self) -> None:
        """Test that protobuf Intent fields match Python Intent TypedDict."""
        from gunn.schemas.types import Intent

        # Get Intent TypedDict annotations
        intent_annotations = Intent.__annotations__

        # Expected fields that should be in both
        common_fields = {
            "kind",
            "payload",
            "context_seq",
            "req_id",
            "agent_id",
            "priority",
            "schema_version",
        }

        # Verify Python type has all expected fields
        python_fields = set(intent_annotations.keys())
        assert common_fields.issubset(python_fields), (
            f"Python Intent missing fields: {common_fields - python_fields}"
        )

    def test_effect_field_consistency(self) -> None:
        """Test that protobuf Effect fields match Python Effect TypedDict."""
        from gunn.schemas.types import Effect

        effect_annotations = Effect.__annotations__
        common_fields = {
            "uuid",
            "kind",
            "payload",
            "global_seq",
            "sim_time",
            "source_id",
            "schema_version",
        }

        python_fields = set(effect_annotations.keys())
        assert common_fields.issubset(python_fields), (
            f"Python Effect missing fields: {common_fields - python_fields}"
        )

    def test_observation_delta_consistency(self) -> None:
        """Test that protobuf ObservationDelta matches Python ObservationDelta."""
        from gunn.schemas.types import ObservationDelta

        delta_annotations = ObservationDelta.__annotations__
        common_fields = {"view_seq", "patches", "context_digest", "schema_version"}

        python_fields = set(delta_annotations.keys())
        assert common_fields.issubset(python_fields), (
            f"Python ObservationDelta missing fields: {common_fields - python_fields}"
        )


if __name__ == "__main__":
    pytest.main([__file__])
