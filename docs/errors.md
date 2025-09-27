# Error Codes and Status Mappings

This document serves as the single source of truth for all error codes and their mappings to HTTP and gRPC status codes in the gunn multi-agent simulation system.

## Error Code Categories

### 1000-1999: Context and State Errors

| Code | Name | Description | HTTP Status | gRPC Status | Recovery Action |
|------|------|-------------|-------------|-------------|-----------------|
| 1001 | STALE_CONTEXT | Agent's context is outdated relative to current world state | 409 Conflict | FAILED_PRECONDITION | REGENERATE |
| 1002 | INVALID_VIEW_SEQ | Provided view_seq is invalid or out of range | 400 Bad Request | INVALID_ARGUMENT | RETRY |
| 1003 | WORLD_STATE_CORRUPTED | World state integrity check failed | 500 Internal Server Error | INTERNAL | ABORT |
| 1004 | MISSING_CONTEXT | Required context information is missing | 400 Bad Request | INVALID_ARGUMENT | RETRY |

### 2000-2999: Intent and Validation Errors

| Code | Name | Description | HTTP Status | gRPC Status | Recovery Action |
|------|------|-------------|-------------|-------------|-----------------|
| 2001 | INTENT_CONFLICT | Intent conflicts with current world state or other intents | 409 Conflict | FAILED_PRECONDITION | RETRY_WITH_DELAY |
| 2002 | INVALID_INTENT | Intent format or content is invalid | 400 Bad Request | INVALID_ARGUMENT | MODIFY_INTENT |
| 2003 | INTENT_VALIDATION_FAILED | Intent failed validation rules | 422 Unprocessable Entity | FAILED_PRECONDITION | MODIFY_INTENT |
| 2004 | DUPLICATE_REQ_ID | Request ID already exists (idempotency violation) | 409 Conflict | ALREADY_EXISTS | RETRY |

### 3000-3999: Resource and Quota Errors

| Code | Name | Description | HTTP Status | gRPC Status | Recovery Action |
|------|------|-------------|-------------|-------------|-----------------|
| 3001 | QUOTA_EXCEEDED | Agent has exceeded rate or resource quota | 429 Too Many Requests | RESOURCE_EXHAUSTED | DEFER |
| 3002 | QUEUE_FULL | Agent's intent queue is at capacity | 503 Service Unavailable | RESOURCE_EXHAUSTED | SHED_OLDEST |
| 3003 | TOKEN_BUDGET_EXCEEDED | LLM token budget exceeded | 429 Too Many Requests | RESOURCE_EXHAUSTED | DEFER |
| 3004 | MEMORY_LIMIT_EXCEEDED | System memory limit exceeded | 503 Service Unavailable | RESOURCE_EXHAUSTED | SHED_OLDEST |

### 4000-4999: Timing and Cancellation Errors

| Code | Name | Description | HTTP Status | gRPC Status | Recovery Action |
|------|------|-------------|-------------|-------------|-----------------|
| 4001 | OPERATION_TIMEOUT | Operation exceeded deadline | 408 Request Timeout | DEADLINE_EXCEEDED | RETRY |
| 4002 | GENERATION_CANCELLED | LLM generation was cancelled | 499 Client Closed Request | CANCELLED | REGENERATE |
| 4003 | STALENESS_DETECTED | Context became stale during processing | 409 Conflict | FAILED_PRECONDITION | REGENERATE |
| 4004 | DEBOUNCE_ACTIVE | Operation blocked by debounce policy | 429 Too Many Requests | RESOURCE_EXHAUSTED | RETRY_WITH_DELAY |

### 5000-5999: Authentication and Authorization Errors

| Code | Name | Description | HTTP Status | gRPC Status | Recovery Action |
|------|------|-------------|-------------|-------------|-----------------|
| 5001 | UNAUTHORIZED | Authentication required | 401 Unauthorized | UNAUTHENTICATED | ABORT |
| 5002 | FORBIDDEN | Insufficient permissions for operation | 403 Forbidden | PERMISSION_DENIED | ABORT |
| 5003 | INVALID_AGENT_ID | Agent ID is invalid or not found | 404 Not Found | NOT_FOUND | ABORT |
| 5004 | TENANT_ISOLATION_VIOLATION | Cross-tenant access attempted | 403 Forbidden | PERMISSION_DENIED | ABORT |
| 5005 | AGENT_QUARANTINED | Agent is quarantined due to security violation | 403 Forbidden | PERMISSION_DENIED | ABORT |

### 6000-6999: External System Errors

| Code | Name | Description | HTTP Status | gRPC Status | Recovery Action |
|------|------|-------------|-------------|-------------|-----------------|
| 6001 | ADAPTER_UNAVAILABLE | External adapter is not available | 503 Service Unavailable | UNAVAILABLE | RETRY_WITH_DELAY |
| 6002 | LLM_SERVICE_ERROR | LLM service returned an error | 502 Bad Gateway | UNAVAILABLE | RETRY |
| 6003 | UNITY_CONNECTION_LOST | Unity adapter connection lost | 503 Service Unavailable | UNAVAILABLE | RETRY_WITH_DELAY |
| 6004 | WEB_API_ERROR | Web API adapter error | 502 Bad Gateway | UNAVAILABLE | RETRY |

### 7000-7999: System and Infrastructure Errors

| Code | Name | Description | HTTP Status | gRPC Status | Recovery Action |
|------|------|-------------|-------------|-------------|-----------------|
| 7001 | CIRCUIT_BREAKER_OPEN | Circuit breaker is open | 503 Service Unavailable | UNAVAILABLE | RETRY_WITH_DELAY |
| 7002 | DATABASE_ERROR | Database operation failed | 500 Internal Server Error | INTERNAL | RETRY |
| 7003 | LOG_CORRUPTION_DETECTED | Event log corruption detected | 500 Internal Server Error | DATA_LOSS | ABORT |
| 7004 | SCHEMA_VERSION_MISMATCH | Schema version incompatibility | 400 Bad Request | INVALID_ARGUMENT | ABORT |

## Recovery Action Definitions

| Action | Description | Implementation |
|--------|-------------|----------------|
| RETRY | Retry the operation immediately | Resend the same request |
| RETRY_WITH_DELAY | Retry after exponential backoff delay | Wait then resend |
| REGENERATE | Cancel current generation and start fresh | Issue new generation with updated context |
| MODIFY_INTENT | Modify the intent and retry | User/agent must adjust intent parameters |
| DEFER | Defer operation until resources available | Queue for later processing |
| SHED_OLDEST | Drop oldest queued items to make room | Remove old items from queue |
| ABORT | Abort operation permanently | Return error to caller |

## Error Response Format

All errors should be returned in a consistent format:

```json
{
  "error": {
    "code": 1001,
    "name": "STALE_CONTEXT",
    "message": "Agent's context is outdated relative to current world state",
    "details": {
      "agent_id": "agent_123",
      "expected_view_seq": 42,
      "actual_view_seq": 45,
      "req_id": "req_456"
    },
    "recovery_action": "REGENERATE",
    "timestamp": "2024-01-15T10:30:00Z",
    "trace_id": "trace_789"
  }
}
```

## Usage Guidelines

1. **Consistent Mapping**: Always use the same error code for the same error condition
2. **Recovery Actions**: Include recovery action hints in error responses
3. **Structured Details**: Provide structured error details for debugging
4. **Tracing**: Include trace IDs for distributed debugging
5. **Logging**: Log all errors with structured context
6. **Metrics**: Emit metrics for error rates by code and category

## Implementation Notes

- Error codes are immutable once assigned
- New error codes should be added to the appropriate range
- HTTP status codes follow RFC 7231 standards
- gRPC status codes follow the official gRPC status code definitions
- Recovery actions should be implemented consistently across all adapters
