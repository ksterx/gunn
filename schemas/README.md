# Contract Definitions

This directory contains golden files for API contracts and schema definitions.

## Files

- `openapi.yaml` - REST API contract for the Web adapter
- `proto/` - Protocol buffer definitions for gRPC communication

## Purpose

These files serve as the single source of truth for external API contracts and are used in CI to detect breaking changes.

## Validation

Contract tests in `tests/contract/` validate that the actual API implementation matches these schema definitions.
