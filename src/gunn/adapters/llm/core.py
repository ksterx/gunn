from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class LLMProvider(StrEnum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE_OPENAI = "azure_openai"
    BEDROCK_ANTHROPIC = "bedrock_anthropic"
    VERTEX = "vertex"
    HUGGINGFACE = "huggingface"


AVAILABLE_MODELS = {
    LLMProvider.OPENAI: ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1", "gpt-4.1-mini"],
    LLMProvider.ANTHROPIC: [
        "claude-4-sonnet",
        "claude-4-haiku",
        "claude-4.1-opus",
        "claude-4.5-sonnet",
    ],
    LLMProvider.GOOGLE: ["gemini-2.5-pro", "gemini-2.5-flash"],
}


class LLMConfig(BaseModel):
    """Configuration for LLM adapter."""

    provider: LLMProvider = Field(..., description="The provider to use")
    model: str = Field(..., description="The model to use")
    api_key: str | None = Field(None, description="The API key to use")
    api_base: str | None = Field(None, description="The API base to use")
    max_tokens: int = Field(
        1000, description="The maximum number of tokens to generate"
    )
    temperature: float = Field(0.7, description="The temperature for the model")
    timeout_seconds: float = Field(30.0, description="The timeout in seconds")
    token_yield_interval_ms: float = Field(
        25.0,
        description="The interval in milliseconds to yield control. 20-30ms for responsive cancellation",
    )
    retry_attempts: int = Field(3, description="The number of retry attempts")
    retry_delay_seconds: float = Field(
        1.0, description="The delay in seconds between retry attempts"
    )


class GenerationRequest(BaseModel):
    """Request for LLM generation."""

    prompt: str = Field(..., description="The prompt to generate text from")
    max_tokens: int | None = None
    temperature: float | None = Field(None, description="The temperature for the model")
    stop_sequences: list[str] | None = Field(
        None, description="The sequences to stop generation"
    )
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")


class GenerationResponse(BaseModel):
    """Response from LLM generation."""

    content: str = Field(..., description="The content generated")
    token_count: int = Field(..., description="The number of tokens generated")
    generation_time_ms: float = Field(
        ..., description="The generation time in milliseconds"
    )
    cancelled: bool = Field(False, description="Whether the generation was cancelled")
    cancellation_time_ms: float | None = Field(
        None, description="The cancellation time in milliseconds"
    )
    error: str | None = Field(
        None, description="The error message if generation failed"
    )
