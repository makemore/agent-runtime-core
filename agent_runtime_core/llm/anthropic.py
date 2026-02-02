"""
Anthropic API client implementation.
"""

import json
import os
from typing import AsyncIterator, Optional

from agent_runtime_core.interfaces import (
    LLMClient,
    LLMResponse,
    LLMStreamChunk,
    Message,
)

try:
    from anthropic import AsyncAnthropic, APIError
except ImportError:
    AsyncAnthropic = None
    APIError = Exception


class AnthropicConfigurationError(Exception):
    """Raised when Anthropic API key is not configured."""
    pass


class AnthropicClient(LLMClient):
    """
    Anthropic API client.

    Supports Claude models.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        **kwargs,
    ):
        if AsyncAnthropic is None:
            raise ImportError(
                "anthropic package is required for AnthropicClient. "
                "Install it with: pip install agent-runtime-core[anthropic]"
            )

        from agent_runtime_core.config import get_config
        config = get_config()
        
        self.default_model = default_model or config.default_model or "claude-sonnet-4-20250514"
        
        # Resolve API key with clear priority
        resolved_api_key = self._resolve_api_key(api_key)
        
        if not resolved_api_key:
            raise AnthropicConfigurationError(
                "Anthropic API key is not configured.\n\n"
                "Configure it using one of these methods:\n"
                "  1. Use configure():\n"
                "     from agent_runtime_core.config import configure\n"
                "     configure(anthropic_api_key='sk-ant-...')\n\n"
                "  2. Set the ANTHROPIC_API_KEY environment variable:\n"
                "     export ANTHROPIC_API_KEY='sk-ant-...'\n\n"
                "  3. Pass api_key directly to get_llm_client():\n"
                "     llm = get_llm_client(api_key='sk-ant-...')"
            )
        
        self._client = AsyncAnthropic(
            api_key=resolved_api_key,
            **kwargs,
        )

    def _resolve_api_key(self, explicit_key: Optional[str]) -> Optional[str]:
        """
        Resolve API key with clear priority order.
        
        Priority:
        1. Explicit api_key parameter passed to __init__
        2. anthropic_api_key in config
        3. ANTHROPIC_API_KEY environment variable
        """
        if explicit_key:
            return explicit_key
        
        from agent_runtime_core.config import get_config
        config = get_config()
        settings_key = config.get_anthropic_api_key()
        if settings_key:
            return settings_key
        
        return os.environ.get("ANTHROPIC_API_KEY")

    def _validate_tool_call_pairs(self, messages: list[Message]) -> list[Message]:
        """
        Validate and repair tool_use/tool_result pairing in message history.

        Anthropic requires that every tool_use block has a corresponding tool_result
        block immediately after. This can be violated if a run fails mid-way through
        tool execution (e.g., timeout, crash, API error during parallel tool calls).

        This method removes orphaned tool_use blocks (assistant messages with tool_calls
        that don't have corresponding tool results).

        Args:
            messages: List of messages in framework-neutral format

        Returns:
            Cleaned list of messages with orphaned tool_use blocks removed
        """
        if not messages:
            return messages

        # First pass: collect all tool_call_ids that have results
        tool_result_ids = set()
        for msg in messages:
            if msg.get("role") == "tool" and msg.get("tool_call_id"):
                tool_result_ids.add(msg["tool_call_id"])

        # Second pass: check each assistant message with tool_calls
        cleaned_messages = []
        orphaned_count = 0

        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                # Check which tool_calls have results
                tool_calls = msg.get("tool_calls", [])
                valid_tool_calls = []
                orphaned_ids = []

                for tc in tool_calls:
                    # Handle both formats: {"id": ...} and {"function": {...}, "id": ...}
                    tc_id = tc.get("id")
                    if tc_id in tool_result_ids:
                        valid_tool_calls.append(tc)
                    else:
                        orphaned_ids.append(tc_id)

                if orphaned_ids:
                    orphaned_count += len(orphaned_ids)
                    print(
                        f"[anthropic] Removing {len(orphaned_ids)} orphaned tool_use blocks "
                        f"without results: {orphaned_ids[:3]}{'...' if len(orphaned_ids) > 3 else ''}",
                        flush=True,
                    )

                if valid_tool_calls:
                    # Keep the message but only with valid tool_calls
                    cleaned_msg = msg.copy()
                    cleaned_msg["tool_calls"] = valid_tool_calls
                    cleaned_messages.append(cleaned_msg)
                elif msg.get("content"):
                    # No valid tool_calls but has text content - keep as regular message
                    cleaned_msg = {
                        "role": "assistant",
                        "content": msg["content"],
                    }
                    cleaned_messages.append(cleaned_msg)
                # else: skip the message entirely (no valid tool_calls, no content)
            else:
                cleaned_messages.append(msg)

        if orphaned_count > 0:
            print(
                f"[anthropic] Cleaned {orphaned_count} orphaned tool_use blocks from message history",
                flush=True,
            )

        return cleaned_messages

    async def generate(
        self,
        messages: list[Message],
        *,
        model: Optional[str] = None,
        stream: bool = False,
        tools: Optional[list[dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        thinking: bool = False,
        thinking_budget: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a completion from Anthropic.

        Args:
            messages: List of messages in framework-neutral format
            model: Model ID to use (defaults to self.default_model)
            stream: Whether to stream the response (not used here, use stream() method)
            tools: List of tools in OpenAI format
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            thinking: Enable extended thinking mode for deeper reasoning
            thinking_budget: Max tokens for thinking (default: 10000, max: 128000)
            **kwargs: Additional parameters passed to the API

        Returns:
            LLMResponse with the generated message
        """
        model = model or self.default_model

        # Validate and repair message history before processing
        messages = self._validate_tool_call_pairs(messages)

        # Extract system message and convert other messages
        system_message = None
        converted_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_message = msg.get("content", "")
            else:
                converted_messages.append(self._convert_message(msg))

        # Merge consecutive messages with the same role (required by Anthropic)
        chat_messages = self._merge_consecutive_messages(converted_messages)

        request_kwargs = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": max_tokens or 4096,
        }

        if system_message:
            request_kwargs["system"] = system_message
        if tools:
            request_kwargs["tools"] = self._convert_tools(tools)

        # Handle extended thinking mode
        if thinking:
            # Extended thinking requires specific configuration
            # Temperature must be 1.0 when using thinking
            request_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget or 10000,
            }
            # Temperature must be exactly 1.0 for extended thinking
            request_kwargs["temperature"] = 1.0
        elif temperature is not None:
            request_kwargs["temperature"] = temperature

        request_kwargs.update(kwargs)

        response = await self._client.messages.create(**request_kwargs)

        message, thinking_content = self._convert_response(response)

        return LLMResponse(
            message=message,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            model=response.model,
            finish_reason=response.stop_reason or "",
            raw_response=response,
            thinking=thinking_content,
        )

    async def stream(
        self,
        messages: list[Message],
        *,
        model: Optional[str] = None,
        tools: Optional[list[dict]] = None,
        thinking: bool = False,
        thinking_budget: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> AsyncIterator[LLMStreamChunk]:
        """
        Stream a completion from Anthropic.

        Args:
            messages: List of messages in framework-neutral format
            model: Model ID to use
            tools: List of tools in OpenAI format
            thinking: Enable extended thinking mode
            thinking_budget: Max tokens for thinking (default: 10000)
            temperature: Sampling temperature (ignored if thinking=True)
            **kwargs: Additional parameters

        Yields:
            LLMStreamChunk with delta content and thinking content
        """
        model = model or self.default_model

        # Validate and repair message history before processing
        messages = self._validate_tool_call_pairs(messages)

        # Extract system message and convert other messages
        system_message = None
        converted_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_message = msg.get("content", "")
            else:
                converted_messages.append(self._convert_message(msg))

        # Merge consecutive messages with the same role (required by Anthropic)
        chat_messages = self._merge_consecutive_messages(converted_messages)

        request_kwargs = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": kwargs.pop("max_tokens", 4096),
        }

        if system_message:
            request_kwargs["system"] = system_message
        if tools:
            request_kwargs["tools"] = self._convert_tools(tools)

        # Handle extended thinking mode
        if thinking:
            request_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget or 10000,
            }
            request_kwargs["temperature"] = 1.0
        elif temperature is not None:
            request_kwargs["temperature"] = temperature

        request_kwargs.update(kwargs)

        async with self._client.messages.stream(**request_kwargs) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        yield LLMStreamChunk(delta=event.delta.text)
                    elif hasattr(event.delta, "thinking"):
                        # Extended thinking content
                        yield LLMStreamChunk(delta="", thinking=event.delta.thinking)
                elif event.type == "message_stop":
                    yield LLMStreamChunk(finish_reason="stop")

    def _convert_message(self, msg: Message) -> dict:
        """
        Convert our message format to Anthropic format.

        Handles:
        - Regular user/assistant messages
        - Assistant messages with tool_calls (need content blocks)
        - Tool result messages (need tool_result content blocks)
        """
        role = msg.get("role", "user")

        # Handle tool result messages
        if role == "tool":
            # Tool results go as user messages with tool_result content blocks
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": msg.get("content", ""),
                    }
                ],
            }

        # Handle assistant messages with tool_calls
        if role == "assistant" and msg.get("tool_calls"):
            content_blocks = []

            # Add text content if present
            text_content = msg.get("content", "")
            if text_content:
                content_blocks.append({
                    "type": "text",
                    "text": text_content,
                })

            # Add tool_use blocks for each tool call
            for tool_call in msg.get("tool_calls", []):
                # Handle both dict format and nested function format
                if "function" in tool_call:
                    # OpenAI-style format: {"id": ..., "function": {"name": ..., "arguments": ...}}
                    func = tool_call["function"]
                    tool_id = tool_call.get("id", "")
                    tool_name = func.get("name", "")
                    # Arguments might be a string (JSON) or already a dict
                    args = func.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                else:
                    # Direct format: {"id": ..., "name": ..., "arguments": ...}
                    tool_id = tool_call.get("id", "")
                    tool_name = tool_call.get("name", "")
                    args = tool_call.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}

                content_blocks.append({
                    "type": "tool_use",
                    "id": tool_id,
                    "name": tool_name,
                    "input": args,
                })

            return {
                "role": "assistant",
                "content": content_blocks,
            }

        # Regular user or assistant message
        return {
            "role": role,
            "content": msg.get("content", ""),
        }

    def _merge_consecutive_messages(self, messages: list[dict]) -> list[dict]:
        """
        Merge consecutive messages with the same role.

        Anthropic requires that messages alternate between user and assistant roles.
        When we have multiple tool results (which become user messages), they need
        to be combined into a single user message with multiple content blocks.
        """
        if not messages:
            return messages

        merged = []
        for msg in messages:
            if not merged:
                merged.append(msg)
                continue

            last_msg = merged[-1]

            # If same role, merge the content
            if msg["role"] == last_msg["role"]:
                last_content = last_msg["content"]
                new_content = msg["content"]

                # Convert to list format if needed
                if isinstance(last_content, str):
                    if last_content:
                        last_content = [{"type": "text", "text": last_content}]
                    else:
                        last_content = []

                if isinstance(new_content, str):
                    if new_content:
                        new_content = [{"type": "text", "text": new_content}]
                    else:
                        new_content = []

                # Merge content blocks
                last_msg["content"] = last_content + new_content
            else:
                merged.append(msg)

        return merged

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert OpenAI tool format to Anthropic format."""
        result = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                result.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {}),
                })
        return result

    def _convert_response(self, response) -> tuple[Message, Optional[str]]:
        """
        Convert Anthropic response to our format.

        Returns:
            Tuple of (message, thinking_content)
        """
        content = ""
        tool_calls = []
        thinking_content = None

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "thinking":
                # Extended thinking block
                thinking_content = block.thinking
            elif block.type == "tool_use":
                # Convert input to JSON string (not Python str() which gives wrong format)
                arguments = json.dumps(block.input) if isinstance(block.input, dict) else str(block.input)
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": arguments,
                    },
                })

        result: Message = {
            "role": "assistant",
            "content": content,
        }

        if tool_calls:
            result["tool_calls"] = tool_calls

        return result, thinking_content
