"""
Tests for the Anthropic LLM client.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestValidateToolCallPairs:
    """Tests for _validate_tool_call_pairs method."""

    @pytest.fixture
    def client(self):
        """Create an AnthropicClient with mocked dependencies."""
        with patch("agent_runtime_core.llm.anthropic.AsyncAnthropic"):
            from agent_runtime_core.llm.anthropic import AnthropicClient
            return AnthropicClient(api_key="test-key")

    def test_empty_messages(self, client):
        """Empty messages should return empty list."""
        result = client._validate_tool_call_pairs([])
        assert result == []

    def test_no_tool_calls(self, client):
        """Messages without tool calls should pass through unchanged."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = client._validate_tool_call_pairs(messages)
        assert result == messages

    def test_valid_tool_call_pairs(self, client):
        """Valid tool call/result pairs should pass through unchanged."""
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "tc1", "name": "get_weather", "arguments": {}}],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": '{"temp": 72}'},
            {"role": "assistant", "content": "It's 72 degrees."},
        ]
        result = client._validate_tool_call_pairs(messages)
        assert len(result) == 4
        assert result[1]["tool_calls"] == [{"id": "tc1", "name": "get_weather", "arguments": {}}]

    def test_orphaned_tool_calls_removed(self, client, capsys):
        """Orphaned tool calls (without results) should be removed."""
        messages = [
            {"role": "user", "content": "Do two things"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc1", "name": "tool1", "arguments": {}},
                    {"id": "tc2", "name": "tool2", "arguments": {}},
                ],
            },
            # Only tc1 has a result, tc2 is orphaned
            {"role": "tool", "tool_call_id": "tc1", "content": "result1"},
        ]
        result = client._validate_tool_call_pairs(messages)
        
        # Should have 3 messages
        assert len(result) == 3
        # Assistant message should only have tc1
        assert len(result[1]["tool_calls"]) == 1
        assert result[1]["tool_calls"][0]["id"] == "tc1"
        
        # Check that warning was printed
        captured = capsys.readouterr()
        assert "orphaned tool_use blocks" in captured.out

    def test_all_tool_calls_orphaned_with_content(self, client):
        """If all tool calls are orphaned but message has content, keep as regular message."""
        messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "Let me check that for you.",
                "tool_calls": [{"id": "tc1", "name": "tool1", "arguments": {}}],
            },
            # No tool result for tc1
        ]
        result = client._validate_tool_call_pairs(messages)
        
        assert len(result) == 2
        # Assistant message should be kept but without tool_calls
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "Let me check that for you."
        assert "tool_calls" not in result[1]

    def test_all_tool_calls_orphaned_no_content(self, client):
        """If all tool calls are orphaned and no content, skip the message entirely."""
        messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "tc1", "name": "tool1", "arguments": {}}],
            },
            # No tool result for tc1
            {"role": "user", "content": "What happened?"},
        ]
        result = client._validate_tool_call_pairs(messages)
        
        # Should skip the orphaned assistant message
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "What happened?"

    def test_multiple_parallel_tool_calls_partial_results(self, client):
        """Handle case where some parallel tool calls have results and some don't."""
        messages = [
            {"role": "user", "content": "Do many things"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc1", "name": "tool1", "arguments": {}},
                    {"id": "tc2", "name": "tool2", "arguments": {}},
                    {"id": "tc3", "name": "tool3", "arguments": {}},
                    {"id": "tc4", "name": "tool4", "arguments": {}},
                ],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "result1"},
            {"role": "tool", "tool_call_id": "tc3", "content": "result3"},
            # tc2 and tc4 are orphaned
        ]
        result = client._validate_tool_call_pairs(messages)
        
        # Should have 4 messages (user, assistant with 2 tool_calls, 2 tool results)
        assert len(result) == 4
        # Assistant message should only have tc1 and tc3
        assert len(result[1]["tool_calls"]) == 2
        tool_ids = [tc["id"] for tc in result[1]["tool_calls"]]
        assert "tc1" in tool_ids
        assert "tc3" in tool_ids
        assert "tc2" not in tool_ids
        assert "tc4" not in tool_ids

