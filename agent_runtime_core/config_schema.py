"""
AgentConfig - Canonical JSON schema for portable agent definitions.

This schema defines the format for agent configurations that can be:
1. Stored in Django as JSON revisions
2. Loaded from standalone .json files
3. Used by agent_runtime_core without Django dependency

Example:
    # Load from file
    config = AgentConfig.from_file("my_agent.json")
    
    # Create runtime and run
    runtime = JsonAgentRuntime(config, llm_client)
    result = await runtime.run(ctx)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import json
from pathlib import Path


@dataclass
class SubAgentToolConfig:
    """
    Configuration for a sub-agent tool (agent-as-tool pattern).

    This allows an agent to delegate to another agent as if it were a tool.
    The sub-agent can either be referenced by slug (resolved at runtime)
    or embedded inline (for portable standalone configs).
    """

    name: str  # Tool name the parent uses to invoke this agent
    description: str  # When to use this agent (shown to parent LLM)

    # Reference to sub-agent (one of these should be set)
    agent_slug: str = ""  # Reference by slug (resolved at runtime from registry)
    agent_config: Optional["AgentConfig"] = None  # Embedded config (for standalone)

    # Invocation settings
    invocation_mode: str = "delegate"  # "delegate" or "handoff"
    context_mode: str = "full"  # "full", "summary", or "message_only"
    max_turns: Optional[int] = None

    def to_dict(self) -> dict:
        result = {
            "name": self.name,
            "description": self.description,
            "tool_type": "subagent",
            "invocation_mode": self.invocation_mode,
            "context_mode": self.context_mode,
        }
        if self.agent_slug:
            result["agent_slug"] = self.agent_slug
        if self.agent_config:
            result["agent_config"] = self.agent_config.to_dict()
        if self.max_turns is not None:
            result["max_turns"] = self.max_turns
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "SubAgentToolConfig":
        agent_config = None
        if "agent_config" in data and data["agent_config"]:
            # Defer import to avoid circular dependency
            agent_config = AgentConfig.from_dict(data["agent_config"])

        return cls(
            name=data["name"],
            description=data["description"],
            agent_slug=data.get("agent_slug", ""),
            agent_config=agent_config,
            invocation_mode=data.get("invocation_mode", "delegate"),
            context_mode=data.get("context_mode", "full"),
            max_turns=data.get("max_turns"),
        )


@dataclass
class ToolConfig:
    """Configuration for a single tool."""

    name: str
    description: str
    parameters: dict  # JSON Schema for parameters
    function_path: str  # Import path like "myapp.services.orders.lookup_order"

    # Optional metadata
    requires_confirmation: bool = False
    is_safe: bool = True  # No side effects
    timeout_seconds: int = 30

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "function_path": self.function_path,
            "requires_confirmation": self.requires_confirmation,
            "is_safe": self.is_safe,
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ToolConfig":
        return cls(
            name=data["name"],
            description=data["description"],
            parameters=data.get("parameters", {}),
            function_path=data.get("function_path", ""),
            requires_confirmation=data.get("requires_confirmation", False),
            is_safe=data.get("is_safe", True),
            timeout_seconds=data.get("timeout_seconds", 30),
        )


@dataclass
class KnowledgeConfig:
    """Configuration for a knowledge source."""
    
    name: str
    knowledge_type: str  # "text", "file", "url"
    inclusion_mode: str = "always"  # "always", "on_demand", "rag"
    
    # Content (depends on type)
    content: str = ""  # For text type
    file_path: str = ""  # For file type
    url: str = ""  # For url type
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.knowledge_type,
            "inclusion_mode": self.inclusion_mode,
            "content": self.content,
            "file_path": self.file_path,
            "url": self.url,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeConfig":
        return cls(
            name=data["name"],
            knowledge_type=data.get("type", "text"),
            inclusion_mode=data.get("inclusion_mode", "always"),
            content=data.get("content", ""),
            file_path=data.get("file_path", ""),
            url=data.get("url", ""),
        )


@dataclass
class AgentConfig:
    """
    Canonical configuration for an agent.

    This is the portable format that can be serialized to JSON and
    loaded by any runtime (Django or standalone).

    For multi-agent systems, sub_agents contains embedded agent configs
    that this agent can delegate to. The sub_agent_tools list defines
    how each sub-agent is exposed as a tool.
    """

    # Identity
    name: str
    slug: str
    description: str = ""

    # Core configuration
    system_prompt: str = ""
    model: str = "gpt-4o"
    model_settings: dict = field(default_factory=dict)

    # Tools and knowledge
    tools: list[ToolConfig] = field(default_factory=list)
    knowledge: list[KnowledgeConfig] = field(default_factory=list)

    # Sub-agent tools (agent-as-tool pattern)
    # These define other agents this agent can delegate to
    sub_agent_tools: list[SubAgentToolConfig] = field(default_factory=list)

    # Metadata
    version: str = "1.0"
    schema_version: str = "1"  # For future schema migrations
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    # Extra config (for extensibility)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dictionary (JSON-compatible)."""
        result = {
            "schema_version": self.schema_version,
            "version": self.version,
            "name": self.name,
            "slug": self.slug,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "model": self.model,
            "model_settings": self.model_settings,
            "tools": [t.to_dict() for t in self.tools],
            "knowledge": [k.to_dict() for k in self.knowledge],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "extra": self.extra,
        }
        # Only include sub_agent_tools if there are any
        if self.sub_agent_tools:
            result["sub_agent_tools"] = [s.to_dict() for s in self.sub_agent_tools]
        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> None:
        """Save to a JSON file."""
        path = Path(path)
        path.write_text(self.to_json())

    @classmethod
    def from_dict(cls, data: dict) -> "AgentConfig":
        """Load from dictionary."""
        # Parse regular tools (skip subagent tools - they're in sub_agent_tools)
        tools = []
        for t in data.get("tools", []):
            if t.get("tool_type") != "subagent":
                tools.append(ToolConfig.from_dict(t))

        knowledge = [KnowledgeConfig.from_dict(k) for k in data.get("knowledge", [])]

        # Parse sub-agent tools
        sub_agent_tools = []
        for s in data.get("sub_agent_tools", []):
            sub_agent_tools.append(SubAgentToolConfig.from_dict(s))
        # Also check tools list for subagent type (backwards compat)
        for t in data.get("tools", []):
            if t.get("tool_type") == "subagent":
                sub_agent_tools.append(SubAgentToolConfig.from_dict(t))

        return cls(
            name=data["name"],
            slug=data["slug"],
            description=data.get("description", ""),
            system_prompt=data.get("system_prompt", ""),
            model=data.get("model", "gpt-4o"),
            model_settings=data.get("model_settings", {}),
            tools=tools,
            knowledge=knowledge,
            sub_agent_tools=sub_agent_tools,
            version=data.get("version", "1.0"),
            schema_version=data.get("schema_version", "1"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            extra=data.get("extra", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "AgentConfig":
        """Load from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_file(cls, path: str | Path) -> "AgentConfig":
        """Load from a JSON file."""
        path = Path(path)
        return cls.from_json(path.read_text())

    def with_timestamp(self) -> "AgentConfig":
        """Return a copy with updated timestamp."""
        now = datetime.utcnow().isoformat() + "Z"
        return AgentConfig(
            name=self.name,
            slug=self.slug,
            description=self.description,
            system_prompt=self.system_prompt,
            model=self.model,
            model_settings=self.model_settings,
            tools=self.tools,
            knowledge=self.knowledge,
            sub_agent_tools=self.sub_agent_tools,
            version=self.version,
            schema_version=self.schema_version,
            created_at=self.created_at or now,
            updated_at=now,
            extra=self.extra,
        )

    def get_all_embedded_agents(self) -> dict[str, "AgentConfig"]:
        """
        Get all embedded agent configs (for standalone execution).

        Returns a dict mapping slug -> AgentConfig for all sub-agents
        that have embedded configs (not just slug references).
        """
        agents = {}
        for sub_tool in self.sub_agent_tools:
            if sub_tool.agent_config:
                agents[sub_tool.agent_config.slug] = sub_tool.agent_config
                # Recursively get nested sub-agents
                agents.update(sub_tool.agent_config.get_all_embedded_agents())
        return agents

