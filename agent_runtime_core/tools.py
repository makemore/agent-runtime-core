"""
Tool utilities for building tool schemas.

This module provides utilities for creating tool schemas that can be passed
to LLMs for function calling. The ToolSchemaBuilder provides a fluent API
for building schemas without manually constructing JSON.

Example:
    from agent_runtime_core.tools import ToolSchemaBuilder
    
    # Build a tool schema fluently
    schema = (
        ToolSchemaBuilder("get_weather")
        .description("Get the current weather for a location")
        .param("location", "string", "City name", required=True)
        .param("units", "string", "Temperature units", enum=["celsius", "fahrenheit"])
        .build()
    )
    
    # Convert to OpenAI format for LLM calls
    openai_format = schema.to_openai_format()

Note: For registering tools with handlers, use Tool and ToolRegistry from
agent_runtime_core.interfaces. This module is for schema building only.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence


@dataclass
class ToolParameter:
    """
    Definition of a tool parameter.
    
    Attributes:
        name: Parameter name
        type: JSON Schema type (string, integer, number, boolean, array, object)
        description: Human-readable description
        required: Whether the parameter is required
        enum: Optional list of allowed values
        items: For array types, the schema of array items
        default: Default value if not provided
    """
    name: str
    type: str
    description: str = ""
    required: bool = False
    enum: Optional[list[str]] = None
    items: Optional[dict[str, Any]] = None
    default: Optional[Any] = None
    
    def to_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema format."""
        schema: dict[str, Any] = {"type": self.type}
        if self.description:
            schema["description"] = self.description
        if self.enum:
            schema["enum"] = self.enum
        if self.items:
            schema["items"] = self.items
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class ToolSchema:
    """
    Complete tool schema in OpenAI function calling format.
    
    Example:
        schema = ToolSchema(
            name="get_weather",
            description="Get the current weather for a location",
            parameters=[
                ToolParameter("location", "string", "City name", required=True),
                ToolParameter("units", "string", "Temperature units", enum=["celsius", "fahrenheit"]),
            ],
        )
        
        # Convert to OpenAI format
        openai_schema = schema.to_openai_format()
    """
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    
    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class ToolSchemaBuilder:
    """
    Builder for creating tool schemas fluently.
    
    Example:
        schema = (
            ToolSchemaBuilder("submit_order")
            .description("Submit a customer order")
            .param("customer_id", "string", "Customer ID", required=True)
            .param("items", "array", "Order items", items={"type": "string"})
            .param("priority", "string", "Priority level", enum=["low", "normal", "high"])
            .build()
        )
    """
    
    def __init__(self, name: str):
        """Initialize builder with tool name."""
        self._name = name
        self._description = ""
        self._parameters: list[ToolParameter] = []
    
    def description(self, desc: str) -> "ToolSchemaBuilder":
        """Set the tool description."""
        self._description = desc
        return self
    
    def param(
        self,
        name: str,
        type: str,
        description: str = "",
        required: bool = False,
        enum: Optional[list[str]] = None,
        items: Optional[dict[str, Any]] = None,
        default: Optional[Any] = None,
    ) -> "ToolSchemaBuilder":
        """Add a parameter to the tool."""
        self._parameters.append(
            ToolParameter(
                name=name,
                type=type,
                description=description,
                required=required,
                enum=enum,
                items=items,
                default=default,
            )
        )
        return self
    
    def build(self) -> ToolSchema:
        """Build the tool schema."""
        return ToolSchema(
            name=self._name,
            description=self._description,
            parameters=self._parameters,
        )
    
    def to_openai_format(self) -> dict[str, Any]:
        """Build and convert to OpenAI format in one step."""
        return self.build().to_openai_format()


def schemas_to_openai_format(schemas: Sequence[ToolSchema]) -> list[dict[str, Any]]:
    """Convert a sequence of ToolSchema objects to OpenAI format."""
    return [schema.to_openai_format() for schema in schemas]

