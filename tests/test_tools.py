"""
Tests for tool schema utilities.

These tests verify the ToolSchemaBuilder and related utilities for creating
tool schemas in OpenAI function calling format.
"""

import pytest
from agent_runtime_core.tools import (
    ToolParameter,
    ToolSchema,
    ToolSchemaBuilder,
    schemas_to_openai_format,
)


class TestToolParameter:
    """Tests for ToolParameter dataclass."""

    def test_basic_parameter(self):
        """Test creating a basic parameter."""
        param = ToolParameter(name="location", type="string", description="City name")

        assert param.name == "location"
        assert param.type == "string"
        assert param.description == "City name"
        assert param.required is False
        assert param.enum is None

    def test_parameter_to_schema(self):
        """Test converting parameter to JSON schema."""
        param = ToolParameter(
            name="location",
            type="string",
            description="City name",
        )
        schema = param.to_schema()

        assert schema == {
            "type": "string",
            "description": "City name",
        }

    def test_parameter_with_enum(self):
        """Test parameter with enum values."""
        param = ToolParameter(
            name="units",
            type="string",
            description="Temperature units",
            enum=["celsius", "fahrenheit"],
        )
        schema = param.to_schema()

        assert schema["enum"] == ["celsius", "fahrenheit"]

    def test_parameter_with_array_items(self):
        """Test array parameter with items schema."""
        param = ToolParameter(
            name="tags",
            type="array",
            description="List of tags",
            items={"type": "string"},
        )
        schema = param.to_schema()

        assert schema["type"] == "array"
        assert schema["items"] == {"type": "string"}

    def test_parameter_with_default(self):
        """Test parameter with default value."""
        param = ToolParameter(
            name="limit",
            type="integer",
            description="Max results",
            default=10,
        )
        schema = param.to_schema()

        assert schema["default"] == 10


class TestToolSchema:
    """Tests for ToolSchema dataclass."""

    def test_basic_schema(self):
        """Test creating a basic tool schema."""
        schema = ToolSchema(
            name="get_weather",
            description="Get the current weather",
            parameters=[
                ToolParameter("location", "string", "City name", required=True),
            ],
        )

        assert schema.name == "get_weather"
        assert schema.description == "Get the current weather"
        assert len(schema.parameters) == 1

    def test_to_openai_format(self):
        """Test converting to OpenAI function calling format."""
        schema = ToolSchema(
            name="get_weather",
            description="Get the current weather for a location",
            parameters=[
                ToolParameter("location", "string", "City name", required=True),
                ToolParameter("units", "string", "Temperature units", enum=["celsius", "fahrenheit"]),
            ],
        )

        openai_format = schema.to_openai_format()

        assert openai_format["type"] == "function"
        assert openai_format["function"]["name"] == "get_weather"
        assert openai_format["function"]["description"] == "Get the current weather for a location"

        params = openai_format["function"]["parameters"]
        assert params["type"] == "object"
        assert "location" in params["properties"]
        assert "units" in params["properties"]
        assert params["required"] == ["location"]
        assert params["properties"]["units"]["enum"] == ["celsius", "fahrenheit"]

    def test_empty_parameters(self):
        """Test schema with no parameters."""
        schema = ToolSchema(name="ping", description="Ping the server")
        openai_format = schema.to_openai_format()

        assert openai_format["function"]["parameters"]["properties"] == {}
        assert openai_format["function"]["parameters"]["required"] == []


class TestToolSchemaBuilder:
    """Tests for ToolSchemaBuilder fluent API."""

    def test_basic_builder(self):
        """Test basic builder usage."""
        schema = (
            ToolSchemaBuilder("get_weather")
            .description("Get the current weather")
            .param("location", "string", "City name", required=True)
            .build()
        )

        assert schema.name == "get_weather"
        assert schema.description == "Get the current weather"
        assert len(schema.parameters) == 1
        assert schema.parameters[0].name == "location"
        assert schema.parameters[0].required is True


    def test_to_openai_format_shortcut(self):
        """Test direct conversion to OpenAI format."""
        openai_format = (
            ToolSchemaBuilder("get_weather")
            .description("Get weather")
            .param("location", "string", "City", required=True)
            .to_openai_format()
        )

        assert openai_format["type"] == "function"
        assert openai_format["function"]["name"] == "get_weather"

    def test_builder_with_all_param_options(self):
        """Test builder with all parameter options."""
        schema = (
            ToolSchemaBuilder("complex_tool")
            .description("A complex tool")
            .param(
                "query",
                "string",
                "Search query",
                required=True,
            )
            .param(
                "filters",
                "array",
                "Filter list",
                items={"type": "object", "properties": {"field": {"type": "string"}}},
            )
            .param(
                "sort",
                "string",
                "Sort order",
                enum=["asc", "desc"],
                default="asc",
            )
            .param(
                "limit",
                "integer",
                "Max results",
                default=10,
            )
            .build()
        )

        assert len(schema.parameters) == 4

        # Check complex array items
        filters_param = schema.parameters[1]
        assert filters_param.items["type"] == "object"

        # Check enum and default
        sort_param = schema.parameters[2]
        assert sort_param.enum == ["asc", "desc"]
        assert sort_param.default == "asc"


class TestSchemasToOpenAIFormat:
    """Tests for schemas_to_openai_format utility."""

    def test_convert_multiple_schemas(self):
        """Test converting multiple schemas to OpenAI format."""
        schemas = [
            ToolSchemaBuilder("tool1").description("First tool").build(),
            ToolSchemaBuilder("tool2").description("Second tool").param("x", "string", "X").build(),
        ]

        openai_formats = schemas_to_openai_format(schemas)

        assert len(openai_formats) == 2
        assert openai_formats[0]["function"]["name"] == "tool1"
        assert openai_formats[1]["function"]["name"] == "tool2"

    def test_convert_empty_list(self):
        """Test converting empty list."""
        assert schemas_to_openai_format([]) == []


class TestRealWorldExamples:
    """Tests with real-world tool schema examples."""

    def test_search_tool(self):
        """Test building a search tool schema."""
        schema = (
            ToolSchemaBuilder("search_documents")
            .description("Search for documents in the knowledge base")
            .param("query", "string", "Search query text", required=True)
            .param("max_results", "integer", "Maximum number of results to return", default=5)
            .param("filters", "object", "Optional filters to apply")
            .build()
        )

        openai_format = schema.to_openai_format()
        assert openai_format["function"]["name"] == "search_documents"
        assert "query" in openai_format["function"]["parameters"]["required"]

    def test_database_query_tool(self):
        """Test building a database query tool schema."""
        schema = (
            ToolSchemaBuilder("query_database")
            .description("Execute a read-only SQL query against the database")
            .param("sql", "string", "SQL SELECT query to execute", required=True)
            .param("params", "array", "Query parameters for prepared statement", items={"type": "string"})
            .build()
        )

        openai_format = schema.to_openai_format()
        params = openai_format["function"]["parameters"]
        assert params["properties"]["params"]["items"] == {"type": "string"}

    def test_api_call_tool(self):
        """Test building an API call tool schema."""
        schema = (
            ToolSchemaBuilder("call_api")
            .description("Make an HTTP API call")
            .param("method", "string", "HTTP method", required=True, enum=["GET", "POST", "PUT", "DELETE"])
            .param("url", "string", "API endpoint URL", required=True)
            .param("headers", "object", "HTTP headers")
            .param("body", "object", "Request body for POST/PUT")
            .build()
        )

        openai_format = schema.to_openai_format()
        assert openai_format["function"]["parameters"]["properties"]["method"]["enum"] == ["GET", "POST", "PUT", "DELETE"]
        assert set(openai_format["function"]["parameters"]["required"]) == {"method", "url"}
