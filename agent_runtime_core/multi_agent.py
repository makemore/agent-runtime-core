"""
Multi-agent support for agent_runtime_core.

This module provides:

1. **SystemContext** - Shared knowledge and configuration for multi-agent systems.
   Allows defining identity, rules, and knowledge that all agents in a system share.

2. **Agent-as-tool pattern** - Allowing agents to invoke other agents as tools.
   This enables router/dispatcher patterns, hierarchical systems, and specialist delegation.

3. **Structured Handback Protocol** - Consistent way for agents to signal completion
   with status, summary, learnings, and recommendations.

4. **Journey Mode** - Allow delegated agents to maintain control across multiple
   user turns without routing back to the entry point each time.

5. **Stuck/Loop Detection** - Detect when conversations are going in circles
   and trigger escalation or alternative approaches.

6. **Fallback Routing** - Automatic return to a fallback agent when agents fail.

Two invocation modes are supported:
- DELEGATE: Sub-agent runs and returns result to parent (parent continues)
- HANDOFF: Control transfers completely to sub-agent (parent exits)

Context passing is configurable:
- FULL: Complete conversation history passed to sub-agent (default)
- SUMMARY: Summarized context + current message
- MESSAGE_ONLY: Only the invocation message

Example - SystemContext:
    from agent_runtime_core.multi_agent import SystemContext, SharedKnowledge

    # Define shared knowledge for a multi-agent system
    system_ctx = SystemContext(
        system_id="sai-system",
        system_name="S'Ai Therapeutic System",
        shared_knowledge=[
            SharedKnowledge(
                key="core_identity",
                title="S'Ai Core Identity",
                content="You are S'Ai, a therapeutic AI assistant...",
                inject_as="system",  # Prepend to system prompt
            ),
            SharedKnowledge(
                key="core_rules",
                title="Core Rules",
                content="1. Never diagnose...",
                inject_as="system",
            ),
        ],
    )

    # Use with a run context
    ctx = InMemoryRunContext(
        run_id=uuid4(),
        system_context=system_ctx,
    )

Example - Agent-as-tool:
    from agent_runtime_core.multi_agent import (
        AgentTool,
        InvocationMode,
        ContextMode,
        invoke_agent,
    )

    # Define a sub-agent as a tool
    billing_agent_tool = AgentTool(
        agent=billing_agent,
        name="billing_specialist",
        description="Handles billing questions, refunds, and payment issues",
        invocation_mode=InvocationMode.DELEGATE,
        context_mode=ContextMode.FULL,
    )

    # Invoke it
    result = await invoke_agent(
        agent_tool=billing_agent_tool,
        message="Customer wants a refund for order #123",
        parent_ctx=ctx,
        conversation_history=messages,
    )

Example - Structured Handback:
    from agent_runtime_core.multi_agent import HandbackResult, HandbackStatus

    # Agent signals completion with structured handback
    handback = HandbackResult(
        status=HandbackStatus.COMPLETED,
        summary="Processed refund for order #123",
        learnings=[
            {"key": "user.refund_history", "value": "Has requested 2 refunds"},
        ],
        recommendation="Consider offering loyalty discount",
    )

Example - Journey Mode:
    from agent_runtime_core.multi_agent import JourneyState, JourneyManager

    # Start a journey (agent maintains control across turns)
    journey = JourneyState(
        agent_key="onboarding_agent",
        started_at=datetime.utcnow(),
        purpose="Complete user onboarding",
        expected_turns=5,
    )

    # Journey manager routes subsequent messages to the journey agent
    manager = JourneyManager(memory_store=store)
    await manager.start_journey(conversation_id, journey)
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Literal, Optional, Protocol, TYPE_CHECKING
from uuid import UUID, uuid4

from agent_runtime_core.interfaces import (
    AgentRuntime,
    Message,
    RunContext,
    RunResult,
    Tool,
    ToolRegistry,
    EventType,
)

if TYPE_CHECKING:
    from agent_runtime_core.contexts import InMemoryRunContext
    from agent_runtime_core.persistence import SharedMemoryStore

logger = logging.getLogger(__name__)


# =============================================================================
# System Context - Shared knowledge for multi-agent systems
# =============================================================================

class InjectMode(str, Enum):
    """
    How shared knowledge is injected into an agent's context.

    SYSTEM: Prepended to the system prompt. Best for identity, rules, and
            instructions that should always be present.

    CONTEXT: Added as a separate context message before the conversation.
             Good for reference material the agent can cite.

    KNOWLEDGE: Added to the agent's knowledge base for RAG retrieval.
               Best for large documents that should be searched.
    """
    SYSTEM = "system"
    CONTEXT = "context"
    KNOWLEDGE = "knowledge"


@dataclass
class SharedKnowledge:
    """
    A piece of shared knowledge that applies to all agents in a system.

    Shared knowledge allows defining content once and having it automatically
    available to all agents in a multi-agent system. This is useful for:
    - Shared identity (who the system is, how to speak)
    - Core rules (non-negotiable principles)
    - Common context (company info, product details)
    - Relational guidelines (how to interact with users)

    Attributes:
        key: Unique identifier for this knowledge item
        title: Human-readable title
        content: The actual knowledge content (markdown supported)
        inject_as: How to inject this into agents (system, context, knowledge)
        priority: Order priority (lower = injected first). Default 0.
        enabled: Whether this knowledge is active
        metadata: Additional metadata

    Example:
        SharedKnowledge(
            key="core_identity",
            title="S'Ai Core Identity",
            content='''
            # Identity
            You are S'Ai, a therapeutic AI assistant. You speak as one unified
            entity, even though you are composed of multiple specialist agents.

            # Voice
            - Warm but professional
            - Curious and exploratory
            - Never prescriptive or diagnostic
            ''',
            inject_as=InjectMode.SYSTEM,
            priority=0,  # Injected first
        )
    """
    key: str
    title: str
    content: str
    inject_as: InjectMode = InjectMode.SYSTEM
    priority: int = 0
    enabled: bool = True
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "key": self.key,
            "title": self.title,
            "content": self.content,
            "inject_as": self.inject_as.value,
            "priority": self.priority,
            "enabled": self.enabled,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SharedKnowledge":
        """Deserialize from dictionary."""
        return cls(
            key=data["key"],
            title=data["title"],
            content=data["content"],
            inject_as=InjectMode(data.get("inject_as", "system")),
            priority=data.get("priority", 0),
            enabled=data.get("enabled", True),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SharedMemoryConfig:
    """
    Configuration for shared memory within a multi-agent system.

    Shared memory allows agents in a system to share learned information
    about users. This is useful for:
    - Cross-agent knowledge sharing (what one agent learns, others can access)
    - User preferences that apply across all agents
    - Accumulated context about the user

    Attributes:
        enabled: Whether shared memory is enabled for this system. Default: True.
        require_auth: Whether authentication is required for shared memory.
                      Should always be True for privacy. Default: True.
        retention_days: How long to retain shared memories. None = forever.
        max_memories_per_user: Maximum memories to store per user. None = unlimited.
        metadata: Additional configuration metadata.

    Example:
        config = SharedMemoryConfig(
            enabled=True,
            require_auth=True,
            retention_days=365,
            max_memories_per_user=1000,
        )
    """
    enabled: bool = True
    require_auth: bool = True
    retention_days: Optional[int] = None
    max_memories_per_user: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "enabled": self.enabled,
            "require_auth": self.require_auth,
            "retention_days": self.retention_days,
            "max_memories_per_user": self.max_memories_per_user,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SharedMemoryConfig":
        """Deserialize from dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            require_auth=data.get("require_auth", True),
            retention_days=data.get("retention_days"),
            max_memories_per_user=data.get("max_memories_per_user"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SystemContext:
    """
    Shared context for a multi-agent system.

    SystemContext holds configuration and knowledge that applies to all agents
    within a system. When an agent runs with a SystemContext, the shared
    knowledge is automatically injected based on each item's inject_as mode.

    This enables the "single source of truth" pattern where identity, rules,
    and common knowledge are defined once and shared across all agents.

    Attributes:
        system_id: Unique identifier for the system
        system_name: Human-readable name
        shared_knowledge: List of SharedKnowledge items
        shared_memory_config: Configuration for cross-agent shared memory
        metadata: Additional system metadata

    Example:
        system_ctx = SystemContext(
            system_id="sai-therapeutic",
            system_name="S'Ai Therapeutic System",
            shared_knowledge=[
                SharedKnowledge(
                    key="identity",
                    title="Core Identity",
                    content="You are S'Ai...",
                    inject_as=InjectMode.SYSTEM,
                ),
                SharedKnowledge(
                    key="rules",
                    title="Core Rules",
                    content="1. Never diagnose...",
                    inject_as=InjectMode.SYSTEM,
                ),
            ],
            shared_memory_config=SharedMemoryConfig(
                enabled=True,
                require_auth=True,
            ),
        )

        # Get content to prepend to system prompt
        system_prefix = system_ctx.get_system_prompt_prefix()
    """
    system_id: str
    system_name: str
    shared_knowledge: list[SharedKnowledge] = field(default_factory=list)
    shared_memory_config: SharedMemoryConfig = field(default_factory=SharedMemoryConfig)
    metadata: dict = field(default_factory=dict)

    def get_system_prompt_prefix(self) -> str:
        """
        Get the content to prepend to an agent's system prompt.

        Returns all enabled SharedKnowledge items with inject_as=SYSTEM,
        sorted by priority, formatted as a single string.
        """
        items = [
            k for k in self.shared_knowledge
            if k.enabled and k.inject_as == InjectMode.SYSTEM
        ]
        items.sort(key=lambda x: x.priority)

        if not items:
            return ""

        parts = []
        for item in items:
            # Add title as header if content doesn't start with one
            if item.content.strip().startswith("#"):
                parts.append(item.content.strip())
            else:
                parts.append(f"## {item.title}\n\n{item.content.strip()}")

        return "\n\n---\n\n".join(parts)

    def get_context_messages(self) -> list[Message]:
        """
        Get messages to add as context before the conversation.

        Returns all enabled SharedKnowledge items with inject_as=CONTEXT
        as system messages.
        """
        items = [
            k for k in self.shared_knowledge
            if k.enabled and k.inject_as == InjectMode.CONTEXT
        ]
        items.sort(key=lambda x: x.priority)

        return [
            {"role": "system", "content": f"[{item.title}]\n\n{item.content}"}
            for item in items
        ]

    def get_knowledge_items(self) -> list[SharedKnowledge]:
        """
        Get SharedKnowledge items that should be added to RAG.

        Returns all enabled items with inject_as=KNOWLEDGE.
        """
        return [
            k for k in self.shared_knowledge
            if k.enabled and k.inject_as == InjectMode.KNOWLEDGE
        ]

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "system_id": self.system_id,
            "system_name": self.system_name,
            "shared_knowledge": [k.to_dict() for k in self.shared_knowledge],
            "shared_memory_config": self.shared_memory_config.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SystemContext":
        """Deserialize from dictionary."""
        shared_memory_data = data.get("shared_memory_config", {})
        return cls(
            system_id=data["system_id"],
            system_name=data["system_name"],
            shared_knowledge=[
                SharedKnowledge.from_dict(k)
                for k in data.get("shared_knowledge", [])
            ],
            shared_memory_config=SharedMemoryConfig.from_dict(shared_memory_data),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Invocation Modes and Context Modes
# =============================================================================

class InvocationMode(str, Enum):
    """
    How the sub-agent is invoked.
    
    DELEGATE: Sub-agent runs, returns result to parent. Parent continues
              its execution with the result. Good for "get me an answer".
              
    HANDOFF: Control transfers completely to sub-agent. Parent's run ends
             and sub-agent takes over the conversation. Good for "transfer
             this customer to billing".
    """
    DELEGATE = "delegate"
    HANDOFF = "handoff"


class ContextMode(str, Enum):
    """
    What context is passed to the sub-agent.

    FULL: Complete conversation history. Sub-agent sees everything the
          parent has seen. Best for sensitive contexts where nothing
          should be forgotten. This is the default.

    SUMMARY: A summary of the conversation + the current message.
             More efficient but may lose nuance.

    MESSAGE_ONLY: Only the invocation message. Clean isolation but
                  sub-agent lacks context.
    """
    FULL = "full"
    SUMMARY = "summary"
    MESSAGE_ONLY = "message_only"


# =============================================================================
# Structured Handback Protocol
# =============================================================================


class HandbackStatus(str, Enum):
    """
    Status codes for structured handback from sub-agents.

    COMPLETED: Task was successfully completed. The agent has done what
               was asked and is ready to hand back control.

    NEEDS_MORE_INFO: Agent needs additional information from the user
                     to complete the task. Includes what info is needed.

    UNCERTAIN: Agent is unsure how to proceed. May need human review
               or escalation to a different agent.

    ESCALATE: Agent explicitly requests escalation to a supervisor
              or different specialist. Includes reason for escalation.

    PARTIAL: Task was partially completed. Some progress was made but
             the full task couldn't be finished.

    FAILED: Task failed and cannot be completed. Includes error details.
    """
    COMPLETED = "completed"
    NEEDS_MORE_INFO = "needs_more_info"
    UNCERTAIN = "uncertain"
    ESCALATE = "escalate"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass
class Learning:
    """
    A piece of information learned during an agent's task.

    Learnings are facts or insights discovered during task execution
    that should be stored in memory for future reference.

    Attributes:
        key: Semantic key for the memory (e.g., "user.preferences.theme")
        value: The learned value
        scope: Memory scope - "conversation", "user", or "system"
        confidence: How confident the agent is (0.0 to 1.0)
        source: Where this learning came from

    Example:
        Learning(
            key="user.communication_style",
            value="Prefers concise responses",
            scope="user",
            confidence=0.8,
            source="inferred from conversation",
        )
    """
    key: str
    value: Any
    scope: str = "user"  # conversation, user, system
    confidence: float = 1.0
    source: str = "agent"

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "value": self.value,
            "scope": self.scope,
            "confidence": self.confidence,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Learning":
        return cls(
            key=data["key"],
            value=data["value"],
            scope=data.get("scope", "user"),
            confidence=data.get("confidence", 1.0),
            source=data.get("source", "agent"),
        )


@dataclass
class HandbackResult:
    """
    Structured result from a sub-agent signaling task completion.

    The handback protocol provides a consistent way for agents to signal
    completion status, summarize what was done, share learnings, and
    make recommendations for next steps.

    Attributes:
        status: The completion status (completed, needs_more_info, etc.)
        summary: Human-readable summary of what was accomplished
        learnings: List of facts/insights to store in memory
        recommendation: Suggested next action for the parent agent
        details: Additional structured details about the task

    Example:
        HandbackResult(
            status=HandbackStatus.COMPLETED,
            summary="Processed refund of $50 for order #123",
            learnings=[
                Learning(
                    key="user.refund_history",
                    value={"count": 2, "total": 75.00},
                    scope="user",
                ),
            ],
            recommendation="Consider offering loyalty discount",
            details={"order_id": "123", "refund_amount": 50.00},
        )
    """
    status: HandbackStatus
    summary: str
    learnings: list[Learning] = field(default_factory=list)
    recommendation: Optional[str] = None
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "summary": self.summary,
            "learnings": [l.to_dict() for l in self.learnings],
            "recommendation": self.recommendation,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HandbackResult":
        return cls(
            status=HandbackStatus(data["status"]),
            summary=data["summary"],
            learnings=[Learning.from_dict(l) for l in data.get("learnings", [])],
            recommendation=data.get("recommendation"),
            details=data.get("details", {}),
        )

    @classmethod
    def parse_from_response(cls, response: str) -> Optional["HandbackResult"]:
        """
        Attempt to parse a HandbackResult from an agent's response.

        Looks for a JSON block with handback data in the response.
        The JSON should be wrapped in ```handback or ```json tags,
        or be a standalone JSON object with a "status" field.

        Returns None if no valid handback is found.
        """
        # Try to find handback JSON block
        patterns = [
            r'```handback\s*\n(.*?)\n```',  # ```handback ... ```
            r'```json\s*\n(\{[^`]*"status"[^`]*\})\n```',  # ```json with status
            r'<handback>(.*?)</handback>',  # <handback>...</handback>
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                try:
                    data = json.loads(match.group(1).strip())
                    if "status" in data:
                        return cls.from_dict(data)
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

        # Try to find a standalone JSON object with status
        try:
            # Look for JSON object pattern
            json_match = re.search(r'\{[^{}]*"status"\s*:\s*"[^"]+?"[^{}]*\}', response)
            if json_match:
                data = json.loads(json_match.group(0))
                if "status" in data and "summary" in data:
                    return cls.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        return None


# =============================================================================
# Stuck/Loop Detection
# =============================================================================


class StuckCondition(str, Enum):
    """
    Types of stuck conditions that can be detected.

    REPEATED_QUESTION: User has asked the same/similar question multiple times
    REPEATED_RESPONSE: Agent has given the same/similar response multiple times
    CIRCULAR_PATTERN: Conversation is going in circles
    NO_PROGRESS: Multiple turns with no meaningful progress
    USER_FRUSTRATION: User is expressing frustration or confusion
    """
    REPEATED_QUESTION = "repeated_question"
    REPEATED_RESPONSE = "repeated_response"
    CIRCULAR_PATTERN = "circular_pattern"
    NO_PROGRESS = "no_progress"
    USER_FRUSTRATION = "user_frustration"


@dataclass
class StuckDetectionResult:
    """
    Result from stuck/loop detection analysis.

    Attributes:
        is_stuck: Whether a stuck condition was detected
        condition: The type of stuck condition (if any)
        confidence: How confident the detection is (0.0 to 1.0)
        evidence: Messages or patterns that triggered detection
        suggestion: Recommended action to break the loop
    """
    is_stuck: bool
    condition: Optional[StuckCondition] = None
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    suggestion: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "is_stuck": self.is_stuck,
            "condition": self.condition.value if self.condition else None,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "suggestion": self.suggestion,
        }


class StuckDetector:
    """
    Detects when conversations are stuck in loops or not making progress.

    Uses simple heuristics to detect:
    - Repeated similar messages (questions or responses)
    - Circular conversation patterns
    - Lack of progress over multiple turns

    Example:
        detector = StuckDetector(
            similarity_threshold=0.8,
            repetition_count=3,
            window_size=10,
        )

        result = detector.analyze(messages)
        if result.is_stuck:
            # Take action - escalate, try different approach, etc.
            print(f"Stuck: {result.condition}, suggestion: {result.suggestion}")
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        repetition_count: int = 3,
        window_size: int = 10,
        frustration_keywords: Optional[list[str]] = None,
    ):
        """
        Initialize the stuck detector.

        Args:
            similarity_threshold: How similar messages must be to count as "same"
            repetition_count: How many repetitions trigger stuck detection
            window_size: How many recent messages to analyze
            frustration_keywords: Words indicating user frustration
        """
        self.similarity_threshold = similarity_threshold
        self.repetition_count = repetition_count
        self.window_size = window_size
        self.frustration_keywords = frustration_keywords or [
            "already said", "told you", "again", "not working",
            "doesn't help", "frustrated", "confused", "same thing",
            "not understanding", "wrong", "still", "keep asking",
        ]

    def analyze(self, messages: list[Message]) -> StuckDetectionResult:
        """
        Analyze conversation messages for stuck conditions.

        Args:
            messages: List of conversation messages

        Returns:
            StuckDetectionResult with detection details
        """
        if len(messages) < self.repetition_count:
            return StuckDetectionResult(is_stuck=False)

        # Get recent messages within window
        recent = messages[-self.window_size:]

        # Separate user and assistant messages
        user_messages = [m for m in recent if m.get("role") == "user"]
        assistant_messages = [m for m in recent if m.get("role") == "assistant"]

        # Check for repeated user questions
        result = self._check_repeated_messages(
            user_messages,
            StuckCondition.REPEATED_QUESTION,
            "User keeps asking similar questions"
        )
        if result.is_stuck:
            result.suggestion = "Try a different approach or ask clarifying questions"
            return result

        # Check for repeated assistant responses
        result = self._check_repeated_messages(
            assistant_messages,
            StuckCondition.REPEATED_RESPONSE,
            "Agent keeps giving similar responses"
        )
        if result.is_stuck:
            result.suggestion = "Acknowledge the repetition and try a new approach"
            return result

        # Check for user frustration
        result = self._check_frustration(user_messages)
        if result.is_stuck:
            return result

        # Check for circular patterns (A -> B -> A -> B)
        result = self._check_circular_pattern(recent)
        if result.is_stuck:
            return result

        return StuckDetectionResult(is_stuck=False)

    def _check_repeated_messages(
        self,
        messages: list[Message],
        condition: StuckCondition,
        description: str,
    ) -> StuckDetectionResult:
        """Check for repeated similar messages."""
        if len(messages) < self.repetition_count:
            return StuckDetectionResult(is_stuck=False)

        contents = [self._normalize(m.get("content", "")) for m in messages]

        # Count similar messages
        for i, content in enumerate(contents):
            if not content:
                continue
            similar_count = 1
            similar_msgs = [content[:100]]

            for j, other in enumerate(contents):
                if i != j and other and self._is_similar(content, other):
                    similar_count += 1
                    similar_msgs.append(other[:100])

            if similar_count >= self.repetition_count:
                return StuckDetectionResult(
                    is_stuck=True,
                    condition=condition,
                    confidence=min(0.5 + (similar_count - self.repetition_count) * 0.1, 1.0),
                    evidence=similar_msgs[:3],
                )

        return StuckDetectionResult(is_stuck=False)

    def _check_frustration(self, user_messages: list[Message]) -> StuckDetectionResult:
        """Check for signs of user frustration."""
        if not user_messages:
            return StuckDetectionResult(is_stuck=False)

        # Check recent user messages for frustration keywords
        recent_user = user_messages[-3:] if len(user_messages) >= 3 else user_messages
        frustration_evidence = []

        for msg in recent_user:
            content = msg.get("content", "").lower()
            for keyword in self.frustration_keywords:
                if keyword in content:
                    frustration_evidence.append(f"'{keyword}' in: {content[:50]}...")

        if len(frustration_evidence) >= 2:
            return StuckDetectionResult(
                is_stuck=True,
                condition=StuckCondition.USER_FRUSTRATION,
                confidence=min(0.6 + len(frustration_evidence) * 0.1, 1.0),
                evidence=frustration_evidence,
                suggestion="Acknowledge frustration, apologize, and try a completely different approach",
            )

        return StuckDetectionResult(is_stuck=False)

    def _check_circular_pattern(self, messages: list[Message]) -> StuckDetectionResult:
        """Check for circular conversation patterns."""
        if len(messages) < 6:
            return StuckDetectionResult(is_stuck=False)

        # Look for A-B-A-B pattern in last 6 messages
        contents = [self._normalize(m.get("content", "")) for m in messages[-6:]]

        # Check if messages 0,2,4 are similar AND messages 1,3,5 are similar
        if (self._is_similar(contents[0], contents[2]) and
            self._is_similar(contents[2], contents[4]) and
            self._is_similar(contents[1], contents[3]) and
            self._is_similar(contents[3], contents[5])):
            return StuckDetectionResult(
                is_stuck=True,
                condition=StuckCondition.CIRCULAR_PATTERN,
                confidence=0.9,
                evidence=[contents[0][:50], contents[1][:50]],
                suggestion="Break the pattern by introducing new information or escalating",
            )

        return StuckDetectionResult(is_stuck=False)

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""
        # Lowercase, remove extra whitespace, remove punctuation
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def _is_similar(self, text1: str, text2: str) -> bool:
        """
        Check if two texts are similar using simple heuristics.

        Uses a combination of:
        - Exact match after normalization
        - Word overlap ratio
        """
        if not text1 or not text2:
            return False

        # Exact match
        if text1 == text2:
            return True

        # Word overlap
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return False

        intersection = words1 & words2
        union = words1 | words2

        jaccard = len(intersection) / len(union)
        return jaccard >= self.similarity_threshold


# =============================================================================
# Journey Mode
# =============================================================================


class JourneyEndReason(str, Enum):
    """
    Reasons why a journey ended.

    COMPLETED: Agent signaled task completion
    TOPIC_CHANGE: User changed to a different topic
    STUCK: Conversation got stuck in a loop
    TIMEOUT: Journey exceeded maximum turns
    USER_EXIT: User explicitly asked to exit/cancel
    ESCALATION: Agent requested escalation
    ERROR: An error occurred during the journey
    """
    COMPLETED = "completed"
    TOPIC_CHANGE = "topic_change"
    STUCK = "stuck"
    TIMEOUT = "timeout"
    USER_EXIT = "user_exit"
    ESCALATION = "escalation"
    ERROR = "error"


@dataclass
class JourneyState:
    """
    State of an active journey (multi-turn agent control).

    A journey allows a delegated agent to maintain control across multiple
    user turns without routing back to the entry point (e.g., Triage) each time.

    Attributes:
        journey_id: Unique identifier for this journey
        agent_key: The agent that owns this journey
        started_at: When the journey started
        purpose: What the journey is trying to accomplish
        expected_turns: Estimated number of turns (for progress tracking)
        current_turn: Current turn number
        max_turns: Maximum allowed turns before timeout
        context: Additional context for the journey
        checkpoints: List of checkpoint summaries

    Example:
        journey = JourneyState(
            agent_key="onboarding_agent",
            purpose="Complete new user onboarding",
            expected_turns=5,
            max_turns=20,
        )
    """
    journey_id: UUID = field(default_factory=uuid4)
    agent_key: str = ""
    started_at: datetime = field(default_factory=datetime.utcnow)
    purpose: str = ""
    expected_turns: int = 5
    current_turn: int = 0
    max_turns: int = 50
    context: dict = field(default_factory=dict)
    checkpoints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "journey_id": str(self.journey_id),
            "agent_key": self.agent_key,
            "started_at": self.started_at.isoformat(),
            "purpose": self.purpose,
            "expected_turns": self.expected_turns,
            "current_turn": self.current_turn,
            "max_turns": self.max_turns,
            "context": self.context,
            "checkpoints": self.checkpoints,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "JourneyState":
        return cls(
            journey_id=UUID(data["journey_id"]) if isinstance(data.get("journey_id"), str) else data.get("journey_id", uuid4()),
            agent_key=data.get("agent_key", ""),
            started_at=datetime.fromisoformat(data["started_at"]) if isinstance(data.get("started_at"), str) else data.get("started_at", datetime.utcnow()),
            purpose=data.get("purpose", ""),
            expected_turns=data.get("expected_turns", 5),
            current_turn=data.get("current_turn", 0),
            max_turns=data.get("max_turns", 50),
            context=data.get("context", {}),
            checkpoints=data.get("checkpoints", []),
        )

    def increment_turn(self) -> None:
        """Increment the turn counter."""
        self.current_turn += 1

    def add_checkpoint(self, summary: str) -> None:
        """Add a checkpoint summary."""
        self.checkpoints.append(summary)

    def is_timeout(self) -> bool:
        """Check if journey has exceeded max turns."""
        return self.current_turn >= self.max_turns


@dataclass
class JourneyEndResult:
    """
    Result when a journey ends.

    Attributes:
        reason: Why the journey ended
        handback: The structured handback from the agent (if any)
        summary: Summary of what was accomplished
        next_agent: Suggested next agent to route to (if any)
    """
    reason: JourneyEndReason
    handback: Optional[HandbackResult] = None
    summary: str = ""
    next_agent: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "reason": self.reason.value,
            "handback": self.handback.to_dict() if self.handback else None,
            "summary": self.summary,
            "next_agent": self.next_agent,
        }


# =============================================================================
# Fallback Routing
# =============================================================================


@dataclass
class FallbackConfig:
    """
    Configuration for fallback routing when agents fail.

    Attributes:
        fallback_agent_key: The agent to route to on failure (e.g., "triage")
        max_retries: Maximum retries before falling back
        retry_delay_seconds: Delay between retries
        fallback_message: Message to include when falling back
        capture_error: Whether to include error details in fallback

    Example:
        config = FallbackConfig(
            fallback_agent_key="triage",
            max_retries=1,
            fallback_message="I encountered an issue. Let me get you some help.",
        )
    """
    fallback_agent_key: str = "triage"
    max_retries: int = 1
    retry_delay_seconds: float = 0.5
    fallback_message: str = "I apologize, but I encountered an issue. Let me connect you with someone who can help."
    capture_error: bool = True

    def to_dict(self) -> dict:
        return {
            "fallback_agent_key": self.fallback_agent_key,
            "max_retries": self.max_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
            "fallback_message": self.fallback_message,
            "capture_error": self.capture_error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FallbackConfig":
        return cls(
            fallback_agent_key=data.get("fallback_agent_key", "triage"),
            max_retries=data.get("max_retries", 1),
            retry_delay_seconds=data.get("retry_delay_seconds", 0.5),
            fallback_message=data.get("fallback_message", cls.fallback_message),
            capture_error=data.get("capture_error", True),
        )


# =============================================================================
# Agent Tool and Invocation
# =============================================================================


@dataclass
class AgentTool:
    """
    Wraps an agent to be used as a tool by another agent.
    
    This is the core abstraction for multi-agent systems. Any agent can
    be wrapped as a tool and added to another agent's tool registry.
    
    Attributes:
        agent: The agent runtime to invoke
        name: Tool name (how the parent agent calls it)
        description: When to use this agent (shown to parent LLM)
        invocation_mode: DELEGATE or HANDOFF
        context_mode: How much context to pass (FULL, SUMMARY, MESSAGE_ONLY)
        max_turns: Optional limit on sub-agent turns (for DELEGATE mode)
        input_schema: Optional custom input schema (defaults to message + context)
        metadata: Additional metadata for the tool
    """
    agent: AgentRuntime
    name: str
    description: str
    invocation_mode: InvocationMode = InvocationMode.DELEGATE
    context_mode: ContextMode = ContextMode.FULL
    max_turns: Optional[int] = None
    input_schema: Optional[dict] = None
    metadata: dict = field(default_factory=dict)
    
    def to_tool_schema(self) -> dict:
        """
        Generate the OpenAI-format tool schema for this agent-tool.
        
        The schema allows the parent agent to invoke this sub-agent
        with a message and optional context override.
        """
        if self.input_schema:
            parameters = self.input_schema
        else:
            # Default schema: message + optional context
            parameters = {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message or task to send to this agent",
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional additional context to include",
                    },
                },
                "required": ["message"],
            }
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }


@dataclass
class AgentInvocationResult:
    """
    Result from invoking a sub-agent.

    Attributes:
        response: The sub-agent's final response text
        messages: All messages from the sub-agent's run
        handoff: True if this was a handoff (parent should exit)
        run_result: The full RunResult from the sub-agent
        sub_agent_key: The key of the agent that was invoked
        handback: Structured handback result (if agent provided one)
    """
    response: str
    messages: list[Message]
    handoff: bool
    run_result: RunResult
    sub_agent_key: str
    handback: Optional[HandbackResult] = None


class SubAgentContext:
    """
    RunContext implementation for sub-agent invocations.
    
    Wraps the parent context but with modified input messages
    based on the context mode.
    """
    
    def __init__(
        self,
        parent_ctx: RunContext,
        input_messages: list[Message],
        sub_run_id: Optional[UUID] = None,
    ):
        self._parent_ctx = parent_ctx
        self._input_messages = input_messages
        self._run_id = sub_run_id or uuid4()
        self._state: Optional[dict] = None
    
    @property
    def run_id(self) -> UUID:
        return self._run_id
    
    @property
    def conversation_id(self) -> Optional[UUID]:
        # Sub-agent shares the same conversation
        return self._parent_ctx.conversation_id
    
    @property
    def input_messages(self) -> list[Message]:
        return self._input_messages
    
    @property
    def params(self) -> dict:
        return self._parent_ctx.params
    
    @property
    def metadata(self) -> dict:
        # Add sub-agent metadata
        meta = dict(self._parent_ctx.metadata)
        meta["parent_run_id"] = str(self._parent_ctx.run_id)
        meta["is_sub_agent"] = True
        return meta
    
    @property
    def tool_registry(self) -> ToolRegistry:
        # Sub-agent uses its own tools, not parent's
        # This is set by the agent itself
        return ToolRegistry()
    
    async def emit(self, event_type: EventType | str, payload: dict) -> None:
        """Emit events through parent context with sub-agent tagging."""
        # Tag the event as coming from a sub-agent
        tagged_payload = dict(payload)
        tagged_payload["sub_agent_run_id"] = str(self._run_id)
        tagged_payload["parent_run_id"] = str(self._parent_ctx.run_id)
        await self._parent_ctx.emit(event_type, tagged_payload)
    
    async def emit_user_message(self, content: str) -> None:
        """Emit a user-visible message."""
        await self.emit(EventType.ASSISTANT_MESSAGE, {
            "content": content,
            "role": "assistant",
        })
    
    async def emit_error(self, error: str, details: dict = None) -> None:
        """Emit an error event."""
        await self.emit(EventType.ERROR, {
            "error": error,
            "details": details or {},
        })
    
    async def checkpoint(self, state: dict) -> None:
        """Save state checkpoint."""
        self._state = state
        # Could also delegate to parent for persistence
    
    async def get_state(self) -> Optional[dict]:
        """Get last checkpointed state."""
        return self._state
    
    def cancelled(self) -> bool:
        """Check if parent has been cancelled."""
        return self._parent_ctx.cancelled()


def build_sub_agent_messages(
    context_mode: ContextMode,
    message: str,
    conversation_history: list[Message],
    additional_context: Optional[str] = None,
) -> list[Message]:
    """
    Build the input messages for a sub-agent based on context mode.
    
    Args:
        context_mode: How much context to include
        message: The invocation message from the parent
        conversation_history: Full conversation history from parent
        additional_context: Optional extra context string
        
    Returns:
        List of messages to pass to the sub-agent
    """
    if context_mode == ContextMode.FULL:
        # Include full history + new message
        messages = list(conversation_history)
        
        # Build the user message
        user_content = message
        if additional_context:
            user_content = f"{additional_context}\n\n{message}"
        
        messages.append({
            "role": "user",
            "content": user_content,
        })
        return messages
    
    elif context_mode == ContextMode.SUMMARY:
        # TODO: Implement summarization
        # For now, fall back to including last few messages
        recent_messages = conversation_history[-5:] if conversation_history else []
        
        user_content = message
        if additional_context:
            user_content = f"{additional_context}\n\n{message}"
        
        return list(recent_messages) + [{
            "role": "user",
            "content": user_content,
        }]
    
    else:  # MESSAGE_ONLY
        user_content = message
        if additional_context:
            user_content = f"{additional_context}\n\n{message}"
        
        return [{
            "role": "user",
            "content": user_content,
        }]


async def invoke_agent(
    agent_tool: AgentTool,
    message: str,
    parent_ctx: RunContext,
    conversation_history: Optional[list[Message]] = None,
    additional_context: Optional[str] = None,
) -> AgentInvocationResult:
    """
    Invoke a sub-agent as a tool.
    
    This is the main entry point for agent-to-agent invocation.
    
    Args:
        agent_tool: The AgentTool wrapping the sub-agent
        message: The message/task to send to the sub-agent
        parent_ctx: The parent agent's run context
        conversation_history: Full conversation history (for FULL context mode)
        additional_context: Optional extra context to include
        
    Returns:
        AgentInvocationResult with the sub-agent's response
        
    Example:
        result = await invoke_agent(
            agent_tool=billing_specialist,
            message="Process refund for order #123",
            parent_ctx=ctx,
            conversation_history=messages,
        )
        
        if result.handoff:
            # Sub-agent took over, return its result
            return RunResult(
                final_output=result.run_result.final_output,
                final_messages=result.messages,
            )
        else:
            # Got a response, continue parent execution
            print(f"Billing says: {result.response}")
    """
    logger.info(
        f"Invoking sub-agent '{agent_tool.name}' "
        f"(mode={agent_tool.invocation_mode.value}, "
        f"context={agent_tool.context_mode.value})"
    )
    
    # Build messages based on context mode
    history = conversation_history or []
    sub_messages = build_sub_agent_messages(
        context_mode=agent_tool.context_mode,
        message=message,
        conversation_history=history,
        additional_context=additional_context,
    )
    
    # Create sub-agent context
    sub_ctx = SubAgentContext(
        parent_ctx=parent_ctx,
        input_messages=sub_messages,
    )
    
    # Emit event for sub-agent invocation (tool call format)
    await parent_ctx.emit(EventType.TOOL_CALL, {
        "name": agent_tool.name,
        "arguments": {"message": message, "context": additional_context},
        "is_agent_tool": True,
        "sub_agent_key": agent_tool.agent.key,
        "invocation_mode": agent_tool.invocation_mode.value,
    })

    # Also emit a custom sub_agent.start event for UI display
    # Get agent name from the agent if available
    agent_name = getattr(agent_tool.agent, 'name', None) or agent_tool.name
    await parent_ctx.emit("sub_agent.start", {
        "sub_agent_key": agent_tool.agent.key,
        "agent_name": agent_name,
        "tool_name": agent_tool.name,
        "invocation_mode": agent_tool.invocation_mode.value,
        "context_mode": agent_tool.context_mode.value,
    })

    # Run the sub-agent
    try:
        run_result = await agent_tool.agent.run(sub_ctx)
    except Exception as e:
        logger.exception(f"Sub-agent '{agent_tool.name}' failed")
        # Emit error event
        await parent_ctx.emit(EventType.TOOL_RESULT, {
            "name": agent_tool.name,
            "is_agent_tool": True,
            "error": str(e),
        })
        # Emit sub_agent.end with error
        await parent_ctx.emit("sub_agent.end", {
            "sub_agent_key": agent_tool.agent.key,
            "agent_name": agent_name,
            "tool_name": agent_tool.name,
            "error": str(e),
        })
        raise

    # Extract response
    response = run_result.final_output.get("response", "")
    if not response and run_result.final_messages:
        # Try to get from last assistant message
        for msg in reversed(run_result.final_messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                response = msg["content"]
                break

    # Try to parse structured handback from response
    handback = None
    if response:
        handback = HandbackResult.parse_from_response(response)
        if handback:
            logger.info(
                f"Sub-agent '{agent_tool.name}' returned structured handback: "
                f"status={handback.status.value}"
            )

    # Emit result event (tool result format)
    await parent_ctx.emit(EventType.TOOL_RESULT, {
        "name": agent_tool.name,
        "is_agent_tool": True,
        "sub_agent_key": agent_tool.agent.key,
        "response": response[:500] if response else "",  # Truncate for event
        "handoff": agent_tool.invocation_mode == InvocationMode.HANDOFF,
        "handback_status": handback.status.value if handback else None,
    })

    # Also emit a custom sub_agent.end event for UI display
    await parent_ctx.emit("sub_agent.end", {
        "sub_agent_key": agent_tool.agent.key,
        "agent_name": agent_name,
        "tool_name": agent_tool.name,
        "success": True,
        "handoff": agent_tool.invocation_mode == InvocationMode.HANDOFF,
        "handback": handback.to_dict() if handback else None,
    })

    return AgentInvocationResult(
        response=response,
        messages=run_result.final_messages,
        handoff=agent_tool.invocation_mode == InvocationMode.HANDOFF,
        run_result=run_result,
        sub_agent_key=agent_tool.agent.key,
        handback=handback,
    )


def create_agent_tool_handler(
    agent_tool: AgentTool,
    get_conversation_history: Callable[[], list[Message]],
    parent_ctx: RunContext,
) -> Callable:
    """
    Create a tool handler function for an AgentTool.
    
    This creates a handler that can be registered in a ToolRegistry,
    allowing the agent-tool to be called like any other tool.
    
    Args:
        agent_tool: The AgentTool to create a handler for
        get_conversation_history: Function that returns current conversation
        parent_ctx: The parent agent's context
        
    Returns:
        Async handler function compatible with ToolRegistry
        
    Example:
        handler = create_agent_tool_handler(
            billing_agent_tool,
            lambda: current_messages,
            ctx,
        )
        
        registry.register(Tool(
            name=billing_agent_tool.name,
            description=billing_agent_tool.description,
            parameters=billing_agent_tool.to_tool_schema()["function"]["parameters"],
            handler=handler,
        ))
    """
    async def handler(message: str, context: Optional[str] = None) -> dict:
        result = await invoke_agent(
            agent_tool=agent_tool,
            message=message,
            parent_ctx=parent_ctx,
            conversation_history=get_conversation_history(),
            additional_context=context,
        )
        
        if result.handoff:
            # Signal to the parent that this is a handoff
            return {
                "handoff": True,
                "response": result.response,
                "sub_agent": result.sub_agent_key,
                "final_output": result.run_result.final_output,
            }
        else:
            return {
                "response": result.response,
                "sub_agent": result.sub_agent_key,
            }
    
    return handler


def register_agent_tools(
    registry: ToolRegistry,
    agent_tools: list[AgentTool],
    get_conversation_history: Callable[[], list[Message]],
    parent_ctx: RunContext,
) -> None:
    """
    Register multiple agent-tools in a ToolRegistry.
    
    Convenience function to add several sub-agents as tools.
    
    Args:
        registry: The ToolRegistry to add tools to
        agent_tools: List of AgentTools to register
        get_conversation_history: Function returning current conversation
        parent_ctx: Parent agent's context
        
    Example:
        register_agent_tools(
            registry=self.tools,
            agent_tools=[billing_agent_tool, support_agent_tool],
            get_conversation_history=lambda: self._messages,
            parent_ctx=ctx,
        )
    """
    for agent_tool in agent_tools:
        handler = create_agent_tool_handler(
            agent_tool=agent_tool,
            get_conversation_history=get_conversation_history,
            parent_ctx=parent_ctx,
        )

        schema = agent_tool.to_tool_schema()

        registry.register(Tool(
            name=agent_tool.name,
            description=agent_tool.description,
            parameters=schema["function"]["parameters"],
            handler=handler,
            metadata={
                "is_agent_tool": True,
                "sub_agent_key": agent_tool.agent.key,
                "invocation_mode": agent_tool.invocation_mode.value,
                "context_mode": agent_tool.context_mode.value,
            },
        ))


# =============================================================================
# Fallback-Enabled Invocation
# =============================================================================


async def invoke_agent_with_fallback(
    agent_tool: AgentTool,
    message: str,
    parent_ctx: RunContext,
    conversation_history: Optional[list[Message]] = None,
    additional_context: Optional[str] = None,
    fallback_config: Optional[FallbackConfig] = None,
    fallback_agent_tool: Optional[AgentTool] = None,
) -> AgentInvocationResult:
    """
    Invoke a sub-agent with automatic fallback on failure.

    If the primary agent fails, automatically routes to a fallback agent
    (typically Triage) to ensure the user is never left without a response.

    Args:
        agent_tool: The primary AgentTool to invoke
        message: The message/task to send
        parent_ctx: The parent agent's run context
        conversation_history: Full conversation history
        additional_context: Optional extra context
        fallback_config: Configuration for fallback behavior
        fallback_agent_tool: The fallback agent to use on failure

    Returns:
        AgentInvocationResult from either primary or fallback agent

    Example:
        result = await invoke_agent_with_fallback(
            agent_tool=billing_specialist,
            message="Process refund",
            parent_ctx=ctx,
            fallback_config=FallbackConfig(fallback_agent_key="triage"),
            fallback_agent_tool=triage_agent_tool,
        )
    """
    import asyncio

    config = fallback_config or FallbackConfig()
    last_error: Optional[Exception] = None

    # Try primary agent with retries
    for attempt in range(config.max_retries + 1):
        try:
            result = await invoke_agent(
                agent_tool=agent_tool,
                message=message,
                parent_ctx=parent_ctx,
                conversation_history=conversation_history,
                additional_context=additional_context,
            )
            return result
        except Exception as e:
            last_error = e
            logger.warning(
                f"Agent '{agent_tool.name}' failed (attempt {attempt + 1}/{config.max_retries + 1}): {e}"
            )
            if attempt < config.max_retries:
                await asyncio.sleep(config.retry_delay_seconds)

    # Primary agent failed, try fallback
    if fallback_agent_tool:
        logger.info(
            f"Falling back to '{fallback_agent_tool.name}' after "
            f"'{agent_tool.name}' failed"
        )

        # Build fallback context
        fallback_context = config.fallback_message
        if config.capture_error and last_error:
            fallback_context += f"\n\nPrevious agent error: {str(last_error)}"

        # Emit fallback event
        await parent_ctx.emit("agent.fallback", {
            "failed_agent": agent_tool.name,
            "fallback_agent": fallback_agent_tool.name,
            "error": str(last_error) if last_error else None,
        })

        try:
            return await invoke_agent(
                agent_tool=fallback_agent_tool,
                message=message,
                parent_ctx=parent_ctx,
                conversation_history=conversation_history,
                additional_context=fallback_context,
            )
        except Exception as fallback_error:
            logger.exception(f"Fallback agent '{fallback_agent_tool.name}' also failed")
            # Re-raise the original error
            raise last_error or fallback_error

    # No fallback available, re-raise the error
    if last_error:
        raise last_error
    raise RuntimeError(f"Agent '{agent_tool.name}' failed with no fallback available")


# =============================================================================
# Journey Manager
# =============================================================================


# Memory key for storing active journey state
JOURNEY_STATE_KEY = "system.active_journey"


class JourneyManager:
    """
    Manages journey mode for multi-turn agent control.

    A journey allows a delegated agent to maintain control across multiple
    user turns without routing back to the entry point (e.g., Triage) each time.

    The JourneyManager:
    - Stores active journey state in conversation-scoped memory
    - Routes subsequent messages to the journey agent
    - Detects journey end conditions (completion, topic change, stuck, timeout)
    - Handles handback to the entry agent with summary

    Example:
        manager = JourneyManager(memory_store=store, conversation_id=conv_id)

        # Start a journey
        await manager.start_journey(JourneyState(
            agent_key="onboarding_agent",
            purpose="Complete user onboarding",
        ))

        # Check if there's an active journey
        journey = await manager.get_active_journey()
        if journey:
            # Route to journey agent instead of Triage
            result = await invoke_agent(journey_agent_tool, message, ctx)

            # Check for journey end
            end_result = await manager.check_journey_end(result, messages)
            if end_result:
                await manager.end_journey(end_result.reason)
    """

    def __init__(
        self,
        memory_store: Optional["SharedMemoryStore"] = None,
        conversation_id: Optional[UUID] = None,
        stuck_detector: Optional[StuckDetector] = None,
        topic_change_threshold: float = 0.3,
    ):
        """
        Initialize the journey manager.

        Args:
            memory_store: Store for persisting journey state
            conversation_id: The conversation this manager is for
            stuck_detector: Detector for stuck/loop conditions
            topic_change_threshold: Similarity threshold for topic change detection
        """
        self.memory_store = memory_store
        self.conversation_id = conversation_id
        self.stuck_detector = stuck_detector or StuckDetector()
        self.topic_change_threshold = topic_change_threshold
        self._journey_state: Optional[JourneyState] = None

    async def get_active_journey(self) -> Optional[JourneyState]:
        """
        Get the currently active journey for this conversation.

        Returns:
            JourneyState if there's an active journey, None otherwise
        """
        # Check in-memory cache first
        if self._journey_state:
            return self._journey_state

        # Try to load from memory store
        if self.memory_store and self.conversation_id:
            try:
                item = await self.memory_store.get(
                    JOURNEY_STATE_KEY,
                    scope="conversation",
                    conversation_id=self.conversation_id,
                )
                if item and item.value:
                    self._journey_state = JourneyState.from_dict(item.value)
                    return self._journey_state
            except Exception as e:
                logger.warning(f"Failed to load journey state: {e}")

        return None

    async def start_journey(self, journey: JourneyState) -> None:
        """
        Start a new journey.

        Args:
            journey: The journey state to start
        """
        self._journey_state = journey

        # Persist to memory store
        if self.memory_store and self.conversation_id:
            try:
                await self.memory_store.set(
                    JOURNEY_STATE_KEY,
                    journey.to_dict(),
                    scope="conversation",
                    conversation_id=self.conversation_id,
                    source="journey_manager",
                )
            except Exception as e:
                logger.warning(f"Failed to persist journey state: {e}")

        logger.info(
            f"Started journey '{journey.journey_id}' with agent '{journey.agent_key}': "
            f"{journey.purpose}"
        )

    async def update_journey(self, checkpoint: Optional[str] = None) -> None:
        """
        Update the journey state (increment turn, add checkpoint).

        Args:
            checkpoint: Optional checkpoint summary to add
        """
        if not self._journey_state:
            return

        self._journey_state.increment_turn()
        if checkpoint:
            self._journey_state.add_checkpoint(checkpoint)

        # Persist updated state
        if self.memory_store and self.conversation_id:
            try:
                await self.memory_store.set(
                    JOURNEY_STATE_KEY,
                    self._journey_state.to_dict(),
                    scope="conversation",
                    conversation_id=self.conversation_id,
                    source="journey_manager",
                )
            except Exception as e:
                logger.warning(f"Failed to update journey state: {e}")

    async def end_journey(self, reason: JourneyEndReason) -> JourneyEndResult:
        """
        End the current journey.

        Args:
            reason: Why the journey is ending

        Returns:
            JourneyEndResult with summary and handback info
        """
        journey = self._journey_state
        if not journey:
            return JourneyEndResult(
                reason=reason,
                summary="No active journey",
            )

        # Build summary from checkpoints
        summary = f"Journey '{journey.purpose}' ended after {journey.current_turn} turns."
        if journey.checkpoints:
            summary += f" Checkpoints: {'; '.join(journey.checkpoints[-3:])}"

        result = JourneyEndResult(
            reason=reason,
            summary=summary,
        )

        # Clear journey state
        self._journey_state = None

        # Remove from memory store
        if self.memory_store and self.conversation_id:
            try:
                await self.memory_store.delete(
                    JOURNEY_STATE_KEY,
                    scope="conversation",
                    conversation_id=self.conversation_id,
                )
            except Exception as e:
                logger.warning(f"Failed to clear journey state: {e}")

        logger.info(
            f"Ended journey '{journey.journey_id}' (reason={reason.value}): {summary}"
        )

        return result

    def check_journey_end(
        self,
        invocation_result: AgentInvocationResult,
        messages: list[Message],
        user_message: Optional[str] = None,
    ) -> Optional[JourneyEndResult]:
        """
        Check if the journey should end based on the latest interaction.

        Args:
            invocation_result: Result from the journey agent
            messages: Full conversation history
            user_message: The latest user message (for topic change detection)

        Returns:
            JourneyEndResult if journey should end, None otherwise
        """
        journey = self._journey_state
        if not journey:
            return None

        # Check for structured handback completion
        if invocation_result.handback:
            handback = invocation_result.handback
            if handback.status == HandbackStatus.COMPLETED:
                return JourneyEndResult(
                    reason=JourneyEndReason.COMPLETED,
                    handback=handback,
                    summary=handback.summary,
                )
            elif handback.status == HandbackStatus.ESCALATE:
                return JourneyEndResult(
                    reason=JourneyEndReason.ESCALATION,
                    handback=handback,
                    summary=handback.summary,
                    next_agent=handback.recommendation,
                )

        # Check for timeout
        if journey.is_timeout():
            return JourneyEndResult(
                reason=JourneyEndReason.TIMEOUT,
                summary=f"Journey exceeded maximum turns ({journey.max_turns})",
            )

        # Check for stuck condition
        stuck_result = self.stuck_detector.analyze(messages)
        if stuck_result.is_stuck and stuck_result.confidence >= 0.7:
            return JourneyEndResult(
                reason=JourneyEndReason.STUCK,
                summary=f"Conversation stuck: {stuck_result.condition.value if stuck_result.condition else 'unknown'}",
            )

        # Check for user exit keywords
        if user_message:
            exit_keywords = ["cancel", "stop", "exit", "quit", "nevermind", "never mind", "forget it"]
            user_lower = user_message.lower()
            for keyword in exit_keywords:
                if keyword in user_lower:
                    return JourneyEndResult(
                        reason=JourneyEndReason.USER_EXIT,
                        summary=f"User requested exit: '{user_message[:50]}'",
                    )

        # Check for topic change (simple heuristic)
        if user_message and journey.purpose:
            # Very basic topic change detection - could be enhanced with embeddings
            purpose_words = set(journey.purpose.lower().split())
            message_words = set(user_message.lower().split())
            overlap = len(purpose_words & message_words) / max(len(purpose_words), 1)

            # If very low overlap and message is a question about something else
            if overlap < self.topic_change_threshold and "?" in user_message:
                # Check if it seems like a new topic
                new_topic_indicators = ["can you", "what about", "how do i", "tell me about", "help me with"]
                if any(indicator in user_message.lower() for indicator in new_topic_indicators):
                    return JourneyEndResult(
                        reason=JourneyEndReason.TOPIC_CHANGE,
                        summary=f"User changed topic: '{user_message[:50]}'",
                    )

        return None

