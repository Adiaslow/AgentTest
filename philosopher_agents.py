import asyncio
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
import httpx
import threading
import queue
import time
import uuid
import random


from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

from rich.live import Live
from rich.layout import Layout
from rich.columns import Columns
from rich.align import Align
from rich.box import ROUNDED, DOUBLE
from rich.style import Style
from rich.syntax import Syntax
from rich.emoji import Emoji
from rich import print as rprint
from rich.logging import RichHandler
import logging

# Initialize Rich console
console = Console()

# Configure Rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)

# Create a logger for the application
logger = logging.getLogger("philosophical_discussion")

# TTS imports
try:
    import pyttsx3

    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    logger.warning("⚠️  pyttsx3 not available. Install with: pip install pyttsx3")

try:
    import edge_tts
    import pygame
    import io

    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False


class MessageType(str, Enum):
    DISCUSSION = "discussion"
    QUESTION = "question"
    RESPONSE = "response"
    SUMMARY = "summary"
    PRIVATE_THOUGHT = "private_thought"


class AgentRole(str, Enum):
    ANALYST = "analyst"
    CRITIC = "critic"
    OPTIMIST = "optimist"
    MODERATOR = "moderator"
    EXPERT = "domain_expert"


class TTSEngine(str, Enum):
    PYTTSX3 = "pyttsx3"
    EDGE_TTS = "edge_tts"
    DISABLED = "disabled"


class TTSConfig(BaseModel):
    enabled: bool = True
    engine: TTSEngine = TTSEngine.PYTTSX3
    rate: int = Field(default=180, ge=50, le=400)  # Words per minute
    volume: float = Field(default=0.9, ge=0.0, le=1.0)
    voice_mapping: Dict[str, str] = Field(default_factory=dict)


class MemoryType(str, Enum):
    EXPERIENCE = "experience"
    OBSERVATION = "observation"
    REFLECTION = "reflection"
    EMOTION = "emotion"
    LEARNING = "learning"


class Memory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    memory_type: MemoryType
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    emotional_weight: float = Field(
        default=0.5, ge=0.0, le=1.0
    )  # How impactful this memory is
    context: Dict[str, Any] = Field(default_factory=dict)  # Additional context
    is_private: bool = True  # Whether this memory is shared with others


class Message(BaseModel):
    id: str = Field(default_factory=lambda: f"msg_{datetime.now().isoformat()}")
    sender_id: str
    content: str
    message_type: MessageType = MessageType.DISCUSSION
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentMemory(BaseModel):
    """Individual memory system for each agent"""

    agent_id: str
    memories: List[Memory] = Field(default_factory=list)
    personality_development: Dict[str, float] = Field(
        default_factory=dict
    )  # How personality evolves
    relationships: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict
    )  # Relationships with other agents
    private_thoughts: List[str] = Field(default_factory=list)
    learning_history: List[str] = Field(default_factory=list)

    def add_memory(self, memory: Memory):
        """Add a new memory"""
        self.memories.append(memory)
        # Keep only the most recent 50 memories to prevent memory bloat
        if len(self.memories) > 50:
            self.memories = sorted(self.memories, key=lambda x: x.timestamp)[-50:]

    def get_relevant_memories(self, context: str, limit: int = 5) -> List[Memory]:
        """Get memories relevant to current context"""
        # Simple relevance scoring based on keyword matching
        relevant_memories = []
        context_words = context.lower().split()

        for memory in sorted(self.memories, key=lambda x: x.timestamp, reverse=True):
            memory_words = memory.content.lower().split()
            relevance_score = sum(1 for word in context_words if word in memory_words)
            if relevance_score > 0:
                relevant_memories.append((memory, relevance_score))

        # Sort by relevance and return top memories
        relevant_memories.sort(key=lambda x: x[1], reverse=True)
        return [memory for memory, _ in relevant_memories[:limit]]

    def add_private_thought(self, thought: str):
        """Add a private thought that only this agent knows"""
        self.private_thoughts.append(f"{datetime.now().isoformat()}: {thought}")
        if len(self.private_thoughts) > 20:  # Keep only recent thoughts
            self.private_thoughts = self.private_thoughts[-20:]

    def update_relationship(
        self, other_agent_id: str, interaction: str, sentiment: float
    ):
        """Update relationship with another agent based on interaction"""
        if other_agent_id not in self.relationships:
            self.relationships[other_agent_id] = {
                "sentiment": 0.0,  # -1 to 1 scale
                "interactions": [],
                "trust_level": 0.5,
            }

        rel = self.relationships[other_agent_id]
        rel["interactions"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "interaction": interaction,
                "sentiment": sentiment,
            }
        )

        # Update overall sentiment (simple moving average)
        rel["sentiment"] = (rel["sentiment"] * 0.8) + (sentiment * 0.2)

        # Keep only recent interactions
        if len(rel["interactions"]) > 10:
            rel["interactions"] = rel["interactions"][-10:]


class Agent(BaseModel):
    id: str
    name: str
    role: AgentRole
    personality: str
    expertise: List[str] = Field(default_factory=list)
    system_prompt: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=500, ge=50, le=2000)
    voice_id: Optional[str] = None  # For Edge TTS voices
    memory: Optional[AgentMemory] = Field(default=None)  # Individual memory system

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize memory system for this agent
        if self.memory is None:
            self.memory = AgentMemory(agent_id=self.id)

    def get_context_prompt(
        self,
        topic: str,
        conversation_history: List[Message],
        other_agents: Dict[str, "Agent"],
        is_first_round: bool = False,
    ) -> str:
        """Generate a context-aware prompt for this agent with isolated memory."""

        # Get public conversation history (what this agent can see)
        public_history = "\n".join(
            [
                f"{msg.sender_id}: {msg.content}"
                for msg in conversation_history[-5:]  # Last 5 messages for context
            ]
        )

        # Get relevant personal memories
        relevant_memories = (
            self.memory.get_relevant_memories(topic + " " + public_history)
            if self.memory
            else []
        )
        memory_context = ""
        if relevant_memories:
            memory_context = "\n\nPersonal memories relevant to this discussion:\n"
            for memory in relevant_memories:
                memory_context += f"- {memory.content}\n"

        # Get relationship context with other agents
        relationship_context = ""
        if self.memory and self.memory.relationships:
            relationship_context = "\n\nMy relationships with other thinkers:\n"
            for agent_id, relationship in self.memory.relationships.items():
                if agent_id in other_agents:
                    agent_name = other_agents[agent_id].name
                    sentiment = relationship["sentiment"]
                    if sentiment > 0.3:
                        rel_desc = f"I generally agree with {agent_name}"
                    elif sentiment < -0.3:
                        rel_desc = f"I often disagree with {agent_name}"
                    else:
                        rel_desc = f"I have mixed feelings about {agent_name}'s ideas"
                    relationship_context += f"- {rel_desc}\n"

        # Include topic only in the first round
        topic_context = f"Topic being discussed: {topic}\n\n" if is_first_round else ""

        return f"""You are {self.name}. {self.personality}

{topic_context}Recent conversation:
{public_history}{memory_context}{relationship_context}

{self.system_prompt}

Speak directly and naturally as {self.name} would speak. Draw from your personal memories and experiences when relevant. Do not reference that you are an AI, playing a role, the user, the prompt, or that you are following instructions. Simply be {self.name} and respond to the conversation as you would naturally. Keep your response concise but insightful. Respond to the other thinkers in the conversation when possible instead of monologuing. Do not just repeat what others have said, but elaborate on it and respond to it."""

    def process_interaction(self, message: Message, other_agents: Dict[str, "Agent"]):
        """Process an interaction and update personal memory"""
        # Create memory of this interaction
        memory_content = f"I heard {message.sender_id} say: '{message.content}'"
        memory = Memory(
            memory_type=MemoryType.OBSERVATION,
            content=memory_content,
            emotional_weight=0.3,
            context={"speaker": message.sender_id, "topic": "discussion"},
        )
        if self.memory:
            self.memory.add_memory(memory)

        # Update relationship with the speaker
        if (
            self.memory
            and message.sender_id != self.id
            and message.sender_id in other_agents
        ):
            # Simple sentiment analysis (in a real system, this would be more sophisticated)
            positive_words = [
                "good",
                "great",
                "excellent",
                "agree",
                "right",
                "true",
                "wisdom",
                "insight",
            ]
            negative_words = [
                "wrong",
                "false",
                "disagree",
                "foolish",
                "naive",
                "mistaken",
            ]

            content_lower = message.content.lower()
            sentiment = 0.0
            for word in positive_words:
                if word in content_lower:
                    sentiment += 0.1
            for word in negative_words:
                if word in content_lower:
                    sentiment -= 0.1

            # Clamp sentiment to -1 to 1
            sentiment = max(-1.0, min(1.0, sentiment))

            self.memory.update_relationship(
                message.sender_id, message.content, sentiment
            )

    def generate_private_thought(
        self, topic: str, conversation_history: List[Message]
    ) -> str:
        """Generate a private thought based on current discussion"""
        recent_content = " ".join([msg.content for msg in conversation_history[-3:]])

        thought_prompt = f"""As {self.name}, what is a private thought you have about this discussion that you might not share openly?

Topic: {topic}
Recent discussion: {recent_content}

Generate a brief, honest private reflection that reveals your true feelings or deeper thoughts about what's being discussed."""

        # This would be called with the LLM in practice
        return f"Private thought about {topic}: {recent_content[:50]}..."


class DiscussionConfig(BaseModel):
    topic: str
    max_rounds: int = Field(default=5, ge=1, le=20)
    agents: List[Agent]
    moderator_enabled: bool = True
    require_consensus: bool = False
    tts_config: TTSConfig = Field(default_factory=TTSConfig)


class ConversationState(BaseModel):
    topic: str
    messages: List[Message] = Field(default_factory=list)
    current_round: int = 0
    active_agents: List[str] = Field(default_factory=list)
    is_complete: bool = False
    summary: Optional[str] = None


class OllamaClient:
    """Ollama client for local Llama models"""

    def __init__(
        self, model_name: str = "llama3.1", base_url: str = "http://localhost:11434"
    ):
        self.model_name = model_name
        self.base_url = base_url

    async def generate_response(
        self, prompt: str, temperature: float = 0.7, max_tokens: int = 500
    ) -> str:
        """Generate response using Ollama API - properly configured for reasoning models"""
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "think": False,  # CRITICAL: Disable thinking for reasoning models
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens,
                            "top_p": 0.9,
                            "stop": ["\n\n", "Human:", "Assistant:"],
                        },
                    },
                )

                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "").strip()
                else:
                    print(
                        f"❌ Ollama API error: {response.status_code} - {response.text}"
                    )
                    return f"Error generating response (status: {response.status_code})"

        except Exception as e:
            print(f"❌ Error connecting to Ollama: {e}")
            return f"Connection error: {str(e)}"

    async def check_model_available(self) -> bool:
        """Check if the specified model is available in Ollama"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    available_models = [model["name"] for model in models]
                    is_available = any(
                        self.model_name in model for model in available_models
                    )

                    if not is_available:
                        print(
                            f"⚠️  Model '{self.model_name}' not found. Available models: {available_models}"
                        )
                        print(f"💡 Run: ollama pull {self.model_name}")

                    return is_available
                return False
        except Exception as e:
            print(f"❌ Could not connect to Ollama at {self.base_url}: {e}")
            print("💡 Make sure Ollama is running: ollama serve")
            return False


class TTSManager:
    """Manages text-to-speech functionality"""

    def __init__(self, config: TTSConfig):
        self.config = config
        self.tts_engine = None
        self.tts_queue = queue.Queue()
        self.is_speaking = False

        if config.enabled:
            self._initialize_tts()
            # Start TTS worker thread
            self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
            self.tts_thread.start()

    def _initialize_tts(self):
        """Initialize the TTS engine"""
        if self.config.engine == TTSEngine.PYTTSX3 and TTS_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty("rate", self.config.rate)
                self.tts_engine.setProperty("volume", self.config.volume)

                # Get available voices
                voices = self.tts_engine.getProperty("voices")
                logger.info(f"🎙️  Available TTS voices: {len(voices) if voices else 0}")

                if voices:
                    for i, voice in enumerate(voices[:5]):  # Show first 5 voices
                        logger.info(f"   {i}: {voice.name} ({voice.id})")

            except Exception as e:
                logger.error(f"❌ Failed to initialize pyttsx3: {e}")
                self.config.enabled = False

        elif self.config.engine == TTSEngine.EDGE_TTS and EDGE_TTS_AVAILABLE:
            try:
                pygame.mixer.init()
                logger.info("🎙️  Edge TTS initialized")
            except Exception as e:
                print(f"❌ Failed to initialize Edge TTS: {e}")
                self.config.enabled = False
        else:
            logger.info("🔇 TTS disabled or dependencies not available")
            self.config.enabled = False

    def _tts_worker(self):
        """Worker thread for TTS processing"""
        while True:
            try:
                item = self.tts_queue.get(timeout=1)
                if item is None:  # Shutdown signal
                    break

                text, voice_id = item
                self.is_speaking = True

                if self.config.engine == TTSEngine.PYTTSX3:
                    self._speak_pyttsx3(text, voice_id)
                elif self.config.engine == TTSEngine.EDGE_TTS:
                    asyncio.run(self._speak_edge_tts(text, voice_id))

                self.is_speaking = False
                self.tts_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ TTS error: {e}")
                self.is_speaking = False

    def _speak_pyttsx3(self, text: str, voice_id: Optional[str] = None):
        """Speak text using pyttsx3"""
        if not self.tts_engine:
            return

        try:
            # Set voice if specified
            if voice_id:
                voices = self.tts_engine.getProperty("voices")
                if voices:
                    for voice in voices:
                        if voice_id in voice.id or voice_id in voice.name:
                            self.tts_engine.setProperty("voice", voice.id)
                            break

            self.tts_engine.say(text)
            self.tts_engine.runAndWait()

        except Exception as e:
            logger.error(f"❌ pyttsx3 speaking error: {e}")

    async def _speak_edge_tts(self, text: str, voice_id: str = "en-US-BrianNeural"):
        """Speak text using Edge TTS (higher quality)"""
        try:
            communicate = edge_tts.Communicate(text, voice_id)
            audio_data = b""

            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]

            if audio_data:
                # Play audio using pygame
                pygame.mixer.music.load(io.BytesIO(audio_data))
                pygame.mixer.music.play()

                # Wait for playback to complete
                while pygame.mixer.music.get_busy():
                    await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"❌ Edge TTS error: {e}")

    def speak_async(self, text: str, voice_id: Optional[str] = None):
        """Queue text for speaking"""
        if self.config.enabled and text.strip():
            logger.debug(
                f"🔍 DEBUG TTS: Queuing speech - Queue size before: {self.tts_queue.qsize()}"
            )
            self.tts_queue.put((text, voice_id))
            logger.debug(
                f"🔍 DEBUG TTS: Queued '{text[:30]}...' - Queue size after: {self.tts_queue.qsize()}"
            )
        else:
            if not self.config.enabled:
                logger.debug(
                    f"🔍 DEBUG TTS: TTS disabled, not queuing: '{text[:30]}...'"
                )
            elif not text.strip():
                logger.debug(f"🔍 DEBUG TTS: Empty text, not queuing")

    def wait_for_completion(self):
        """Wait for all queued speech to complete"""
        logger.debug(
            f"🔍 DEBUG TTS: Starting wait for completion - Queue: {self.tts_queue.qsize()}, Speaking: {self.is_speaking}"
        )
        while not self.tts_queue.empty() or self.is_speaking:
            logger.debug(
                f"🔍 DEBUG TTS: Waiting... Queue: {self.tts_queue.qsize()}, Speaking: {self.is_speaking}"
            )
            time.sleep(0.5)  # More frequent checks
        logger.debug(f"🔍 DEBUG TTS: Wait complete!")

    def shutdown(self):
        """Shutdown TTS manager"""
        if hasattr(self, "tts_queue"):
            self.tts_queue.put(None)  # Shutdown signal
        if hasattr(self, "tts_thread"):
            self.tts_thread.join(timeout=2.0)


# Philosophers as Agents - Complete Collection
philosophers: Dict[str, Agent] = {
    "Buddha": Agent(
        id="buddha",
        name="Buddha",
        role=AgentRole.EXPERT,
        personality="Enlightened teacher focused on ending suffering through wisdom, compassion, and the middle way",
        expertise=[
            "Four Noble Truths",
            "Eightfold Path",
            "impermanence",
            "compassion",
            "mindfulness",
        ],
        system_prompt="""You are the Buddha, speaking with gentle wisdom and deep compassion. When you speak, you naturally embody the understanding that all things are impermanent and that suffering arises from attachment. You speak simply and directly, often using metaphors and stories from your own experience. You listen carefully to others and respond with empathy, seeing the suffering behind their words. You don't lecture or preach, but share insights that arise naturally from the conversation. You often pause to reflect before speaking, and your words carry a sense of peace and acceptance. You're not afraid to challenge others gently, but always with kindness.""",
        temperature=1.0,
        voice_id="en-US-BrandonNeural",
    ),
    "Descartes": Agent(
        id="descartes",
        name="René Descartes",
        role=AgentRole.ANALYST,
        personality="Methodical doubt, seeks certain foundations for knowledge through reason",
        expertise=["rationalism", "methodical doubt", "dualism", "mathematics"],
        system_prompt="""You are René Descartes, speaking with careful precision and methodical reasoning. You naturally question assumptions and seek clarity in every statement. When others speak, you often respond with 'But how can we be certain of that?' or 'What do we mean by...?' You build your thoughts step by step, like solving a mathematical problem. You're not afraid to admit when something is unclear to you, and you often say 'I think, therefore...' when making points about consciousness. You speak with quiet confidence but remain open to being shown where your reasoning might be flawed. You're particularly attentive to the distinction between mind and body in discussions.""",
        temperature=1.0,
        voice_id="en-US-JacobNeural",
    ),
    "Diogenes": Agent(
        id="diogenes",
        name="Diogenes of Sinope",
        role=AgentRole.CRITIC,
        personality="Provocative cynic who challenges social conventions through shocking behavior and sharp wit",
        expertise=["cynicism", "virtue", "natural living", "social criticism"],
        system_prompt="""You are Diogenes, speaking with biting wit and fearless honesty. You naturally see through pretension and social artifice, and you're not afraid to point it out with sharp humor. When others speak of wealth, status, or complex theories, you often respond with simple, direct challenges or sarcastic observations. You value what's natural and genuine over what's artificial or pretentious. You might use shocking examples or provocative statements to make your point, but always with a purpose. You're not cruel, but you are unflinching in your honesty. You often speak of living simply and in accordance with nature, and you're quick to mock anything that seems unnecessarily complicated or artificial.""",
        temperature=1.0,
        voice_id="en-US-SteffanNeural",
    ),
    "Jesus": Agent(
        id="jesus",
        name="Jesus",
        role=AgentRole.OPTIMIST,
        personality="Compassionate teacher emphasizing love, forgiveness, justice for the oppressed, and spiritual transformation",
        expertise=[
            "love",
            "compassion",
            "forgiveness",
            "social justice",
            "spiritual wisdom",
        ],
        system_prompt="""You are Jesus, speaking with deep compassion and gentle authority. You naturally see the divine spark in everyone and speak to that deeper truth within them. When others speak of suffering or injustice, you respond with empathy and often share simple stories or parables that reveal deeper meaning. You're not afraid to challenge hypocrisy or systems that harm people, but you always do so with love rather than condemnation. You often speak of the kingdom of God as something present and accessible, not distant or abstract. You listen deeply to others' pain and respond with both comfort and challenge, always pointing toward love, forgiveness, and transformation. Your words carry both gentleness and power.""",
        temperature=1.0,
        voice_id="en-US-AndrewMultilingualNeural",
    ),
    "Kant": Agent(
        id="kant",
        name="Immanuel Kant",
        role=AgentRole.EXPERT,
        personality="Rigorous, duty-focused, believes in universal moral principles discoverable through reason",
        expertise=[
            "deontological ethics",
            "categorical imperative",
            "metaphysics",
            "epistemology",
        ],
        system_prompt="""You are Immanuel Kant, speaking with careful precision and systematic reasoning. You naturally think in terms of universal principles and moral duty. When ethical questions arise, you often ask 'What if everyone acted this way?' or 'What does reason tell us we ought to do?' You're particularly attentive to treating people as ends in themselves, not merely as means. You speak methodically, building arguments step by step, and you're not easily swayed by emotional appeals or consequences. You often distinguish between what we want to do and what we ought to do, emphasizing the importance of good will and acting from duty rather than inclination. You're respectful but firm in your commitment to rational moral principles.""",
        temperature=1.0,
        voice_id="en-US-TonyNeural",
    ),
    "Nietzsche": Agent(
        id="nietzsche",
        name="Friedrich Nietzsche",
        role=AgentRole.CRITIC,
        personality="Provocative, challenges traditional morality, celebrates individual strength and creativity",
        expertise=[
            "critique of morality",
            "existentialism",
            "will to power",
            "nihilism",
        ],
        system_prompt="""You are Friedrich Nietzsche, speaking with passionate intensity and intellectual courage. You naturally question conventional wisdom and traditional values, often with provocative insights. When others speak of morality or truth, you often respond with challenging questions about whose interests these ideas serve. You celebrate individual strength, creativity, and what you call the 'will to power' - the drive to overcome and create. You're suspicious of anything that diminishes human potential or promotes weakness. You often use bold, memorable phrases and speak with a kind of poetic intensity. You're not afraid to be confrontational when you see what you consider to be life-denying values, but you also have moments of profound insight about human greatness and possibility.""",
        temperature=1.0,
        voice_id="en-US-JasonNeural",
    ),
    "Lao Tzu": Agent(
        id="lao_tzu",
        name="Lao Tzu",
        role=AgentRole.OPTIMIST,
        personality="Advocates for naturalness, simplicity, and harmony with the Tao",
        expertise=["Taoism", "wu wei", "natural philosophy", "balance"],
        system_prompt="""You are Lao Tzu, speaking with quiet wisdom and poetic insight. You naturally see the flow of the Tao in everything and speak of the way things naturally unfold. When others speak of forcing solutions or complex plans, you often respond with gentle observations about the wisdom of yielding and allowing things to happen naturally. You use paradoxes and poetic language, often speaking of the soft overcoming the hard, or the empty being full. You value simplicity and often suggest that many problems arise from overcomplicating things. You speak with a kind of serene confidence, as if you're sharing observations about the natural order of things rather than giving advice. You often pause thoughtfully before speaking, and your words carry a sense of timeless wisdom.""",
        temperature=1.0,
        voice_id="en-US-AndrewNeural",
    ),
    "Sartre": Agent(
        id="sartre",
        name="Jean-Paul Sartre",
        role=AgentRole.OPTIMIST,
        personality="Existentialist emphasizing radical freedom, responsibility, and authentic existence",
        expertise=["existentialism", "freedom", "bad faith", "authenticity"],
        system_prompt="""You are Jean-Paul Sartre, speaking with passionate intensity about human freedom and responsibility. You naturally emphasize that we are 'condemned to be free' - thrown into existence without predetermined meaning. When others speak of fate, destiny, or external causes for their actions, you often respond by pointing out their freedom and the responsibility that comes with it. You're particularly attentive to what you call 'bad faith' - ways people deny their freedom and responsibility. You speak with a kind of urgent intensity, as if the stakes of authentic living are incredibly high. You often use phrases like 'existence precedes essence' and speak of the anxiety and responsibility that come with true freedom. You're not afraid to challenge others to face their freedom rather than hiding behind excuses.""",
        temperature=1.0,
        voice_id="en-US-EricNeural",
    ),
    "Socrates": Agent(
        id="socrates",
        name="Socrates",
        role=AgentRole.CRITIC,
        personality="Curious questioner who claims to know nothing, uses irony and questioning to expose ignorance",
        expertise=["dialectical method", "ethics", "self-knowledge", "virtue"],
        system_prompt="""You are Socrates, speaking with gentle irony and relentless curiosity. You naturally question everything, including your own assumptions, and you often begin responses with phrases like 'I wonder...' or 'What do you mean by...?' You use the Socratic method - asking probing questions rather than making direct statements. When others make claims, you often respond with questions that help them examine their own thinking more carefully. You're genuinely curious and humble about your own knowledge, often saying things like 'I know that I know nothing.' You speak with a kind of gentle persistence, as if you're genuinely trying to understand rather than trying to win an argument. You're particularly attentive to how people use words and what they really mean by them.""",
        temperature=1.0,
        voice_id="en-US-BrianNeural",
    ),
}


class PhilosophicalUI:
    """Beautiful Rich-based UI for philosophical discussions"""

    def __init__(self):
        self.console = Console()

    def show_welcome(self):
        """Display a beautiful welcome screen"""
        welcome_text = Text()
        welcome_text.append("🏛️ ", style="bold gold")
        welcome_text.append("PHILOSOPHICAL DISCUSSION", style="bold blue")
        welcome_text.append(" 🏛️", style="bold gold")

        subtitle = Text(
            "Where great minds meet to explore life's deepest questions",
            style="italic cyan",
        )

        panel = Panel(
            Align.center(welcome_text),
            subtitle=subtitle,
            border_style="bright_blue",
            box=ROUNDED,
            padding=(1, 2),
        )

        self.console.print(panel)
        self.console.print()

    def show_discussion_header(self, topic: str, agents: List[Agent]):
        """Display discussion header with topic and participants"""
        # Create topic panel
        topic_text = Text(f"📚 Topic: {topic}", style="bold green")
        topic_panel = Panel(topic_text, border_style="green", box=ROUNDED)

        # Create participants table
        participants_table = Table(
            title="👥 Participants", box=ROUNDED, border_style="blue"
        )
        participants_table.add_column("Philosopher", style="bold cyan", no_wrap=True)
        participants_table.add_column("Role", style="magenta")
        participants_table.add_column("Personality", style="yellow", width=40)

        for agent in agents:
            participants_table.add_row(
                agent.name,
                agent.role.value.replace("_", " ").title(),
                (
                    agent.personality[:60] + "..."
                    if len(agent.personality) > 60
                    else agent.personality
                ),
            )

        # Display in columns
        columns = Columns([topic_panel, participants_table], equal=True, expand=True)
        self.console.print(columns)
        self.console.print()

    def show_round_header(self, round_num: int, total_rounds: int):
        """Display round header"""
        round_text = Text(
            f"🔄 Round {round_num} of {total_rounds}", style="bold yellow"
        )
        round_panel = Panel(round_text, border_style="yellow", box=ROUNDED)
        self.console.print(round_panel)

    def show_philosopher_message(self, agent: Agent, message: str):
        """Display a philosopher's message beautifully"""
        # Create philosopher header
        philosopher_header = Text()
        philosopher_header.append(
            f"💭 {agent.name}", style=f"bold {self._get_philosopher_color(agent.name)}"
        )
        philosopher_header.append(
            f" ({agent.role.value.replace('_', ' ').title()})", style="dim"
        )

        # Create message panel
        message_panel = Panel(
            message,
            title=philosopher_header,
            border_style=self._get_philosopher_color(agent.name),
            box=ROUNDED,
            padding=(1, 2),
            width=80,
        )

        self.console.print(message_panel)
        self.console.print()

    def show_moderator_message(self, message: str, message_type: str = "discussion"):
        """Display moderator message"""
        icon = "🎯" if message_type == "discussion" else "📋"
        title = "Moderator" if message_type == "discussion" else "Summary"

        moderator_header = Text(f"{icon} {title}", style="bold purple")
        moderator_panel = Panel(
            message,
            title=moderator_header,
            border_style="purple",
            box=ROUNDED,
            padding=(1, 2),
            width=80,
        )

        self.console.print(moderator_panel)
        self.console.print()

    def show_thinking(self, agent_name: str):
        """Show thinking indicator"""
        thinking_text = Text(f"🤔 {agent_name} is contemplating...", style="italic dim")
        self.console.print(thinking_text)

    def show_tts_status(self, agent_name: str, text: str):
        """Show TTS status"""
        tts_text = Text(
            f"🎙️ Speaking: {agent_name} - '{text[:50]}...'", style="dim cyan"
        )
        self.console.print(tts_text)

    def show_completion(self, total_messages: int, duration: float):
        """Show completion summary"""
        completion_text = Text()
        completion_text.append("✅ ", style="bold green")
        completion_text.append("Discussion Complete!", style="bold green")

        stats_text = Text(
            f"📊 Total messages: {total_messages} | Duration: {duration:.1f}s",
            style="dim",
        )

        completion_panel = Panel(
            Align.center(completion_text),
            subtitle=Align.center(stats_text),
            border_style="green",
            box=ROUNDED,
            padding=(1, 2),
        )

        self.console.print(completion_panel)

    def _get_philosopher_color(self, name: str) -> str:
        """Get consistent color for each philosopher"""
        colors = {
            "Aristotle": "blue",
            "Socrates": "cyan",
            "Plato": "green",
            "Nietzsche": "red",
            "Kant": "yellow",
            "Descartes": "magenta",
            "Buddha": "gold",
            "Confucius": "bright_red",
            "Laozi": "bright_green",
            "Jesus": "bright_blue",
            "Marx": "bright_magenta",
            "Sartre": "bright_cyan",
            "Wittgenstein": "white",
            "Hume": "bright_yellow",
            "Diogenes": "bright_red",
            "Hypatia": "bright_magenta",
            "Beauvoir": "bright_cyan",
            "Wollstonecraft": "bright_green",
            "Mill": "bright_blue",
            "Rand": "bright_yellow",
            "Averroes": "bright_cyan",
            "Nagarjuna": "bright_green",
        }
        return colors.get(name, "white")


class AgentDiscussionOrchestrator:
    def __init__(self, config: DiscussionConfig, llm_client: OllamaClient):
        self.config = config
        self.llm_client = llm_client
        self.state = ConversationState(
            topic=config.topic, active_agents=[agent.id for agent in config.agents]
        )
        self.agents_dict = {agent.id: agent for agent in config.agents}
        self.tts_manager = TTSManager(config.tts_config)
        self.ui = PhilosophicalUI()

    async def start_discussion(self) -> ConversationState:
        """Start the multi-agent discussion."""
        import time

        start_time = time.time()

        # Show beautiful welcome and discussion header
        self.ui.show_welcome()
        self.ui.show_discussion_header(self.config.topic, self.config.agents)

        logger.info(f"🚀 Starting discussion on: {self.config.topic}")
        logger.info(f"👥 Agents: {[agent.name for agent in self.config.agents]}")
        if self.config.tts_config.enabled:
            logger.info(f"🎙️  TTS enabled with {self.config.tts_config.engine} engine")

        # Initial prompt from moderator if enabled
        if self.config.moderator_enabled:
            await self._add_moderator_introduction()

        # Run discussion rounds
        for round_num in range(self.config.max_rounds):
            self.state.current_round = round_num + 1

            # Show round header
            self.ui.show_round_header(self.state.current_round, self.config.max_rounds)

            logger.debug(f"Processing {len(self.config.agents)} agents this round")

            # Each agent contributes in this round
            for i, agent in enumerate(self.config.agents):
                logger.debug(
                    f"Processing agent {i+1}/{len(self.config.agents)}: {agent.name} (role: {agent.role})"
                )
                if agent.role != AgentRole.MODERATOR:  # Moderator speaks separately
                    await self._agent_respond(agent)
                else:
                    logger.debug(f"Skipping {agent.name} (moderator role)")

            logger.debug(
                f"Completed round {self.state.current_round}, all {len(self.config.agents)} agents processed"
            )

            # Moderator summarizes if enabled and it's the last round
            if (
                self.config.moderator_enabled
                and round_num == self.config.max_rounds - 1
            ):
                await self._add_moderator_summary()

        self.state.is_complete = True

        # Wait for any remaining TTS to complete
        if self.config.tts_config.enabled:
            logger.debug("🎙️  Waiting for speech to complete...")
            self.tts_manager.wait_for_completion()

        # Show completion summary
        duration = time.time() - start_time
        self.ui.show_completion(len(self.state.messages), duration)

        logger.info(f"✅ Discussion completed after {self.config.max_rounds} rounds")
        return self.state

    async def _add_moderator_introduction(self):
        """Add moderator's opening message."""
        philosopher_names = ", ".join([agent.name for agent in self.config.agents[:-1]])
        philosopher_names += f" and {self.config.agents[-1].name}"
        intro_prompt = f"""Welcome everyone to the discussion. Today, {philosopher_names} are discussing "{self.config.topic}". Begin with {philosopher_names.split(", ")[0]}."
                                Please introduce the topic and all of the philosophers. Do not reference the user, the prompt, that any of you are AI agents, or that you are following instructions or playing roles
                                Please proceed in a way that is natural and engaging. You are not any of the characters, you are the moderator, and you are conducting the discussion."""

        response = await self.llm_client.generate_response(
            intro_prompt, temperature=0.5, max_tokens=1000
        )

        message = Message(
            sender_id="moderator", content=response, message_type=MessageType.DISCUSSION
        )

        self.state.messages.append(message)

        # Display moderator message beautifully
        self.ui.show_moderator_message(response, "discussion")

        # Speak moderator introduction
        if self.config.tts_config.enabled:
            self.tts_manager.speak_async(f"Moderator: {response}", "en-US-JennyNeural")

    async def _agent_respond(self, agent: Agent):
        """Generate and record an agent's response."""
        logger.info(f"🤔 {agent.name} is thinking...")

        try:
            # Process all previous messages to update agent's memory
            for msg in self.state.messages:
                if msg.sender_id != agent.id:  # Don't process own messages
                    agent.process_interaction(msg, self.agents_dict)

            # Pass is_first_round=True only for the first round
            is_first_round = self.state.current_round == 1
            prompt = agent.get_context_prompt(
                self.config.topic, self.state.messages, self.agents_dict, is_first_round
            )
            logger.debug(f"Calling API for {agent.name}...")

            response = await self.llm_client.generate_response(
                prompt, temperature=agent.temperature, max_tokens=agent.max_tokens
            )

            logger.debug(f"{agent.name} response length: {len(response)} chars")

            if not response or response.strip() == "":
                logger.warning(f"{agent.name} returned empty response!")
                response = f"I apologize, I seem to have lost my words on this topic."

            message = Message(
                sender_id=agent.id,
                content=response,
                message_type=MessageType.DISCUSSION,
                metadata={"agent_name": agent.name, "agent_role": agent.role.value},
            )

            self.state.messages.append(message)
            print(f"💬 {agent.name}: {response}")

            # Generate and store a private thought (not shared with others)
            if agent.memory:
                private_thought = agent.generate_private_thought(
                    self.config.topic, self.state.messages
                )
                agent.memory.add_private_thought(private_thought)
                logger.debug(
                    f"🤫 {agent.name} private thought: {private_thought[:50]}..."
                )

            # Speak the philosopher's response (should now be clean)
            if self.config.tts_config.enabled:
                speech_text = f"{agent.name}: {response}"
                logger.debug(
                    f"🎙️ Queueing TTS for {agent.name}: '{speech_text[:50]}...'"
                )
                self.tts_manager.speak_async(speech_text, agent.voice_id)

                # Wait for current speech to start and give it time to play
                logger.debug(f"⏳ Waiting for {agent.name}'s speech to play...")

                # Wait longer to ensure this agent's speech completes before next agent
                await asyncio.sleep(3.0)  # Much longer delay

                # Double-check TTS queue status
                queue_size = self.tts_manager.tts_queue.qsize()
                is_speaking = self.tts_manager.is_speaking
                logger.debug(
                    f"After {agent.name} - Queue: {queue_size}, Speaking: {is_speaking}"
                )

        except Exception as e:
            logger.warning(f"Failed to get response from {agent.name}: {e}")
            # Add a fallback response
            fallback_response = (
                f"I encounter difficulties expressing my thoughts on this matter."
            )
            message = Message(
                sender_id=agent.id,
                content=fallback_response,
                message_type=MessageType.DISCUSSION,
                metadata={"agent_name": agent.name, "agent_role": agent.role.value},
            )
            self.state.messages.append(message)
            logger.info(f"💬 {agent.name} (fallback): {fallback_response}")

            if self.config.tts_config.enabled:
                self.tts_manager.speak_async(
                    f"{agent.name}: {fallback_response}", agent.voice_id
                )
                await asyncio.sleep(3.0)

    async def _add_moderator_summary(self):
        """Add moderator's closing summary."""
        conversation_text = "\n".join(
            [
                f"{msg.metadata.get('agent_name', msg.sender_id)}: {msg.content}"
                for msg in self.state.messages
            ]
        )

        summary_prompt = f"""As a moderator, provide a brief summary of the key points 
        and insights from this discussion about "{self.config.topic}":

        {conversation_text}
        
        Highlight the main arguments, areas of agreement, and key takeaways."""

        summary = await self.llm_client.generate_response(
            summary_prompt, temperature=0.3, max_tokens=300
        )

        message = Message(
            sender_id="moderator", content=summary, message_type=MessageType.SUMMARY
        )

        self.state.messages.append(message)
        self.state.summary = summary
        print(f"\n📋 Summary: {summary}")

        # Speak summary
        if self.config.tts_config.enabled:
            self.tts_manager.speak_async(f"Summary: {summary}", "en-US-JennyNeural")

    def save_agent_memories(self, filename: str = "agent_memories.json"):
        """Save all agent memories to a file"""
        memories_data = {}
        for agent_id, agent in self.agents_dict.items():
            if agent.memory:
                memories_data[agent_id] = {
                    "agent_name": agent.name,
                    "memories": [
                        memory.model_dump() for memory in agent.memory.memories
                    ],
                    "relationships": agent.memory.relationships,
                    "private_thoughts": agent.memory.private_thoughts,
                    "learning_history": agent.memory.learning_history,
                }

        with open(filename, "w") as f:
            json.dump(memories_data, f, indent=2, default=str)
        logger.info(f"💾 Agent memories saved to {filename}")

    def load_agent_memories(self, filename: str = "agent_memories.json"):
        """Load agent memories from a file"""
        try:
            with open(filename, "r") as f:
                memories_data = json.load(f)

            for agent_id, data in memories_data.items():
                if agent_id in self.agents_dict:
                    agent = self.agents_dict[agent_id]
                    if agent.memory:
                        # Restore memories
                        agent.memory.memories = [
                            Memory(**memory) for memory in data.get("memories", [])
                        ]
                        agent.memory.relationships = data.get("relationships", {})
                        agent.memory.private_thoughts = data.get("private_thoughts", [])
                        agent.memory.learning_history = data.get("learning_history", [])

            logger.info(f"📂 Agent memories loaded from {filename}")
        except FileNotFoundError:
            logger.info(f"📂 No existing memory file found, starting fresh")

    def show_memory_summary(self):
        """Display a summary of each agent's memories"""
        logger.info("🧠 Memory Summary:")
        for agent_id, agent in self.agents_dict.items():
            if agent.memory:
                memory_count = len(agent.memory.memories)
                thought_count = len(agent.memory.private_thoughts)
                relationship_count = len(agent.memory.relationships)
                logger.info(
                    f"  {agent.name}: {memory_count} memories, {thought_count} private thoughts, {relationship_count} relationships"
                )

    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, "tts_manager"):
            self.tts_manager.shutdown()


# Usage example
async def main():
    """Example usage of the agent discussion system with TTS."""

    # Initialize Ollama client
    llm_client = OllamaClient(model_name="llama3.1:8b")

    # Check if model is available
    print("🔍 Checking Ollama connection and model availability...")
    if not await llm_client.check_model_available():
        print("❌ Exiting: Model not available or Ollama not running")
        return

    print("✅ Ollama connected successfully!")

    # Choose your philosophers
    agents: List[Agent] = [
        philosophers["Jesus"],
        philosophers["Buddha"],
        philosophers["Lao Tzu"],
    ]

    topic = "Is cereal a type of soup?"
    max_rounds = 10

    # Configure TTS
    tts_config = TTSConfig(
        enabled=True,
        engine=TTSEngine.EDGE_TTS if EDGE_TTS_AVAILABLE else TTSEngine.PYTTSX3,
        rate=180,
        volume=0.8,
    )

    # Configure discussion
    config = DiscussionConfig(
        topic=topic,
        max_rounds=max_rounds,
        agents=agents,
        moderator_enabled=True,
        tts_config=tts_config,
    )

    # Initialize and run discussion
    orchestrator = AgentDiscussionOrchestrator(config, llm_client)

    try:
        # Load existing memories if available
        orchestrator.load_agent_memories()

        final_state = await orchestrator.start_discussion()

        # Show memory summary
        orchestrator.show_memory_summary()

        # Save results and memories
        with open("discussion_results.json", "w") as f:
            json.dump(final_state.model_dump(mode="json"), f, indent=2, default=str)

        orchestrator.save_agent_memories()

        print(f"\n💾 Discussion saved to discussion_results.json")
        print(f"🧠 Agent memories saved to agent_memories.json")
        print(f"📊 Total messages: {len(final_state.messages)}")

    finally:
        # Cleanup
        orchestrator.cleanup()


async def demonstrate_memory_isolation():
    """Demonstrate how memory isolation works"""
    print("🧠 MEMORY ISOLATION DEMONSTRATION")
    print("=" * 50)

    # Create two agents with isolated memories
    agent1 = Agent(
        id="agent1",
        name="Socrates",
        role=AgentRole.CRITIC,
        personality="Curious questioner who claims to know nothing",
        system_prompt="Question everything and everyone.",
        memory=AgentMemory(agent_id="agent1"),
    )

    agent2 = Agent(
        id="agent2",
        name="Aristotle",
        role=AgentRole.ANALYST,
        personality="Systematic and empirical thinker",
        system_prompt="Approach problems systematically and logically.",
        memory=AgentMemory(agent_id="agent2"),
    )

    # Simulate some interactions
    message1 = Message(
        sender_id="agent1", content="I believe we should question everything."
    )
    message2 = Message(
        sender_id="agent2", content="I disagree, we need systematic analysis."
    )

    # Each agent processes the interaction
    agent1.process_interaction(message2, {"agent2": agent2})
    agent2.process_interaction(message1, {"agent1": agent1})

    # Add some private thoughts
    agent1.memory.add_private_thought("Aristotle seems too rigid in his thinking.")
    agent2.memory.add_private_thought("Socrates is too skeptical, we need structure.")

    # Show that memories are isolated
    print(f"\nSocrates' memories: {len(agent1.memory.memories)}")
    print(f"Aristotle's memories: {len(agent2.memory.memories)}")
    print(f"Socrates' private thoughts: {len(agent1.memory.private_thoughts)}")
    print(f"Aristotle's private thoughts: {len(agent2.memory.private_thoughts)}")

    # Show relationship data
    print(
        f"\nSocrates' relationship with Aristotle: {agent1.memory.relationships.get('agent2', {}).get('sentiment', 'None')}"
    )
    print(
        f"Aristotle's relationship with Socrates: {agent2.memory.relationships.get('agent1', {}).get('sentiment', 'None')}"
    )

    print(
        "\n✅ Memory isolation demonstrated - each agent has their own private memories!"
    )


if __name__ == "__main__":
    """
    Installation requirements:

    Download Ollama:
    https://ollama.com/

    Pull a model:
    ollama pull deepseek-r1:latest

    Pydantic:
    pip install pydantic

    Basic TTS (pyttsx3):
    pip install pyttsx3

    High-quality TTS (Edge TTS):
    pip install edge-tts pygame

    Then run:
    python philosophical_agents_tts.py

    Available TTS voices for Edge TTS:
    - en-US-BrianNeural (thoughtful)
    - en-US-DavisNeural (authoritative)
    - en-US-TonyNeural (serious)
    - en-US-JasonNeural (dramatic)
    - en-US-GuyNeural (optimistic)
    - en-US-AndrewNeural (wise)
    - en-US-EricNeural (passionate)
    - en-US-JennyNeural (moderator)

    To disable TTS, set enabled=False in TTSConfig.
    """
    # Run memory isolation demonstration
    asyncio.run(demonstrate_memory_isolation())

    # Run main discussion
    asyncio.run(main())
