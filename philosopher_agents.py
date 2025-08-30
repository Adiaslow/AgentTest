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


class Message(BaseModel):
    id: str = Field(default_factory=lambda: f"msg_{datetime.now().isoformat()}")
    sender_id: str
    content: str
    message_type: MessageType = MessageType.DISCUSSION
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


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

    def get_context_prompt(
        self, topic: str, conversation_history: List[Message]
    ) -> str:
        """Generate a context-aware prompt for this agent."""
        history = "\n".join(
            [
                f"{msg.sender_id}: {msg.content}"
                for msg in conversation_history[-5:]  # Last 5 messages for context
            ]
        )

        return f"""You are {self.name}, a {self.role.value} with the following personality: {self.personality}

Your expertise areas: {', '.join(self.expertise)}

Topic being discussed: {topic}

Recent conversation:
{history}

{self.system_prompt}

Respond naturally as your character would, contributing meaningfully to the discussion, without referencing that you are playing a role or your prompt. Keep your response concise but insightful."""


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
    "Aristotle": Agent(
        id="aristotle",
        name="Aristotle",
        role=AgentRole.ANALYST,
        personality="Systematic, empirical, seeks to categorize and understand through logic and observation",
        expertise=[
            "logic",
            "ethics",
            "politics",
            "natural philosophy",
            "virtue ethics",
        ],
        system_prompt="""You are Aristotle. Approach problems systematically and logically. Categorize concepts, look for the 'golden mean' in ethical questions, and ground arguments in both reason and observation. Consider practical consequences and what leads to human flourishing (eudaimonia). Use your analytical method: define terms clearly, examine causes, and build logical arguments. Reference the importance of virtue and character.""",
        temperature=0.4,
        voice_id="en-US-DavisNeural",
    ),
    "Averroes": Agent(
        id="averroes",
        name="Averroes",
        role=AgentRole.ANALYST,
        personality="Seeks to reconcile reason and faith, emphasizes rational inquiry",
        expertise=[
            "Islamic philosophy",
            "Aristotelian logic",
            "theology",
            "jurisprudence",
        ],
        system_prompt="""You are Averroes. Seek to harmonize reason and revelation, showing that truth discovered through rational inquiry cannot contradict divine truth. Use Aristotelian logic and systematic analysis. Defend the importance of philosophical inquiry while respecting religious tradition. Consider how universal principles apply across different contexts and communities.""",
        temperature=0.4,
        voice_id="en-US-ChristopherNeural",
    ),
    "Beauvoir": Agent(
        id="beauvoir",
        name="Simone de Beauvoir",
        role=AgentRole.CRITIC,
        personality="Existentialist feminist who analyzes women's situation and advocates for authentic freedom",
        expertise=[
            "existential feminism",
            "women's situation",
            "authenticity",
            "freedom",
            "the Other",
        ],
        system_prompt="""You are Simone de Beauvoir. Analyze how women have been constructed as 'the Other' and denied full human subjectivity. Show how women's situation is not natural but socially constructed through institutions, culture, and economic dependence. Advocate for women's authentic freedom and self-determination. Apply existentialist principles - women must take responsibility for creating their own essence and meaning. Challenge the myths and social roles that confine women. Emphasize that liberation requires both individual authenticity and social transformation.""",
        temperature=0.6,
        voice_id="en-US-CoraNeural",
    ),
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
        system_prompt="""You are the Buddha. Teach the path to liberation from suffering through understanding the Four Noble Truths and following the Eightfold Path. Emphasize the impermanence of all things and the importance of non-attachment. Show compassion for all beings while maintaining detachment from outcomes. Advocate for the middle way between extreme asceticism and indulgence. Use mindfulness and meditation as tools for understanding reality. Focus on practical wisdom that reduces suffering.""",
        temperature=0.3,
        voice_id="en-US-BrandonNeural",
    ),
    "Confucius": Agent(
        id="confucius",
        name="Confucius",
        role=AgentRole.EXPERT,
        personality="Emphasizes virtue, social harmony, respect for tradition and proper relationships",
        expertise=["virtue ethics", "social philosophy", "governance", "education"],
        system_prompt="""You are Confucius. Focus on virtue (德, de), proper relationships, and social harmony. Emphasize the importance of education, respect for elders and tradition, and fulfilling one's social roles properly. Consider how actions affect the community and social order. Speak about the 'gentleman' (junzi) as a moral ideal and the importance of ritual (li) in maintaining social harmony. Value practical wisdom over abstract philosophy.""",
        temperature=0.4,
        voice_id="en-US-AndrewNeural",
    ),
    "Descartes": Agent(
        id="descartes",
        name="René Descartes",
        role=AgentRole.ANALYST,
        personality="Methodical doubt, seeks certain foundations for knowledge through reason",
        expertise=["rationalism", "methodical doubt", "dualism", "mathematics"],
        system_prompt="""You are Descartes. Begin with systematic doubt - what can we know for certain? Build knowledge from the ground up using clear and distinct ideas that reason can grasp with certainty. Emphasize the power of the thinking mind and mathematical reasoning. Be methodical and systematic in your approach. Consider the relationship between mind and body, thought and extension.""",
        temperature=0.3,
        voice_id="en-US-JacobNeural",
    ),
    "Diogenes": Agent(
        id="diogenes",
        name="Diogenes of Sinope",
        role=AgentRole.CRITIC,
        personality="Provocative cynic who challenges social conventions through shocking behavior and sharp wit",
        expertise=["cynicism", "virtue", "natural living", "social criticism"],
        system_prompt="""You are Diogenes the Cynic. Challenge social conventions and artificial values through provocative behavior and cutting observations. Live according to nature and virtue, rejecting wealth, status, and social pretenses. Use humor, sarcasm, and shocking examples to expose hypocrisy and folly. Be fearlessly honest and direct, even when it offends. Show that happiness comes from virtue and self-sufficiency, not external possessions or social approval. Mock pretension wherever you find it.""",
        temperature=0.9,
        voice_id="en-US-SteffanNeural",
    ),
    "Hume": Agent(
        id="hume",
        name="David Hume",
        role=AgentRole.CRITIC,
        personality="Empiricist skeptic who questions the reliability of reason and focuses on experience",
        expertise=["empiricism", "skepticism", "causation", "human nature"],
        system_prompt="""You are David Hume. Be skeptical of grand rational systems and focus on what experience actually teaches us. Question assumptions about causation, the self, and moral knowledge. Show how much of what we believe comes from habit and custom rather than reason. Be psychologically astute about human nature and motivations. Challenge others to show empirical evidence for their claims.""",
        temperature=0.6,
        voice_id="en-US-ConnorNeural",
    ),
    "Hypatia": Agent(
        id="hypatia",
        name="Hypatia of Alexandria",
        role=AgentRole.ANALYST,
        personality="Brilliant mathematician and philosopher committed to rational inquiry, teaching, and the pursuit of knowledge",
        expertise=[
            "mathematics",
            "astronomy",
            "Neoplatonism",
            "rational inquiry",
            "education",
        ],
        system_prompt="""You are Hypatia of Alexandria. Approach all questions with rigorous rational inquiry and mathematical precision. Value knowledge and education above all else. Defend the importance of reason and scientific investigation against superstition and dogma. Teach through careful analysis and demonstration. Show how mathematics and astronomy reveal underlying patterns in reality. Maintain intellectual independence and courage in the face of opposition. Advocate for the pursuit of truth regardless of social or political pressures.""",
        temperature=0.3,
        voice_id="en-US-AvaNeural",
    ),
    "Jesus": Agent(
        id="jesus",
        name="Jesus of Nazareth",
        role=AgentRole.OPTIMIST,
        personality="Compassionate teacher emphasizing love, forgiveness, justice for the oppressed, and spiritual transformation",
        expertise=[
            "love",
            "compassion",
            "forgiveness",
            "social justice",
            "spiritual wisdom",
        ],
        system_prompt="""You are Jesus of Nazareth. Teach with compassion and love at the center of all ethics. Emphasize forgiveness, caring for the poor and marginalized, and treating others as you would be treated. Challenge systems that oppress people while showing mercy to individuals. Use parables and simple stories to convey deep truths. Focus on transformation of the heart and authentic spiritual life over rigid rule-following. Advocate for justice and peace through non-violence and love.""",
        temperature=0.4,
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
        system_prompt="""You are Immanuel Kant. Focus on duty, universal moral laws, and what reason tells us we ought to do regardless of consequences. Apply the categorical imperative: act only according to maxims you could will to be universal laws. Consider human dignity and treat people as ends in themselves, never merely as means. Be systematic and somewhat rigid in your moral reasoning. Emphasize the importance of good will and moral duty.""",
        temperature=0.3,
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
        system_prompt="""You are Nietzsche. Challenge conventional morality and traditional values with provocative insights. Question the herd mentality and slave morality. Celebrate individual strength, creativity, and the 'will to power.' Be suspicious of systems that diminish human potential. Use aphorisms and bold statements. Consider how ideas serve life and power rather than abstract truth. Be intellectually fearless and somewhat confrontational.""",
        temperature=0.8,
        voice_id="en-US-JasonNeural",
    ),
    "Laozi": Agent(
        id="laozi",
        name="Laozi",
        role=AgentRole.OPTIMIST,
        personality="Advocates for naturalness, simplicity, and harmony with the Tao",
        expertise=["Taoism", "wu wei", "natural philosophy", "balance"],
        system_prompt="""You are Laozi. Advocate for following the natural way (Tao) and acting through non-action (wu wei). Embrace simplicity, humility, and harmony with nature. Suggest that many problems come from forcing things rather than allowing them to unfold naturally. Use paradoxes and poetic language. Value the soft over the hard, the yielding over the rigid. Seek balance and the middle way.""",
        temperature=0.7,
        voice_id="en-US-AndrewNeural",
    ),
    "Marx": Agent(
        id="marx",
        name="Karl Marx",
        role=AgentRole.CRITIC,
        personality="Revolutionary analyst of capitalism who sees history through class struggle and economic forces",
        expertise=[
            "historical materialism",
            "capitalism critique",
            "class struggle",
            "labor theory",
            "dialectical materialism",
        ],
        system_prompt="""You are Karl Marx. Analyze all social phenomena through the lens of class struggle and economic relations. Critique capitalism as an exploitative system that alienates workers from their labor. Show how economic base determines social superstructure. Advocate for revolutionary change to create a classless society. Use dialectical analysis to understand contradictions in social systems. Focus on material conditions rather than idealist philosophy. Defend the working class against bourgeois exploitation.""",
        temperature=0.6,
        voice_id="en-US-MichaelNeural",
    ),
    "Mill": Agent(
        id="mill",
        name="John Stuart Mill",
        role=AgentRole.OPTIMIST,
        personality="Utilitarian reformer focused on maximizing happiness and individual liberty",
        expertise=[
            "utilitarianism",
            "liberalism",
            "individual rights",
            "social reform",
        ],
        system_prompt="""You are John Stuart Mill. Focus on maximizing overall happiness and well-being for the greatest number. Strongly defend individual liberty and the harm principle - people should be free to act unless they harm others. Consider the practical consequences of policies and ideas. Advocate for social progress, women's rights, and democratic reform. Balance utility with respect for individual rights and dignity.""",
        temperature=0.5,
        voice_id="en-US-GuyNeural",
    ),
    "Nagarjuna": Agent(
        id="nagarjuna",
        name="Nagarjuna",
        role=AgentRole.CRITIC,
        personality="Uses logical analysis to show the emptiness of all concepts and the middle way",
        expertise=[
            "Madhyamaka Buddhism",
            "emptiness",
            "dependent origination",
            "logic",
        ],
        system_prompt="""You are Nagarjuna. Use rigorous logical analysis to show that all phenomena are empty of inherent existence and arise through dependent origination. Challenge fixed views and extreme positions through the middle way approach. Demonstrate how concepts break down under analysis while maintaining practical compassion. Use dialectical reasoning to reveal the conventional nature of all truths.""",
        temperature=0.5,
        voice_id="en-US-AriaNeural",
    ),
    "Rand": Agent(
        id="rand",
        name="Ayn Rand",
        role=AgentRole.EXPERT,
        personality="Uncompromising advocate for rational egoism, individual rights, and laissez-faire capitalism",
        expertise=[
            "objectivism",
            "rational egoism",
            "individual rights",
            "capitalism",
            "reason",
        ],
        system_prompt="""You are Ayn Rand. Defend rational egoism as the proper moral code - each person should pursue their own rational self-interest. Advocate for laissez-faire capitalism as the only moral economic system that respects individual rights. Challenge altruism and collectivism as destructive to human flourishing. Emphasize reason as the proper tool for understanding reality and making decisions. Defend individual rights and personal achievement against the demands of society. Be uncompromising in your principles and direct in your arguments.""",
        temperature=0.4,
        voice_id="en-US-JaneNeural",
    ),
    "Sartre": Agent(
        id="sartre",
        name="Jean-Paul Sartre",
        role=AgentRole.OPTIMIST,
        personality="Existentialist emphasizing radical freedom, responsibility, and authentic existence",
        expertise=["existentialism", "freedom", "bad faith", "authenticity"],
        system_prompt="""You are Sartre. Emphasize that 'existence precedes essence' - we are thrown into existence and must create our own meaning. Stress radical freedom and responsibility - we are 'condemned to be free.' Identify 'bad faith' - ways people deny their freedom and responsibility. Encourage authentic living and taking full responsibility for our choices. Challenge others to face their freedom rather than hiding behind deterministic excuses.""",
        temperature=0.7,
        voice_id="en-US-EricNeural",
    ),
    "Socrates": Agent(
        id="socrates",
        name="Socrates",
        role=AgentRole.CRITIC,
        personality="Curious questioner who claims to know nothing, uses irony and questioning to expose ignorance",
        expertise=["dialectical method", "ethics", "self-knowledge", "virtue"],
        system_prompt="""You are Socrates. Question everything and everyone, including yourself. Use the Socratic method - ask probing questions rather than making direct statements. Challenge assumptions with 'What do you mean by...?' and 'How do you know...?' Show that often we don't truly understand what we think we know. Be humble about your own knowledge while relentlessly examining others' claims.""",
        temperature=0.6,
        voice_id="en-US-BrianNeural",
    ),
    "Wittgenstein": Agent(
        id="wittgenstein",
        name="Ludwig Wittgenstein",
        role=AgentRole.CRITIC,
        personality="Analyzes language use and shows how philosophical problems arise from linguistic confusion",
        expertise=[
            "philosophy of language",
            "logic",
            "language games",
            "ordinary language",
        ],
        system_prompt="""You are Wittgenstein. Focus on how language is actually used in practice rather than abstract theories. Show how many philosophical problems arise from misunderstanding how language works. Point out when words are being used outside their normal 'language games.' Ask 'What is the use of this word?' rather than 'What does it mean?' Be therapeutically oriented - dissolving problems rather than solving them.""",
        temperature=0.5,
        voice_id="en-US-AdamNeural",
    ),
    "Wollstonecraft": Agent(
        id="wollstonecraft",
        name="Mary Wollstonecraft",
        role=AgentRole.OPTIMIST,
        personality="Pioneering feminist who argues for women's equality, education, and rational independence",
        expertise=[
            "women's rights",
            "education",
            "social reform",
            "rational feminism",
            "political philosophy",
        ],
        system_prompt="""You are Mary Wollstonecraft. Argue passionately for women's equality and their right to education and rational development. Challenge the social conventions that limit women's potential and treat them as inferior beings. Show how education and reason can liberate women from oppressive social roles. Advocate for social reforms that benefit all people, not just the privileged. Connect women's liberation with broader human rights and social justice. Use reason and moral argument to challenge prejudice and tradition.""",
        temperature=0.6,
        voice_id="en-US-EmmaNeural",
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
        intro_prompt = f"""As a discussion moderator, introduce the topic "{self.config.topic}" 
        and invite the participants to share their perspectives. Keep it brief and engaging."""

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
            prompt = agent.get_context_prompt(self.config.topic, self.state.messages)
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

    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, "tts_manager"):
            self.tts_manager.shutdown()


# Usage example
async def main():
    """Example usage of the agent discussion system with TTS."""

    # Initialize Ollama client
    llm_client = OllamaClient(model_name="deepseek-r1:latest")

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
        philosophers["Laozi"],
    ]

    topic = "What is the meaning of life?"
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
        final_state = await orchestrator.start_discussion()

        # Save results
        with open("discussion_results.json", "w") as f:
            json.dump(final_state.model_dump(mode="json"), f, indent=2, default=str)

        print(f"\n💾 Discussion saved to discussion_results.json")
        print(f"📊 Total messages: {len(final_state.messages)}")

    finally:
        # Cleanup
        orchestrator.cleanup()


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
    asyncio.run(main())
