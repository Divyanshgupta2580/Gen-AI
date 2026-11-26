"""
Learning modes and pedagogical style definitions for PDF Tutor AI.
"""

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class LearningMode:
    name: str
    icon: str
    description: str
    guidance: str


LEARNING_MODES: Dict[str, LearningMode] = {
    "Explain Concept": LearningMode(
        name="Explain Concept",
        icon="💡",
        description="Core conceptual explanation with definitions and intuition.",
        guidance="Deliver a clear, foundational explanation grounded strictly in the PDF.",
    ),
    "Explain Like I'm a Beginner": LearningMode(
        name="Explain Like I'm a Beginner",
        icon="🌱",
        description="Simplified explanation with everyday analogies for newcomers.",
        guidance="Explain simply without assuming prior expertise. Use friendly analogies.",
    ),
    "Detailed Explanation": LearningMode(
        name="Detailed Explanation",
        icon="🔬",
        description="Comprehensive technical deep-dive with architectural mechanisms.",
        guidance="Provide a rigorous, step-by-step technical explanation covering mechanics and math.",
    ),
    "Summarize": LearningMode(
        name="Summarize",
        icon="📝",
        description="Executive summary and key bullet-point takeaways from the document.",
        guidance="Synthesize the retrieved text into concise, high-yield bullet points.",
    ),
    "Generate Quiz": LearningMode(
        name="Generate Quiz",
        icon="🎯",
        description="Self-assessment practice questions with answers and citations.",
        guidance="Formulate 3-4 grounded test questions with answer explanations.",
    ),
    "Test My Knowledge": LearningMode(
        name="Test My Knowledge",
        icon="❓",
        description="Interactive diagnostic questions for the student to answer.",
        guidance="Ask 1-2 interactive questions based on the PDF to evaluate student comprehension.",
    ),
    "Ask Follow-up Question": LearningMode(
        name="Ask Follow-up Question",
        icon="🔄",
        description="Conversational follow-up extending the previous dialogue.",
        guidance="Maintain continuity with earlier questions while using newly retrieved passages.",
    ),
}

LEARNING_MODE_NAMES: List[str] = list(LEARNING_MODES.keys())
