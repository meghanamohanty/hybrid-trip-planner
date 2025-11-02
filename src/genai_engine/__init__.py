"""
Generative AI Engine Module
===========================

Integrates Google Gemini AI for intelligent reasoning and natural language processing:
- Structured prompt engineering for itinerary generation
- Constraint satisfaction using AI reasoning capabilities
- Natural language explanations for ML recommendations
- Context-aware itinerary synthesis and optimization
- Human-readable trip summaries and descriptions

Classes:
    GeminiReasoner: Main interface to Google Gemini AI for reasoning tasks
    PromptEngineer: Constructs structured prompts for different AI tasks
    ConstraintValidator: Uses AI to validate and resolve planning constraints
    ItinerarySynthesizer: Combines ML outputs with AI reasoning for final itinerary
    ExplanationGenerator: Creates human-readable explanations for recommendations
    ContextProcessor: Handles contextual information for AI decision making

Functions:
    generate_itinerary(): Main function to create AI-enhanced itineraries
    explain_recommendations(): Generate explanations for POI selections
    resolve_constraints(): Use AI to resolve conflicting constraints
    create_trip_summary(): Generate natural language trip descriptions
    validate_itinerary(): AI-powered validation of generated plans
"""

# Version and module info
__version__ = "1.0.0"
__module_name__ = "genai_engine"

# Import core GenAI classes (will be created later)
from .gemini_reasoner import GeminiReasoner
from .prompt_engineer import PromptEngineer
from .constraint_validator import ConstraintValidator
from .itinerary_synthesizer import ItinerarySynthesizer
from .explanation_generator import ExplanationGenerator
from .context_processor import ContextProcessor

# Import AI algorithms and utilities
from .ai_algorithms import (
    constraint_reasoning,
    contextual_optimization,
    natural_language_synthesis
)

# Import utility functions
from .utils import (
    generate_itinerary,
    explain_recommendations,
    resolve_constraints,
    create_trip_summary,
    validate_itinerary,
    format_ai_response
)

# Define public API
__all__ = [
    # Core GenAI classes
    "GeminiReasoner",
    "PromptEngineer",
    "ConstraintValidator", 
    "ItinerarySynthesizer",
    "ExplanationGenerator",
    "ContextProcessor",
    
    # AI algorithm functions
    "constraint_reasoning",
    "contextual_optimization",
    "natural_language_synthesis",
    
    # Utility functions
    "generate_itinerary",
    "explain_recommendations",
    "resolve_constraints",
    "create_trip_summary",
    "validate_itinerary",
    "format_ai_response"
]

def get_prompt_templates():
    """
    Return standard prompt templates for different AI tasks
    """
    return {
        "itinerary_generation": {
            "system": "You are an expert travel planner who creates detailed, feasible itineraries.",
            "user_template": "Create a {days}-day itinerary for {city} with budget {budget} {currency}. Consider: {constraints}"
        },
        "constraint_resolution": {
            "system": "You are a constraint solver for travel planning. Resolve conflicts logically.",
            "user_template": "Resolve these planning conflicts: {conflicts}. Constraints: {constraints}"
        },
        "explanation": {
            "system": "You explain travel recommendations in a friendly, informative way.",
            "user_template": "Explain why these POIs were recommended: {pois}. User preferences: {preferences}"
        },
        "validation": {
            "system": "You validate travel itineraries for feasibility and logic.",
            "user_template": "Validate this itinerary: {itinerary}. Check for timing, budget, and logic issues."
        }
    }

def get_ai_model_config():
    """
    Return Gemini AI model configuration parameters
    """
    return {
        "model_name": "gemini-2.0-flash-exp",
        "temperature": 0.7,          # Balance creativity and consistency
        "max_output_tokens": 2048,   # Sufficient for detailed itineraries
        "top_p": 0.9,               # Nucleus sampling parameter
        "top_k": 40,                # Top-k sampling parameter
        "safety_settings": {
            "harassment": "BLOCK_MEDIUM_AND_ABOVE",
            "hate_speech": "BLOCK_MEDIUM_AND_ABOVE", 
            "sexually_explicit": "BLOCK_MEDIUM_AND_ABOVE",
            "dangerous_content": "BLOCK_MEDIUM_AND_ABOVE"
        }
    }

def get_reasoning_capabilities():
    """
    Return list of AI reasoning capabilities available
    """
    return {
        "constraint_solving": "Resolve conflicts between budget, time, and preferences",
        "contextual_reasoning": "Consider weather, local events, and cultural factors",
        "natural_language": "Generate human-readable explanations and summaries",
        "itinerary_synthesis": "Combine ML outputs with logical reasoning",
        "validation": "Check feasibility and logic of generated plans",
        "personalization": "Adapt recommendations to user preferences and constraints"
    }

def get_genai_engine_info():
    """
    Return comprehensive information about GenAI engine capabilities
    """
    return {
        "module": __module_name__,
        "version": __version__,
        "ai_model": "Google Gemini 2.0 Flash",
        "capabilities": get_reasoning_capabilities(),
        "features": [
            "Structured prompt engineering for travel planning",
            "Constraint satisfaction using AI reasoning",
            "Natural language explanations for recommendations", 
            "Context-aware itinerary synthesis",
            "AI-powered validation and optimization",
            "Human-readable trip summaries and descriptions"
        ],
        "prompt_templates": list(get_prompt_templates().keys()),
        "model_config": get_ai_model_config()
    }

# Module-level constants
DEFAULT_TEMPERATURE = 0.7
MAX_RETRY_ATTEMPTS = 3
DEFAULT_TIMEOUT_SECONDS = 30
MIN_RESPONSE_LENGTH = 50
MAX_PROMPT_LENGTH = 4000

# AI reasoning modes
REASONING_MODES = {
    "conservative": {"temperature": 0.3, "top_p": 0.8},
    "balanced": {"temperature": 0.7, "top_p": 0.9}, 
    "creative": {"temperature": 0.9, "top_p": 0.95}
}