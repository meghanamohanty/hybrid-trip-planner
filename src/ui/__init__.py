"""
User Interface Module
====================

Streamlit-based user interface for the trip planner application:
- Interactive input forms for trip preferences and constraints
- Real-time preview and customization of generated itineraries
- Map visualizations showing POI locations and daily clusters
- Export functionality for JSON and PDF itinerary formats
- User feedback collection and preference learning

Classes:
    StreamlitApp: Main application controller and page router
    InputForm: Handles user input collection (budget, days, preferences)
    ItineraryViewer: Displays generated itineraries with interactive features
    MapVisualizer: Creates interactive maps showing POIs and routes
    ExportManager: Handles PDF and JSON export functionality
    FeedbackCollector: Collects user ratings and preferences

Components:
    TripInputPage: Page 1 - Trip input form and initial preferences
    PreviewPage: Page 2 - Preview POI clusters and make adjustments
    ItineraryPage: Page 3 - Final itinerary display with timeline
    ExportPage: Page 4 - Download and export options
    SettingsPage: User preferences and application settings

Functions:
    run_app(): Main function to start the Streamlit application
    render_input_form(): Create trip input form with validation
    display_itinerary(): Show formatted itinerary with timeline
    create_map_visualization(): Generate interactive maps
    export_to_pdf(): Generate PDF version of itinerary
    export_to_json(): Export itinerary data as JSON
"""

# Version and module info
__version__ = "1.0.0"
__module_name__ = "ui"

# Import main UI classes (will be created later)
from .streamlit_app import StreamlitApp
from .input_form import InputForm
from .itinerary_viewer import ItineraryViewer
from .map_visualizer import MapVisualizer
from .export_manager import ExportManager
from .feedback_collector import FeedbackCollector

# Import page components
from .pages import (
    TripInputPage,
    PreviewPage, 
    ItineraryPage,
    ExportPage,
    SettingsPage
)

# Import UI utilities and widgets
from .components import (
    BudgetSlider,
    InterestSelector,
    WeatherDisplay,
    POICard,
    TimelineView,
    CostBreakdown
)

# Import utility functions
from .utils import (
    run_app,
    render_input_form,
    display_itinerary,
    create_map_visualization,
    export_to_pdf,
    export_to_json,
    validate_user_input,
    format_currency
)

# Define public API
__all__ = [
    # Main UI classes
    "StreamlitApp",
    "InputForm",
    "ItineraryViewer",
    "MapVisualizer", 
    "ExportManager",
    "FeedbackCollector",
    
    # Page components
    "TripInputPage",
    "PreviewPage",
    "ItineraryPage", 
    "ExportPage",
    "SettingsPage",
    
    # UI widgets
    "BudgetSlider",
    "InterestSelector",
    "WeatherDisplay",
    "POICard",
    "TimelineView",
    "CostBreakdown",
    
    # Utility functions
    "run_app",
    "render_input_form",
    "display_itinerary",
    "create_map_visualization",
    "export_to_pdf",
    "export_to_json",
    "validate_user_input",
    "format_currency"
]

def get_page_config():
    """
    Return Streamlit page configuration settings
    """
    return {
        "page_title": "Hybrid Trip Planner",
        "page_icon": "✈️",
        "layout": "wide",
        "initial_sidebar_state": "expanded",
        "menu_items": {
            "Get Help": "https://github.com/your-repo/issues",
            "Report a Bug": "https://github.com/your-repo/issues/new",
            "About": "Hybrid ML + GenAI Context-Aware Itinerary Planner v1.0.0"
        }
    }

def get_ui_themes():
    """
    Return available UI theme configurations
    """
    return {
        "default": {
            "primary_color": "#FF6B6B",
            "background_color": "#FFFFFF", 
            "secondary_background_color": "#F0F2F6",
            "text_color": "#262730"
        },
        "dark": {
            "primary_color": "#FF6B6B",
            "background_color": "#0E1117",
            "secondary_background_color": "#262730", 
            "text_color": "#FAFAFA"
        },
        "travel": {
            "primary_color": "#1E88E5",
            "background_color": "#FFFFFF",
            "secondary_background_color": "#E3F2FD",
            "text_color": "#1565C0"
        }
    }

def get_input_validation_rules():
    """
    Return validation rules for user inputs
    """
    return {
        "budget": {
            "min": 1000,      # Minimum budget in INR
            "max": 500000,    # Maximum budget in INR
            "step": 500       # Budget increment step
        },
        "days": {
            "min": 1,         # Minimum trip days
            "max": 14,        # Maximum trip days
            "default": 3      # Default trip length
        },
        "city": {
            "min_length": 2,  # Minimum city name length
            "max_length": 50, # Maximum city name length
            "required": True  # City input is required
        },
        "interests": {
            "min_selection": 1,   # Minimum interests to select
            "max_selection": 8,   # Maximum interests to select
            "available": [
                "Culture & History", "Food & Dining", "Nature & Parks",
                "Adventure & Sports", "Shopping", "Nightlife", 
                "Museums & Art", "Religious Sites", "Architecture",
                "Local Markets", "Beaches", "Mountains"
            ]
        }
    }

def get_export_formats():
    """
    Return available export formats and their configurations
    """
    return {
        "pdf": {
            "format": "PDF",
            "extension": ".pdf",
            "mime_type": "application/pdf",
            "description": "Formatted PDF document with maps and timeline"
        },
        "json": {
            "format": "JSON", 
            "extension": ".json",
            "mime_type": "application/json",
            "description": "Structured data format for developers"
        },
        "csv": {
            "format": "CSV",
            "extension": ".csv", 
            "mime_type": "text/csv",
            "description": "Spreadsheet-compatible format"
        }
    }

def get_ui_info():
    """
    Return comprehensive information about UI capabilities
    """
    return {
        "module": __module_name__,
        "version": __version__,
        "framework": "Streamlit",
        "features": [
            "Interactive trip input forms with real-time validation",
            "Preview mode with POI clustering visualization",
            "Timeline-based itinerary display with cost breakdown",
            "Interactive maps showing locations and routes",
            "Multi-format export (PDF, JSON, CSV)",
            "User feedback collection and preference learning",
            "Responsive design with mobile support",
            "Dark/light theme support"
        ],
        "pages": [
            "Trip Input - Budget, days, city, preferences",
            "Preview - POI clusters and customization options", 
            "Itinerary - Final timeline with maps and costs",
            "Export - Download options and sharing",
            "Settings - User preferences and app configuration"
        ],
        "validation_rules": get_input_validation_rules(),
        "export_formats": list(get_export_formats().keys()),
        "themes": list(get_ui_themes().keys())
    }

# Module-level constants
DEFAULT_THEME = "default"
DEFAULT_PAGE = "Trip Input"
MAX_FILE_SIZE_MB = 10
SESSION_TIMEOUT_MINUTES = 30

# UI component settings
CARD_HEIGHT = 200
MAP_HEIGHT = 400
SIDEBAR_WIDTH = 300
MAIN_CONTENT_WIDTH = 800

# Animation and interaction settings
LOADING_SPINNER_DELAY = 1
TOAST_DURATION = 3
PROGRESS_BAR_STEPS = 5