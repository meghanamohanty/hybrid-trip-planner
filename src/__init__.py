"""
Hybrid ML + GenAI Context-Aware Itinerary Planner
=================================================

A comprehensive travel planning system that combines machine learning 
and generative AI to create personalized, context-aware trip itineraries.

Author: Hybrid Trip Planner Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Hybrid Trip Planner Team"

def get_version():
    """Return the current version of the application"""
    return __version__

def get_info():
    """Return basic information about the application"""
    return {
        "name": "Hybrid ML + GenAI Trip Planner",
        "version": __version__,
        "author": __author__,
        "description": "Context-aware itinerary planner using ML and GenAI"
    }