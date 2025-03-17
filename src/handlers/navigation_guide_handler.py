from typing import List

from src.utils.logger import logger
from schemas import NavigationGuide, ObjectWithDepth
from src.helpers.navigation_helper import (
    sort_objects_by_priority,
    generate_optimized_guidance
)

class NavigationGuideHandler:
    """Handler for creating navigation guidance from objects and depth information using Priority Score method"""
    
    def __init__(self):
        """Initialize navigation guide handler"""
        logger.info("Initializing Navigation Guide Handler with Priority Score")
        
    def _get_warning_level(self, priority_score: float) -> str:
        """
        Determine warning level based on Priority Score
        
        Args:
            priority_score (float): Priority score (0-1)
            
        Returns:
            str: Warning level (High/Medium/None)
        """
        if priority_score > 0.7:
            return "High"
        elif priority_score > 0.3:
            return "Medium"
        else:
            return "None"
            
    async def generate_navigation_guide(self, objects_with_depth: List[ObjectWithDepth]) -> NavigationGuide:
        """
        Generate navigation guidance using Priority Score method
        
        Args:
            objects_with_depth (List[ObjectWithDepth]): List of objects with depth information
                
        Returns:
            NavigationGuide: Navigation guidance information
        """
        try:
            if not objects_with_depth:
                return NavigationGuide(
                    navigation_text="No objects detected, the path ahead is clear.",
                    priority_objects=[]
                )
            
            # Sort objects by priority using Priority Score
            sorted_objects = sort_objects_by_priority(objects_with_depth)
            
            # Get the most important objects (maximum 3)
            important_objects = sorted_objects[:3]
            
            # Generate optimized guidance text (one sentence)
            navigation_text = generate_optimized_guidance(important_objects)
            
            priority_objects_dict = [obj.model_dump() for obj in important_objects]

            return NavigationGuide(
                navigation_text=navigation_text,
                priority_objects=priority_objects_dict
            )
                
        except Exception as e:
            logger.error(f"Error generating navigation guidance: {str(e)}")
            return NavigationGuide(
                navigation_text="Unable to generate accurate guidance. Please move carefully.",
                priority_objects=[]
            )