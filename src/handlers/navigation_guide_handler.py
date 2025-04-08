from typing import List, Tuple

from src.utils.logger import logger
from schemas import NavigationGuide, ObjectWithDepth
from src.helpers.navigation_helper import (
    sort_objects_by_priority,
    generate_optimized_guidance,
    get_priority_score
)
from src.utils.constant import WARNING_HIGH_THRESHOLD, WARNING_MEDIUM_THRESHOLD, WARNING_THRESHOLD

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
        if priority_score > WARNING_HIGH_THRESHOLD:
            return "High" 
        elif priority_score > WARNING_MEDIUM_THRESHOLD:
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
            
            # Filter out objects below the minimum threshold
            filtered_objects = []
            warning_levels = []
            
            for obj in important_objects:
                # Calculate priority score for the object
                p_score = get_priority_score(obj)
                
                # Skip objects with low priority score
                if p_score <= WARNING_THRESHOLD:
                    continue
                
                # Add object to filtered list
                filtered_objects.append(obj)
                
                # Determine warning level
                warning_level = self._get_warning_level(p_score)
                warning_levels.append(warning_level)
            
            # Check if we have any important objects left
            if not filtered_objects:
                return NavigationGuide(
                    navigation_text="No significant obstacles detected, proceed with caution.",
                    priority_objects=[]
                )
            
            # Generate optimized guidance text with warning level
            navigation_text = generate_optimized_guidance(filtered_objects, warning_levels)
            
            priority_objects_dict = [obj.model_dump() for obj in filtered_objects]

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