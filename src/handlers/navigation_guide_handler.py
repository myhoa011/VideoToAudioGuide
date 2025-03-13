from src.utils.logger import logger
from src.helpers.navigation_helper import (
    sort_objects_by_priority,
    generate_direction_guidance
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
            
    async def generate_navigation_guide(self, objects_with_depth: list) -> dict:
        """
        Generate navigation guidance using Priority Score method
        
        Args:
            objects_with_depth (list): List of objects with depth information
                [{'box_2d': [x1, y1, x2, y2], 'label': 'person', 'position': 'center',
                  'type': 'person', 'depth': 0.17, 'distance_rank': 8}, ...]
            
        Returns:
            dict: Navigation guidance information
        """
        try:
            if not objects_with_depth:
                return {"navigation_text": "No objects detected, the path ahead is clear."}
            
            # Sort objects by priority using Priority Score
            sorted_objects = sort_objects_by_priority(objects_with_depth)
            
            # Get the most important objects (maximum 3)
            important_objects = sorted_objects[:3]
            
            # Generate guidance for primary object using helper function
            primary_guidance = generate_direction_guidance(important_objects[0])
            
            # Create comprehensive guidance text
            navigation_text = primary_guidance
            
            # Add information about other objects if available
            if len(important_objects) > 1:
                additional_info = []
                for obj in important_objects[1:]:
                    additional_info.append(generate_direction_guidance(obj))
                
                if additional_info:
                    navigation_text += ". " + ". ".join(additional_info)
            
            return {
                "navigation_text": navigation_text,
                "priority_objects": important_objects
            }
            
        except Exception as e:
            logger.error(f"Error generating navigation guidance: {str(e)}")
            return {"navigation_text": "Unable to generate accurate guidance. Please move carefully."}