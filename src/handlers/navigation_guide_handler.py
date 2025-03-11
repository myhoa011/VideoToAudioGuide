from src.utils.logger import logger
from src.helpers.navigation_helper import (
    sort_objects_by_priority,
    convert_depth_to_distance_text,
    generate_direction_guidance
)

class NavigationGuideHandler:
    """Handler for creating navigation guidance from objects and depth information"""
    
    def __init__(self):
        """Initialize navigation guide handler"""
        logger.info("Khởi tạo Navigation Guide Handler")
        
    async def generate_navigation_guide(self, objects_with_depth: list) -> dict:
        """
        Generate navigation guidance from a list of objects with depth information
        
        Args:
            objects_with_depth (list): List of objects with depth information
                [{'box_2d': [x1, y1, x2, y2], 'label': 'person', 'position': 'center', 
                  'depth': 0.17, 'distance_rank': 8}, ...]
            
        Returns:
            dict: Navigation guidance information
        """
        try:
            if not objects_with_depth:
                return {"navigation_text": "Không phát hiện đối tượng, đường đi phía trước thông thoáng."}
            
            # Sort objects by priority
            sorted_objects = sort_objects_by_priority(objects_with_depth)
            
            # Get the most important objects (maximum 3)
            important_objects = sorted_objects[:3]
            
            # Generate main guidance for the highest priority object
            primary_object = important_objects[0]
            primary_guidance = generate_direction_guidance(primary_object)
            
            # Create comprehensive guidance text
            navigation_text = primary_guidance
            
            # Add information about other objects if available
            if len(important_objects) > 1:
                additional_info = []
                
                for obj in important_objects[1:]:
                    label = obj.get('label', 'đối tượng')
                    position = obj.get('position', 'phía trước')
                    depth = obj.get('depth', 0)
                    distance_text = convert_depth_to_distance_text(depth)
                    
                    info = f"Cũng có {label} ở {position}, cách bạn {distance_text}"
                    additional_info.append(info)
                
                if additional_info:
                    navigation_text += ". " + ". ".join(additional_info)
            
            return {
                "navigation_text": navigation_text,
                "priority_objects": important_objects
            }
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo hướng dẫn điều hướng: {str(e)}")
            return {"navigation_text": "Không thể tạo hướng dẫn chính xác. Hãy di chuyển cẩn thận."} 