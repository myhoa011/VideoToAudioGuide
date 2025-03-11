from src.utils.logger import logger

def sort_objects_by_priority(objects: list) -> list:
    """
    Sort objects list by priority based on depth and position
    
    Sorting criteria:
    1. depth (higher value means closer)
    2. Position (center has higher priority than left/right)
    
    Args:
        objects (list): List of objects with depth information
        
    Returns:
        list: Sorted list of objects
    """
    try:
        # Assign priority scores to positions
        position_scores = {
            'center': 3,
            'bottom center': 2.5,
            'top center': 2,
            'left': 1.5,
            'right': 1.5,
            'bottom left': 1,
            'bottom right': 1,
            'top left': 1,
            'top right': 1,
        }
        
        # Define priority score calculation function
        def get_priority_score(obj):
            # Priority 1: Use depth (higher depth means closer)
            depth_score = obj.get('depth', 0)
                
            # Priority 2: Object position
            position = obj.get('position', 'center')
            pos_score = position_scores.get(position, 0)
            
            # Calculate composite score (weights can be adjusted)
            # Depth has higher weight (70%) as it's the primary factor
            # Position has lower weight (30%) as supplementary information
            total_score = (depth_score * 0.7) + (pos_score * 0.3)
            
            return total_score
        
        # Sort objects by priority score in descending order
        sorted_objects = sorted(objects, key=get_priority_score, reverse=True)
        
        return sorted_objects
        
    except Exception as e:
        logger.error(f"Lỗi khi sắp xếp đối tượng theo ưu tiên: {str(e)}")
        return objects

def convert_depth_to_distance_text(depth: float) -> str:
    """
    Convert depth value to distance description
    
    Args:
        depth (float): Depth value (0-1, closer to 1 means closer)
        
    Returns:
        str: Distance description
    """
    if depth > 0.7:
        return "rất gần"
    elif depth > 0.5:
        return "khá gần"
    elif depth > 0.3:
        return "khoảng cách trung bình"
    elif depth > 0.1:
        return "khá xa"
    else:
        return "rất xa"


def generate_direction_guidance(obj: dict) -> str:
    """
    Generate movement guidance based on object information
    
    Args:
        obj (dict): Object information
        
    Returns:
        str: Movement guidance
    """
    label = obj.get('label', 'đối tượng')
    position = obj.get('position', 'phía trước')
    depth = obj.get('depth', 0)
    
    # Convert depth to distance description
    distance_text = convert_depth_to_distance_text(depth)
    
    # Generate guidance based on position and distance
    if 'center' in position:
        if depth > 0.5:  # Very close or fairly close
            return f"Chú ý! Có {label} ở ngay phía trước, cách bạn {distance_text}. Hãy đổi hướng để tránh va chạm"
        else:
            return f"Có {label} ở phía trước, cách bạn {distance_text}. Hãy chuẩn bị đổi hướng"
            
    elif 'left' in position:
        if depth > 0.5:  # Very close or fairly close
            return f"Chú ý! Có {label} ở bên trái, cách bạn {distance_text}. Hãy đi sang phải"
        else:
            return f"Có {label} ở bên trái, cách bạn {distance_text}"
            
    elif 'right' in position:
        if depth > 0.5:  # Very close or fairly close
            return f"Chú ý! Có {label} ở bên phải, cách bạn {distance_text}. Hãy đi sang trái"
        else:
            return f"Có {label} ở bên phải, cách bạn {distance_text}"
    
    else:
        return f"Phát hiện {label} ở {position}, cách bạn {distance_text}" 