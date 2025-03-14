from src.utils.logger import logger

def sort_objects_by_priority(objects: list) -> list:
    """
    Sort objects list by priority using Priority Score method
    
    Priority Score formula:
    P = w1*d + w2*(x - xcenter)/W + w3*Area/TotalArea + w4*TypeScore
    
    Where:
    - d: depth (0-1)
    - x: x-coordinate of object center
    - xcenter: x-coordinate of frame center
    - W: frame width
    - Area: object area
    - TotalArea: total frame area
    - TypeScore: priority score based on object type (0-1)
    
    Args:
        objects (list): List of objects with depth information
        
    Returns:
        list: Sorted list of objects
    """
    try:
        # Define weights
        w1 = 0.5  # depth weight
        w2 = 0.2  # position weight
        w3 = 0.1  # size weight
        w4 = 0.2  # type weight
        
        def get_type_score(label: str) -> float:
            """Calculate priority score based on object type"""
            label = label.lower()
            
            # High risk - highest priority
            if label in ['person', 'car', 'motorcycle', 'truck', 'bus']:
                return 1.0
                
            # Medium risk
            if label in ['bicycle', 'dog', 'pothole', 'stairs']:
                return 0.7
                
            # Low risk
            if label in ['traffic_light', 'stop_sign', 'door']:
                return 0.4
                
            # Static objects, minimal risk
            if label in ['bench', 'wall', 'tree']:
                return 0.2
                
            # Default for undefined objects
            return 0.1
        
        def get_priority_score(obj):
            # Get depth score (already 0-1)
            depth_score = obj.get('depth')
            
            # Calculate position score
            box = obj.get('box_2d', [0, 0, 0, 0])
            x_center = (box[0] + box[2]) / 2  # object center x
            frame_width = 1.0  # normalized width
            frame_center = frame_width / 2
            position_score = abs(x_center - frame_center) / frame_width
            
            # Calculate size score
            box_width = box[2] - box[0]
            box_height = box[3] - box[1]
            area = box_width * box_height
            total_area = 1.0  # normalized area
            size_score = area / total_area
            
            # Calculate type score
            type_score = get_type_score(obj.get('type'))
            
            # Calculate priority score
            priority_score = (w1 * depth_score) + (w2 * (1 - position_score)) + \
                           (w3 * size_score) + (w4 * type_score)
            
            return priority_score
        
        # Sort objects by priority score in descending order
        sorted_objects = sorted(objects, key=get_priority_score, reverse=True)
        
        return sorted_objects
        
    except Exception as e:
        logger.error(f"Error sorting objects by priority: {str(e)}")
        return objects

def convert_depth_to_distance_text(depth: float) -> str:
    """
    Convert depth value to distance description based on Priority Score thresholds
    
    Args:
        depth (float): Depth value (0-1, closer to 1 means closer)
        
    Returns:
        str: Distance description
    """
    if depth > 0.7:  # Strong warning threshold
        return "very close"
    elif depth > 0.3:  # Mild warning threshold
        return "quite close"
    else:  # No warning needed
        return "far away"


def generate_direction_guidance(obj: dict) -> str:
    """
    Generate movement guidance based on object information
    
    Args:
        obj (dict): Object information
        
    Returns:
        str: Movement guidance
    """
    label = obj.get('label')
    position = obj.get('position')
    depth = obj.get('depth')
    
    # Convert depth to distance description
    distance_text = convert_depth_to_distance_text(depth)
    
    # Generate guidance based on position and distance
    if 'center' in position:
        if depth > 0.5:  # Very close or fairly close
            return f"Warning! There is a {label} right in front of you, {distance_text}."
        else:
            return f"There is a {label} in front of you, {distance_text}."
            
    elif 'left' in position:
        if depth > 0.5:  # Very close or fairly close
            return f"Warning! There is a {label} on the left, {distance_text}."
        else:
            return f"There is a {label} on the left, {distance_text}."
            
    elif 'right' in position:
        if depth > 0.5:  # Very close or fairly close
            return f"Warning! There is a {label} on the right, {distance_text}."
        else:
            return f"There is a {label} on the right, {distance_text}."
    
    else:
        return f"Detected a {label} at {position}, {distance_text}."


def consolidate_navigation_guidance(objects_guidance: list) -> str:
    """
    Consolidate multiple object guidance into single coherent navigation text
    
    Args:
        objects_guidance (list): List of guidance texts for individual objects
        
    Returns:
        str: Consolidated navigation text
    """
    if not objects_guidance:
        return "Path is clear. No obstacles detected."
        
    # Group similar messages to avoid redundancy
    message_groups = {}
    
    for guidance in objects_guidance:
        # Create a simplified key for grouping
        # For example: "Warning! There is a person on the left, very close."
        # Key would be: "person_left_very close"
        
        parts = guidance.lower().replace("warning! ", "").replace("there is a ", "").replace(".", "").split()
        if len(parts) >= 4:
            # Extract key elements: object type, position, distance
            obj_type = parts[0]  # e.g., "person"
            position = None
            
            # Be more precise about position detection
            if "left" in guidance.lower():
                position = "left"
            elif "right" in guidance.lower():
                position = "right"
            elif "front" in guidance.lower() or "center" in guidance.lower():
                position = "center"
            else:
                position = "unknown"
                
            # Extract distance info
            distance = "close"
            if "very close" in guidance.lower():
                distance = "very close"
            elif "quite close" in guidance.lower():
                distance = "quite close"
            elif "far away" in guidance.lower():
                distance = "far away"
            
            # Create a key including position to prevent mixing different positions
            key = f"{obj_type}_{position}_{distance}"
            
            if key in message_groups:
                message_groups[key]["count"] += 1
            else:
                message_groups[key] = {
                    "count": 1,
                    "guidance": guidance,
                    "is_warning": guidance.startswith("Warning!"),
                    "distance_priority": 2 if "very close" in guidance else 1 if "quite close" in guidance else 0,
                    "position": position
                }
    
    # Sort by priority: warnings first, then by distance
    sorted_messages = sorted(
        message_groups.values(),
        key=lambda x: (0 if x["is_warning"] else 1, -x["distance_priority"], -x["count"])
    )
    
    # Generate consolidated text
    consolidated_texts = []
    
    for msg in sorted_messages:
        if msg["count"] > 1:
            # Replace singular with plural form
            guidance = msg["guidance"]
            guidance = guidance.replace("There is a ", f"There are {msg['count']} ")
            guidance = guidance.replace("Warning! There is a ", f"Warning! There are {msg['count']} ")
            
            # Handle plural forms of common objects
            for singular, plural in [
                ("person", "people"), 
                ("child", "children"),
                ("man", "men"),
                ("woman", "women")
            ]:
                if f" {singular}" in guidance:
                    guidance = guidance.replace(f" {singular}", f" {plural}")
            
            consolidated_texts.append(guidance)
        else:
            consolidated_texts.append(msg["guidance"])
    
    # Join with proper spacing and use commas correctly
    return " ".join(consolidated_texts)
