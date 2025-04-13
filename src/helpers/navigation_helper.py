from typing import List

from src.utils.logger import logger
from src.schemas import ObjectWithDepth
from collections import defaultdict
from src.utils.constant import (
    PRIORITY_DEPTH_WEIGHT, PRIORITY_POSITION_WEIGHT, PRIORITY_SIZE_WEIGHT, PRIORITY_TYPE_WEIGHT,
    DISTANCE_CLOSE_THRESHOLD, DISTANCE_MEDIUM_THRESHOLD,
    FRAME_NORMALIZED_WIDTH, FRAME_NORMALIZED_AREA,
    HIGH_RISK_OBJECTS, MEDIUM_RISK_OBJECTS, LOW_RISK_OBJECTS, MINIMAL_RISK_OBJECTS
)

def get_type_score(obj_type: str) -> float:
    """Calculate priority score based on object type"""
    obj_type = obj_type.lower()
    
    # Không ưu tiên các đối tượng không liên quan đến điều hướng
    if obj_type in ['geographical feature', 'atmospheric', 'body of water']:
        return 0.0
    
    # High risk - highest priority
    if obj_type in HIGH_RISK_OBJECTS:
        return 1.0
        
    # Medium risk
    if obj_type in MEDIUM_RISK_OBJECTS:
        return 0.7
        
    # Low risk
    if obj_type in LOW_RISK_OBJECTS:
        return 0.4
        
    # Static objects, minimal risk
    if obj_type in MINIMAL_RISK_OBJECTS:
        return 0.2
        
    # Default for undefined objects
    return 0.1

def get_priority_score(obj: ObjectWithDepth) -> float:
    """
    Calculate priority score for an object
    
    Args:
        obj (ObjectWithDepth): Object with depth information
        
    Returns:
        float: Priority score (0-1)
    """
    try:
        # Get depth score (already 0-1)
        depth_score = obj.depth
        
        # Calculate position score
        y_min, x_min, y_max, x_max = obj.box_2d
        
        # Calculate center point (by x)
        x_center = (x_min + x_max) / 2
        frame_width = FRAME_NORMALIZED_WIDTH
        frame_center = frame_width / 2
        position_score = abs(x_center - frame_center) / frame_width
        
        # Calculate size score
        width = x_max - x_min
        height = y_max - y_min
        size_score = (width * height) / FRAME_NORMALIZED_AREA
        
        # Calculate type score
        type_score = get_type_score(obj.type)
        
        # Define weights
        w1 = PRIORITY_DEPTH_WEIGHT
        w2 = PRIORITY_POSITION_WEIGHT
        w3 = PRIORITY_SIZE_WEIGHT
        w4 = PRIORITY_TYPE_WEIGHT
        
        # Calculate priority score
        priority_score = (w1 * depth_score) + (w2 * (1 - position_score)) + \
                       (w3 * size_score) + (w4 * type_score)
        
        return priority_score
        
    except Exception as e:
        logger.error(f"Error calculating priority score: {str(e)}")
        return 0.0

def sort_objects_by_priority(objects: List[ObjectWithDepth]) -> List[ObjectWithDepth]:
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
        objects (List[ObjectWithDepth]): List of objects with depth information
        
    Returns:
        List[ObjectWithDepth]: Sorted list of objects
    """
    try:
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
    if depth > DISTANCE_CLOSE_THRESHOLD:
        return "very close"
    elif depth > DISTANCE_MEDIUM_THRESHOLD:
        return "quite close"
    else:
        return "far away"

def should_add_warning(warning_levels: List[str]) -> bool:
    """
    Determine if warning prefix should be added based on warning levels
    
    Args:
        warning_levels (List[str]): List of warning levels for objects
        
    Returns:
        bool: True if warning should be added
    """
    # Add warning if any object has "High" warning level
    return "High" in warning_levels

def generate_optimized_guidance(important_objects: List[ObjectWithDepth], warning_levels: List[str]) -> str:
    """
    Generate optimized guidance text by combining all objects into a single concise sentence
    
    Args:
        important_objects (List[ObjectWithDepth]): List of priority objects
        warning_levels (List[str]): Warning levels for each object
        
    Returns:
        str: Optimized guidance text as a single sentence
    """
    if not important_objects:
        return "No objects detected, the path ahead is clear."
        
    # Group objects by position and label
    position_objects = defaultdict(lambda: defaultdict(int))
    position_depth = defaultdict(lambda: defaultdict(list))
    
    for obj in important_objects:
        label = obj.label
        position = obj.position
        depth = obj.depth
        
        # Count objects by label at each position
        position_objects[position][label] += 1
        # Track depths for each object type at each position
        position_depth[position][label].append(depth)
    
    # Start building the sentence
    sentence_parts = []
    
    # Add warning prefix based on warning levels 
    if should_add_warning(warning_levels):
        sentence_parts.append("Warning!")
    
    # Process each position (center, left, right)
    positions_text = []
    
    # Order of positions for natural language: center, left, right
    for position in ['center', 'left', 'right']:
        if position not in position_objects:
            continue
            
        # Format objects at this position
        object_texts = []
        
        for label, count in position_objects[position].items():
            # Calculate average depth for this object type
            avg_depth = sum(position_depth[position][label]) / len(position_depth[position][label])
            distance = convert_depth_to_distance_text(avg_depth)
            
            # Format text based on count
            if count == 1:
                object_texts.append(f"a {label} ({distance})")
            else:
                object_texts.append(f"{count} {label}s ({distance})")
        
        # Format the position description
        if position == 'center':
            position_text = "directly ahead"
        else:
            position_text = f"to the {position}"
            
        # Combine objects at this position
        if len(object_texts) == 1:
            positions_text.append(f"{object_texts[0]} {position_text}")
        else:
            combined = ", ".join(object_texts[:-1]) + f" and {object_texts[-1]}"
            positions_text.append(f"{combined} {position_text}")
    
    # Combine all positions into one sentence
    if len(positions_text) == 1:
        sentence_parts.append(f"There is {positions_text[0]}.")
    elif len(positions_text) == 2:
        sentence_parts.append(f"There are {positions_text[0]} and {positions_text[1]}.")
    else:
        sentence_parts.append(f"There are {', '.join(positions_text[:-1])}, and {positions_text[-1]}.")
    
    return " ".join(sentence_parts)

def calculate_object_size(box_2d):
    """
    Calculate object size from normalized coordinates
    
    Args:
        box_2d: Coordinates [y_min, x_min, y_max, x_max] normalized (0-1000)
    
    Returns:
        float: Relative area of the object (0-1)
    """
    y_min, x_min, y_max, x_max = box_2d
    
    # Calculate width and height correctly based on [y_min, x_min, y_max, x_max] format
    width = x_max - x_min  # Width = x_max - x_min
    height = y_max - y_min  # Height = y_max - y_min
    
    # Calculate relative area (on 1000x1000 scale)
    area = width * height
    total_area = 1000 * 1000  # Total normalized area
    
    return area / total_area