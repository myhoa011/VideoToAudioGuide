import os
import csv
from datetime import datetime
from typing import List
from src.utils.logger import logger
from src.schemas.response import ExecutionTime

def save_execution_time_to_csv(execution_times: List[ExecutionTime], video_folder: str = None):
    """
    Save execution time data to a CSV file in the outputs directory
    
    Args:
        execution_times (List[ExecutionTime]): List of execution time objects
        video_folder (str, optional): Video folder name for grouping reports
    """
    try:
        # Create reports directory if it doesn't exist
        reports_dir = os.path.join("outputs", "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if video_folder:
            # Create subfolder for this video if specified
            video_reports_dir = os.path.join(reports_dir, video_folder)
            os.makedirs(video_reports_dir, exist_ok=True)
            filename = os.path.join(video_reports_dir, f"execution_time_{timestamp}.csv")
        else:
            filename = os.path.join(reports_dir, f"execution_time_{timestamp}.csv")
        
        # Write to CSV
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['frame_index', 'object_detection', 'depth_estimation', 
                         'navigation_generation', 'text_to_speech', 'total']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for i, exec_time in enumerate(execution_times):
                writer.writerow({
                    'frame_index': i,
                    'object_detection': exec_time.object_detection,
                    'depth_estimation': exec_time.depth_estimation,
                    'navigation_generation': exec_time.navigation_generation,
                    'text_to_speech': exec_time.text_to_speech,
                    'total': exec_time.total
                })
        
        logger.info(f"Execution time report saved to: {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"Error saving execution time report: {str(e)}")
        return None 