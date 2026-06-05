import logging
import os
import json
import numpy as np
import base64
import io
from PIL import Image, ImageDraw, ImageFont, ImageColor, UnidentifiedImageError
from typing import List, Tuple, Optional, Dict, Any
import math
from queue import Queue
# from dataclasses import dataclass # BoundingBoxRegion will be imported from abstract
from art_agent_team.agents.vision_agent_abstract import VisionAgentAbstract, BoundingBoxRegion
from google.cloud import vision
import google.auth # For credentials
import google.oauth2.service_account # For credentials

# Custom Exceptions for Image Processing Errors
class CorruptedImageError(Exception):
    """Custom exception for corrupted image files."""
    pass

class UnsupportedImageFormatError(Exception):
    """Custom exception for unsupported image formats."""
    pass


# BoundingBoxRegion is now imported from vision_agent_abstract.py
# @dataclass(frozen=True)
# class BoundingBoxRegion:
#     """Data class for storing object bounding box information."""
#     label: str
#     score: float
#     # Normalized vertices of the bounding box (0.0 to 1.0)
#     # [(x0,y0), (x1,y1), (x2,y2), (x3,y3)]
#     normalized_vertices: List[Tuple[float, float]]
    
#     def get_absolute_vertices(self, img_width: int, img_height: int) -> List[Tuple[int, int]]:
#         """Converts normalized vertices to absolute pixel coordinates."""
#         return [(int(v.x * img_width), int(v.y * img_height)) for v in self.normalized_vertices]


class VisionAgentStillLife(VisionAgentAbstract):
    """Agent for analyzing still life art using Vertex AI Vision APIs."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the VisionAgentStillLife with configuration."""
        super().__init__(config)
        self.output_folder = config.get('output_folder', 'output')

        # Initialize Google Cloud Vision Client
        self.vision_client = None
        google_credentials_path = self.config.get('google_credentials_path')

        if not google_credentials_path:
            logging.warning("VisionAgentStillLife: Google credentials path not found in config. Attempting to use default credentials.")
            try:
                self.vision_client = vision.ImageAnnotatorClient()
                logging.info("VisionAgentStillLife: Google Cloud Vision client initialized successfully with default credentials.")
            except google.auth.exceptions.DefaultCredentialsError as e:
                logging.critical(f"VisionAgentStillLife: CRITICAL - Default credentials not found or insufficient: {e}. This agent will not function without credentials.")
            except Exception as e:
                logging.critical(f"VisionAgentStillLife: CRITICAL - Failed to initialize Google Cloud Vision client with default credentials: {e}. This agent will not function.")
        else:
            try:
                credentials = google.oauth2.service_account.Credentials.from_service_account_file(google_credentials_path)
                self.vision_client = vision.ImageAnnotatorClient(credentials=credentials)
                logging.info(f"VisionAgentStillLife: Google Cloud Vision client initialized successfully using credentials from {google_credentials_path}.")
            except FileNotFoundError:
                logging.critical(f"VisionAgentStillLife: CRITICAL - Credentials file not found at {google_credentials_path}. This agent will not function.")
            except Exception as e:
                logging.critical(f"VisionAgentStillLife: CRITICAL - Failed to initialize Google Cloud Vision client using {google_credentials_path}: {e}. This agent will not function.")

        # Colors for visualization
        self.colors = ['red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple',
                      'brown', 'gray', 'beige', 'turquoise', 'cyan', 'magenta',
                      'lime', 'navy', 'maroon', 'teal', 'olive', 'coral', 'lavender',
                      'violet', 'gold', 'silver'] + [c for c in ImageColor.colormap.keys()]

        # Store research data for use in methods
        self.primary_subject: Optional[str] = None
        self.secondary_subjects: List[str] = []

    def _perform_specific_analysis(self,
                                   image_path: str,
                                   pil_image: Image.Image, # PIL.Image object passed from abstract class
                                   artist_name: Optional[str] = None,
                                   art_movement: Optional[str] = None,
                                   title: Optional[str] = None,
                                   year: Optional[str] = None,
                                   research_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyzes still life image using Google Cloud Vision API, incorporating research data.
        Focuses on object localization and label detection.
        This method implements the abstract `_perform_specific_analysis`.
        """
        logging.info(f"{self.__class__.__name__}: Starting specific analysis for image: {image_path}")

        if self.vision_client is None:
            logging.error(f"{self.__class__.__name__}: Google Cloud Vision client not initialized. Cannot analyze image {image_path}.")
            # Return structure expected by abstract class on error
            return {
                "description": "Google Cloud Vision client not initialized.",
                "bounding_boxes": [],
                "segmentation_masks": [],
                "raw_output": {"error": "Google Cloud Vision client not initialized."}
            }
        
        if research_data is None: # Ensure research_data is a dict for consistency
            research_data = {}

        try:
            # Image content for GCP API
            with open(image_path, "rb") as image_file:
                content = image_file.read()
            gcp_image = vision.Image(content=content)
            
            # Use dimensions from the passed PIL image object
            img_width, img_height = pil_image.size

            # --- Object Localization ---
            localized_objects_response = self.vision_client.object_localization(image=gcp_image)
            gcp_localized_objects = localized_objects_response.localized_object_annotations
            logging.info(f"{self.__class__.__name__}: Found {len(gcp_localized_objects)} objects in {image_path}.")

            parsed_bounding_boxes: List[BoundingBoxRegion] = []
            for obj in gcp_localized_objects:
                # The BoundingBoxRegion expects normalized_vertices to be a list of objects/tuples with x,y
                # google.cloud.vision.NormalizedVertex already has .x and .y attributes.
                parsed_bounding_boxes.append(BoundingBoxRegion(
                    label=obj.name,
                    score=obj.score,
                    normalized_vertices=obj.bounding_poly.normalized_vertices
                ))

            # --- Label Detection ---
            label_response = self.vision_client.label_detection(image=gcp_image)
            gcp_labels = label_response.label_annotations
            detected_labels_dict = {label.description: label.score for label in gcp_labels}
            logging.info(f"{self.__class__.__name__}: Detected labels: {detected_labels_dict}")

            # --- Combine raw API results ---
            raw_api_output = {
                "localized_objects_raw": [{"name": obj.name, "score": obj.score, "mid": obj.mid,
                                        "bounding_poly_normalized": [{"x":v.x, "y":v.y} for v in obj.bounding_poly.normalized_vertices]}
                                       for obj in gcp_localized_objects],
                "detected_labels_raw": [{"description": l.description, "score": l.score, "mid": l.mid} for l in gcp_labels],
                "image_dimensions": {"width": img_width, "height": img_height}
            }
            
            # Store research data on self for helper methods (if they use self.primary_subject etc.)
            self.primary_subject = research_data.get("primary_subject")
            self.secondary_subjects = research_data.get("secondary_subjects", [])

            # Calculate importance scores for still life elements
            # Note: _calculate_object_importance expects `analysis` dict with 'detected_labels'
            # We pass raw_api_output which contains 'detected_labels_raw' (list of dicts)
            # Adjusting the call or the helper method. Let's pass detected_labels_dict directly.
            object_importance = self._calculate_object_importance(
                parsed_bounding_boxes,
                detected_labels_dict, # Pass the dict of labels:scores
                research_data,
                img_width,
                img_height
            )
            important_objects_names = self._threshold_important_objects(object_importance)

            description = f"Still life analysis for '{title if title else os.path.basename(image_path)}'. Identified {len(parsed_bounding_boxes)} objects. Important: {', '.join(important_objects_names)}. Labels: {', '.join(detected_labels_dict.keys())[:100]}..."

            logging.info(f"{self.__class__.__name__}: Analysis complete for {image_path}.")
            
            # Return dictionary conforming to VisionAgentAbstract expectations
            return {
                'description': description,
                'bounding_boxes': parsed_bounding_boxes, # This is List[BoundingBoxRegion]
                'segmentation_masks': [], # No pixel masks from this API for object localization
                'important_objects': important_objects_names, # Custom key for this agent
                'raw_output': raw_api_output, # Store all raw responses
                "still_life_specific_details": { # Other agent-specific structured data
                    "primary_subject_found": self.primary_subject in important_objects_names if self.primary_subject else False,
                    "common_labels": list(detected_labels_dict.keys())
                }
            }

        except (UnidentifiedImageError, OSError, FileNotFoundError) as e:
            logging.error(f"{self.__class__.__name__}: Image processing error for {image_path}: {e}")
            raise CorruptedImageError(f"Could not process image {image_path}: {str(e)}") # Re-raise for abstract's handler
        except Exception as e:
            logging.error(f"{self.__class__.__name__}: Error analyzing image {image_path} with Google Cloud Vision API: {str(e)}.", exc_info=True)
            # Return error structure consistent with abstract class expectations
            return {
                "description": f"Error during analysis: {str(e)}",
                "bounding_boxes": [],
                "segmentation_masks": [],
                "raw_output": {"error": str(e), "details": traceback.format_exc()}
            }

    def _calculate_object_importance(self,
                                     bounding_boxes: List[BoundingBoxRegion],
                                     detected_labels: Dict[str, float],
                                     research_data: Dict[str, Any], # research_data can be Optional
                                     img_width: int,
                                     img_height: int) -> Dict[str, float]:
        """
        Calculate importance scores for objects in still life context based on Cloud Vision API output.
        Prioritizes object score, size, centrality, and relevance to still life.
        """
        importance_scores = {}
        
        img_center_x_norm = 0.5
        img_center_y_norm = 0.5
        max_norm_dist = math.sqrt(0.5**2 + 0.5**2) # Max distance from center to corner in normalized coords

        for i, bbox_region in enumerate(bounding_boxes):
            score = 0.0
            label = bbox_region.label.lower()
            api_score = bbox_region.score # Confidence from API

            # Base score from API confidence
            score += api_score * 0.4 # Weight API score

            # Calculate normalized bounding box area
            # Assuming normalized_vertices are [v0, v1, v2, v3] where v0=top-left, v2=bottom-right for a simple rect
            # For a general polygon, more complex area calculation might be needed, but API usually gives axis-aligned.
            # We'll approximate with min/max of normalized coords for simplicity.
            all_x = [v.x for v in bbox_region.normalized_vertices]
            all_y = [v.y for v in bbox_region.normalized_vertices]
            norm_x0, norm_x1 = min(all_x), max(all_x)
            norm_y0, norm_y1 = min(all_y), max(all_y)
            
            norm_area = (norm_x1 - norm_x0) * (norm_y1 - norm_y0)
            size_score = min(norm_area / 0.25, 1.0) * 0.2 # Max 20% score from size (e.g. if it covers 25% of image)
            score += size_score

            # Calculate distance from image center (normalized)
            center_x_norm = (norm_x0 + norm_x1) / 2
            center_y_norm = (norm_y0 + norm_y1) / 2
            
            distance_from_center_norm = math.sqrt(
                (center_x_norm - img_center_x_norm)**2 +
                (center_y_norm - img_center_y_norm)**2
            ) / max_norm_dist # Normalize distance by max possible

            centrality_score = (1 - distance_from_center_norm) * 0.2 # Max 20% score from centrality
            score += centrality_score

            # Boost for traditional still life elements (using object names)
            still_life_keywords = ['fruit', 'flower', 'vase', 'table', 'cloth', 'bottle', 'cup', 'plate', 'book', 'candle', 'food']
            if any(keyword in label for keyword in still_life_keywords):
                score += 0.15
            
            # Boost if the label is also a general image label (reinforces significance)
            if bbox_region.label in detected_labels:
                 score += 0.05

            # Boost for research data matches
            if self.primary_subject and label == self.primary_subject.lower():
                score += 0.15
            elif self.secondary_subjects and label in [s.lower() for s in self.secondary_subjects]:
                score += 0.10
            
            # Use a unique key for each object if multiple objects have the same label
            unique_label_key = f"{label}_{i}"
            importance_scores[unique_label_key] = min(score, 1.0)

        return importance_scores

    def _threshold_important_objects(self, object_importance: Dict[str, float]) -> List[str]:
        """
        Filter objects based on importance scores.
        Returns a list of original object names (without the _index suffix).
        """
        if not object_importance:
            return []

        scores = list(object_importance.values())
        if not scores: return []

        mean_score = np.mean(scores) if scores else 0
        std_score = np.std(scores) if scores else 0
        # Adjust threshold: aim for a reasonable number of important objects
        # Consider a base threshold and adjust by std dev, or a percentile.
        # For now, a slightly more generous threshold than before.
        threshold = mean_score + 0.25 * std_score
        # Ensure threshold is not too low, e.g., at least 0.3 if mean is low
        threshold = max(threshold, 0.3, min(scores) if scores else 0.3)


        important_object_names = set() # Use a set to store unique original names
        still_life_keywords_for_inclusion = ['fruit', 'flower', 'vase', 'table', 'book', 'candle']

        for unique_label_key, score_val in object_importance.items():
            original_label = unique_label_key.rsplit('_', 1)[0] # Get original label
            if score_val >= threshold or any(keyword in original_label.lower() for keyword in still_life_keywords_for_inclusion):
                important_object_names.add(original_label)
        
        # Sort for consistency, though set itself is unordered
        return sorted(list(important_object_names))

# _save_labeled_version is removed. Visualization is handled by VisionAgentAbstract.generate_outputs
# using the _create_bounding_box_image method.

# Removed _save_masked_version as pixel-level masks are not directly available from object_localization

import unittest.mock # For mocking
import traceback # For error details in __main__

if __name__ == '__main__':
    # Configure logging for the test
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    
    dummy_still_life_image_path = "dummy_still_life_image.png"
    try:
        # Create a dummy PIL Image object to pass to process/ _perform_specific_analysis
        pil_dummy_image = Image.new('RGB', (600, 400), color='lightyellow')
        draw = ImageDraw.Draw(pil_dummy_image)
        draw.text((10, 10), "Dummy Still Life", fill=(0, 0, 0))
        # Simulate some objects for detection
        draw.rectangle([(60, 40), (180, 160)], fill="tomato", outline="black")  # Expected "Apple"
        draw.ellipse([(300, 160), (420, 320)], fill="lightblue", outline="black") # Expected "Vase"
        pil_dummy_image.save(dummy_still_life_image_path) # Save it so it can be opened by path
        logging.info(f"Created dummy still life test image: {dummy_still_life_image_path}")

        agent_config = {
            "output_folder": "output/still_life_agent_main_test",
            "google_credentials_path": "mock_credentials.json" # Path doesn't need to exist for this mock
        }
        
        # Mocking the Google Cloud Vision client
        mock_client_instance = unittest.mock.MagicMock(spec=vision.ImageAnnotatorClient)

        # Helper for mock GCP structures
        class MockGCPVertex:
            def __init__(self, x, y): self.x = x; self.y = y
        class MockGCPlocalizedObjectAnnotation:
            def __init__(self, name, score, normalized_vertices_data, mid=None):
                self.name = name
                self.score = score
                self.mid = mid if mid else f"/m/{name.lower().replace(' ', '_')}"
                self.bounding_poly = unittest.mock.MagicMock()
                self.bounding_poly.normalized_vertices = [MockGCPVertex(v[0], v[1]) for v in normalized_vertices_data]
        class MockGCPLabelAnnotation:
            def __init__(self, description, score, mid=None):
                self.description = description
                self.score = score
                self.mid = mid if mid else f"/m/{description.lower().replace(' ', '_')}"

        # Mock response for object_localization
        mock_object_response = unittest.mock.MagicMock()
        mock_object_response.localized_object_annotations = [
            MockGCPlocalizedObjectAnnotation("Apple", 0.92, [(0.1, 0.1), (0.3, 0.1), (0.3, 0.4), (0.1, 0.4)]),
            MockGCPlocalizedObjectAnnotation("Vase", 0.88, [(0.5, 0.4), (0.7, 0.4), (0.7, 0.8), (0.5, 0.8)])
        ]
        mock_client_instance.object_localization.return_value = mock_object_response

        # Mock response for label_detection
        mock_label_response = unittest.mock.MagicMock()
        mock_label_response.label_annotations = [
            MockGCPLabelAnnotation("Still life", 0.95), MockGCPLabelAnnotation("Fruit", 0.90),
            MockGCPLabelAnnotation("Tableware", 0.85), MockGCPLabelAnnotation("Apple", 0.80),
        ]
        mock_client_instance.label_detection.return_value = mock_label_response
        
        still_life_agent = VisionAgentStillLife(config=agent_config)
        still_life_agent.vision_client = mock_client_instance # Inject mock client

        logging.info(f"{still_life_agent.__class__.__name__} initialized with MOCKED Google Cloud Vision client for testing.")

        try:
            research_info = {"primary_subject": "Apple", "secondary_subjects": ["Vase"]}
            
            # Test the full `process` method from the abstract class
            logging.info("\n--- Testing process method ---")
            full_process_result = still_life_agent.process(
                image_path=dummy_still_life_image_path, # process opens the image again
                artist_name="Mock Artist",
                title="Mock Still Life",
                research_data=research_info
            )
            
            logging.info(f"{still_life_agent.__class__.__name__} (mocked) process result: ")
            specific_analysis = full_process_result.get('specific_analysis', {})
            logging.info(f"  Description: {specific_analysis.get('description')}")
            logging.info(f"  Important Objects: {specific_analysis.get('important_objects')}")
            logging.info(f"  Bounding Boxes Count: {len(specific_analysis.get('bounding_boxes', []))}")
            if specific_analysis.get('bounding_boxes'):
                for bbox_idx, bbox_item in enumerate(specific_analysis['bounding_boxes']):
                    logging.info(f"    - BBox {bbox_idx}: Label: {bbox_item.label}, Score: {bbox_item.score:.2f}")
            logging.info(f"  Output Image Paths: {full_process_result.get('output_image_paths')}")


            assert len(specific_analysis.get('bounding_boxes', [])) == 2, "Expected 2 bounding boxes"
            assert "Apple" in specific_analysis.get('important_objects', []), "Expected 'Apple' in important objects"
            assert full_process_result.get('output_image_paths', {}).get('bounding_box_visualization') is not None, "Bounding box visualization should be created"
            logging.info("Basic assertions for process method passed.")

        except (CorruptedImageError, UnsupportedImageFormatError) as e:
            logging.error(f"Image processing error for dummy still life: {e}")
        except Exception as e:
            logging.error(f"Error during {still_life_agent.__class__.__name__} (mocked) processing: {e}", exc_info=True)
        finally:
            if os.path.exists(dummy_still_life_image_path):
                os.remove(dummy_still_life_image_path)
                logging.info(f"Removed dummy still life test image: {dummy_still_life_image_path}")
            output_dir_to_clean = agent_config["output_folder"]
            if os.path.exists(output_dir_to_clean) and "still_life_agent_main_test" in output_dir_to_clean : # Safety check
                import shutil
                # shutil.rmtree(output_dir_to_clean) # Be careful with rmtree in automated scripts
                # logging.info(f"Cleaned up dummy output directory: {output_dir_to_clean}")
                logging.info(f"Test output generated in: {output_dir_to_clean}. Manual cleanup might be needed.")


    except ImportError as e:
        logging.warning(f"Pillow (PIL), unittest.mock or other dependencies might not be installed. Dummy agent test may not run fully: {e}")
    except Exception as e:
        logging.error(f"Error in __main__ setup for {VisionAgentStillLife.__name__}: {e}", exc_info=True)
