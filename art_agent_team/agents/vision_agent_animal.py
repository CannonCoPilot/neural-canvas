import logging
import os
import json
import numpy as np
import base64
import io
from PIL import Image, ImageDraw, ImageFont, ImageColor, UnidentifiedImageError
from typing import List, Tuple, Optional, Dict, Any
import math
# from queue import Queue # Not used
# from dataclasses import dataclass # Using dataclasses from abstract

# Import from abstract class
from art_agent_team.agents.vision_agent_abstract import (
    VisionAgentAbstract,
    # BoundingBox,
    # SegmentationMask,
    CorruptedImageError,
    UnsupportedImageFormatError,
    InvalidMaskError,
    BoundingBoxRegion # Import from abstract class
)
# Local BoundingBoxRegion definition/import from still_life is no longer needed.
# try:
#     from art_agent_team.agents.vision_agent_still_life import BoundingBoxRegion
# except ImportError:
#     logging.warning("Could not import BoundingBoxRegion from vision_agent_still_life. Define locally or fix import.")
#     from dataclasses import dataclass
#     @dataclass(frozen=True)
#     class BoundingBoxRegion: # Minimal fallback
#         label: str
#         score: float
#         normalized_vertices: List[Any]


from google.cloud import vision
import google.auth
import google.oauth2.service_account

# Configure logging - This might be already configured by the calling script or abstract class
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Custom Exceptions for Image Processing Errors are now in abstract_vision_agent
# Remove local definitions if they are identical or ensure they are not conflicting.
# For now, assuming the abstract class's exceptions are canonical.

# The SegmentationMask dataclass is now defined in vision_agent_abstract.py
# Remove the local definition:
# @dataclass(frozen=True)
# class SegmentationMask:
#     """Data class for storing segmentation mask information."""
#     y0: int  # in [0..height - 1]
#     x0: int  # in [0..width - 1]
#     y1: int  # in [0..height - 1]
#     x1: int  # in [0..width - 1]
#     mask: np.ndarray  # [img_height, img_width] with values 0..255
#     label: str


class VisionAgentAnimal(VisionAgentAbstract):
    """Agent for analyzing animal art using Vertex AI Vision APIs."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the VisionAgentAnimal with configuration."""
        super().__init__(config)
        logging.info(f"Initializing {self.__class__.__name__} (Animal Specific)")
        self.output_folder = self.config.get('output_folder', 'output/animal_analysis') # Ensure output folder is configured
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder, exist_ok=True)

        # Initialize Google Cloud Vision Client
        self.vision_client = None
        google_credentials_path = self.config.get('google_credentials_path')

        if not google_credentials_path:
            logging.warning(f"{self.__class__.__name__}: Google credentials path not found in config. Attempting to use default credentials.")
            try:
                self.vision_client = vision.ImageAnnotatorClient()
                logging.info(f"{self.__class__.__name__}: Google Cloud Vision client initialized successfully with default credentials.")
            except google.auth.exceptions.DefaultCredentialsError as e:
                logging.critical(f"{self.__class__.__name__}: CRITICAL - Default credentials not found or insufficient: {e}. This agent will not function.")
            except Exception as e:
                logging.critical(f"{self.__class__.__name__}: CRITICAL - Failed to initialize Google Cloud Vision client with default credentials: {e}. This agent will not function.")
        else:
            try:
                credentials = google.oauth2.service_account.Credentials.from_service_account_file(google_credentials_path)
                self.vision_client = vision.ImageAnnotatorClient(credentials=credentials)
                logging.info(f"{self.__class__.__name__}: Google Cloud Vision client initialized successfully using credentials from {google_credentials_path}.")
            except FileNotFoundError:
                logging.critical(f"{self.__class__.__name__}: CRITICAL - Credentials file not found at {google_credentials_path}. This agent will not function.")
            except Exception as e:
                logging.critical(f"{self.__class__.__name__}: CRITICAL - Failed to initialize Google Cloud Vision client using {google_credentials_path}: {e}. This agent will not function.")
        
        # Colors for visualization (can be inherited or defined here)
        self.colors = ['red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 'cyan', 'magenta', 'lime']

        # self.colors and self.font are initialized in VisionAgentAbstract's __init__

        # Store research data for use in methods (if needed by specific analysis)
        self.primary_subject: Optional[str] = None # Example, can be passed via params if needed
        self.secondary_subjects: List[str] = []   # Example

    def _perform_specific_analysis(self, image_path: str, pil_image: Image.Image, artist_name: Optional[str] = None, art_movement: Optional[str] = None, title: Optional[str] = None, year: Optional[str] = None, research_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyzes animal art using Google Cloud Vision API.
        Focuses on object localization (for animals) and face detection (if applicable).
        """
        if self.vision_client is None:
            logging.error(f"{self.__class__.__name__}: Google Cloud Vision client not initialized. Cannot perform analysis.")
            return {
                "description": "Google Cloud Vision client not initialized.",
                "bounding_boxes": [], "segmentation_masks": [], # segmentation_masks will be empty
                "raw_output": {"error": "Google Cloud Vision client not initialized."}
            }

        logging.info(f"{self.__class__.__name__} performing specific analysis on animal image: {image_path}")
        
        if research_data: # Store research data if provided
            self.primary_subject = research_data.get("primary_subject")
            self.secondary_subjects = research_data.get("secondary_subjects", [])
        else: # Ensure they are reset if not provided for this call
            self.primary_subject = None
            self.secondary_subjects = []


        try:
            with open(image_path, "rb") as image_file:
                content = image_file.read()
            gcp_image = vision.Image(content=content)
            img_width, img_height = pil_image.size

            # --- Object Localization ---
            localized_objects_response = self.vision_client.object_localization(image=gcp_image)
            localized_objects = localized_objects_response.localized_object_annotations
            logging.info(f"{self.__class__.__name__}: Found {len(localized_objects)} objects in {image_path}.")

            parsed_bounding_boxes: List[BoundingBoxRegion] = []
            for obj in localized_objects:
                parsed_bounding_boxes.append(BoundingBoxRegion(
                    label=obj.name,
                    score=obj.score,
                    normalized_vertices=obj.bounding_poly.normalized_vertices
                ))

            # --- Face Detection (Optional, can add more details) ---
            # Faces are often also detected as objects. This can provide more detail if faces are clear.
            face_annotations = []
            try:
                face_response = self.vision_client.face_detection(image=gcp_image)
                if face_response.face_annotations:
                    logging.info(f"{self.__class__.__name__}: Found {len(face_response.face_annotations)} faces.")
                    for face in face_response.face_annotations:
                        face_data = {
                            "confidence": face.detection_confidence,
                            "bounding_poly_normalized": [(v.x, v.y) for v in face.bounding_poly.vertices],
                            "joy_likelihood": vision.Likelihood(face.joy_likelihood).name,
                            "sorrow_likelihood": vision.Likelihood(face.sorrow_likelihood).name,
                            "anger_likelihood": vision.Likelihood(face.anger_likelihood).name,
                            "surprise_likelihood": vision.Likelihood(face.surprise_likelihood).name,
                        }
                        face_annotations.append(face_data)
                        # Optionally, add faces as BoundingBoxRegion if they aren't already covered by object_localization
                        # For now, keeping them separate in raw_output.
            except Exception as e_face:
                logging.warning(f"{self.__class__.__name__}: Face detection failed or not applicable: {e_face}")


            # --- Construct Analysis Dictionary ---
            description = f"Animal art analysis for '{title if title else 'image'}'. Identified {len(parsed_bounding_boxes)} objects."
            if face_annotations:
                description += f" Detected {len(face_annotations)} faces."

            raw_output = {
                "localized_objects": [{"name": obj.name, "score": obj.score, "mid": obj.mid, "normalized_bounding_poly": [{"x":v.x, "y":v.y} for v in obj.bounding_poly.normalized_vertices]} for obj in localized_objects],
                "face_annotations": face_annotations,
                "image_dimensions": {"width": img_width, "height": img_height}
            }
            
            # --- Importance Calculation ---
            # Pass img_width and img_height for any calculations needing absolute dimensions
            object_importance = self._calculate_object_importance(parsed_bounding_boxes, raw_output, research_data if research_data else {}, img_width, img_height)
            important_objects_names = self._threshold_important_objects(object_importance) # Returns list of names

            # --- Save Labeled Image ---
            # The abstract 'process' method will call 'generate_outputs', which in turn can call
            # a method like '_create_bounding_box_image'. We need to ensure the data is in BoundingBoxRegion format.
            # For now, we'll prepare a path for a visualization that can be created by generate_outputs.
            # We can also create one here directly if preferred for animal-specific labeling.
            # output_visualization_path = self._save_animal_labeled_image(image_path, pil_image, parsed_bounding_boxes, img_width, img_height)
            output_visualization_path = None # To be generated by abstract class


            return {
                'description': description,
                'bounding_boxes': parsed_bounding_boxes, # List of BoundingBoxRegion
                'segmentation_masks': [], # Google Cloud Vision API object_localization does not provide pixel masks
                'important_objects': important_objects_names, # List of important object names
                'raw_output': raw_output,
                # 'visualization_path': output_visualization_path # Removed, handled by abstract class
                # Add any other animal-specific structured data here
                "animal_specific_details": {
                    "face_count": len(face_annotations),
                    "primary_subject_in_important": self.primary_subject in important_objects_names if self.primary_subject else False,
                }
            }

        except (UnidentifiedImageError, OSError, FileNotFoundError) as e:
            logging.error(f"{self.__class__.__name__}: Image processing error for {image_path}: {e}")
            raise CorruptedImageError(f"Could not process image {image_path}: {str(e)}") # Re-raise for abstract's handler
        except InvalidMaskError as e: # Should not occur as we are not processing masks here
            logging.error(f"{self.__class__.__name__}: Invalid mask error (unexpected): {e}")
            raise # Re-raise
        except Exception as e:
            logging.error(f"{self.__class__.__name__}: Unexpected error analyzing image {image_path}: {str(e)}", exc_info=True)
            import traceback
            # Fallback structure consistent with abstract class
            return {
                "description": f"Error during animal analysis: {str(e)}",
                "bounding_boxes": [],
                "segmentation_masks": [],
                "raw_output": {"error": str(e), "details": traceback.format_exc()}
            }


    def _calculate_object_importance(self,
                                   bounding_boxes: List[BoundingBoxRegion],
                                   analysis_raw: Dict[str, Any],
                                   research_data: Dict[str, Any],
                                   img_width: int,
                                   img_height: int) -> Dict[str, float]:
        """
        Calculate importance scores for objects in animal art context.
        Uses BoundingBoxRegion objects.
        """
        importance_scores = {}
        image_area_pixels = img_width * img_height
        if image_area_pixels == 0: image_area_pixels = 1

        img_center_x_norm = 0.5
        img_center_y_norm = 0.5
        max_norm_dist = math.sqrt(0.5**2 + 0.5**2)

        for i, bbox_region in enumerate(bounding_boxes):
            score = 0.0
            label = bbox_region.label.lower()
            api_score = bbox_region.score

            score += api_score * 0.5 # Weight API score heavily

            # Normalized area
            all_x = [v.x for v in bbox_region.normalized_vertices]
            all_y = [v.y for v in bbox_region.normalized_vertices]
            norm_x0, norm_x1 = min(all_x) if all_x else 0, max(all_x) if all_x else 0
            norm_y0, norm_y1 = min(all_y) if all_y else 0, max(all_y) if all_y else 0
            
            norm_area = (norm_x1 - norm_x0) * (norm_y1 - norm_y0)
            size_score = min(norm_area / 0.3, 1.0) * 0.2 # Max 20% if it covers 30% of image
            score += size_score
            
            # Centrality
            center_x_norm = (norm_x0 + norm_x1) / 2
            center_y_norm = (norm_y0 + norm_y1) / 2
            distance_from_center_norm = math.sqrt(
                (center_x_norm - img_center_x_norm)**2 +
                (center_y_norm - img_center_y_norm)**2
            ) / max_norm_dist
            centrality_score = (1 - distance_from_center_norm) * 0.1 # Max 10% from centrality
            score += centrality_score

            # Boost for animal subjects - use a more comprehensive list or check against known animal categories
            animal_keywords = ['animal', 'bird', 'mammal', 'reptile', 'fish', 'insect', 'cat', 'dog', 'horse', 'lion', 'tiger', 'bear', 'elephant', 'monkey', 'deer', 'wolf', 'fox', 'rabbit', 'squirrel', 'cow', 'sheep', 'pig', 'chicken', 'duck', 'eagle', 'owl', 'snake', 'lizard', 'frog', 'turtle', 'butterfly']
            if any(keyword in label for keyword in animal_keywords) or "animal" in label: # General "animal" catch-all
                score += 0.25 # Strong boost if it's likely an animal

            # Boost for research data matches
            current_primary_subject = (research_data.get("primary_subject") or self.primary_subject)
            current_secondary_subjects = (research_data.get("secondary_subjects") or self.secondary_subjects)

            if current_primary_subject and label == current_primary_subject.lower():
                score += 0.15
            elif current_secondary_subjects and label in [s.lower() for s in current_secondary_subjects]:
                score += 0.10
            
            unique_label_key = f"{label}_{i}" # Ensure unique key for multiple instances of same label
            importance_scores[unique_label_key] = min(score, 1.0)
            
        return importance_scores

    def _threshold_important_objects(self, object_importance: Dict[str, float]) -> List[str]:
        """Filter objects based on importance scores. Returns list of original object names."""
        if not object_importance:
            return []
        scores = list(object_importance.values())
        if not scores: return []
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        threshold = mean_score + 0.3 * std_score
        threshold = max(threshold, 0.4) # Ensure a minimum threshold, e.g., 0.4

        important_object_names = set()
        animal_keywords_for_inclusion = ['animal', 'bird', 'mammal', 'cat', 'dog', 'horse', 'lion', 'tiger', 'eagle']


        for unique_label_key, score_val in object_importance.items():
            original_label = unique_label_key.rsplit('_',1)[0]
            if score_val >= threshold or any(keyword in original_label.lower() for keyword in animal_keywords_for_inclusion):
                important_object_names.add(original_label)
        
        return sorted(list(important_object_names))

# _save_animal_labeled_image is removed. Visualization is handled by VisionAgentAbstract.generate_outputs
# using the _create_bounding_box_image method.

    # The main 'process' method is inherited from VisionAgentAbstract.
    # It will call _perform_specific_analysis and then generate_outputs.

    # _save_labeled_version and _save_masked_version are removed as their functionality
    # is now handled by _create_bounding_box_image, _create_segmentation_mask_image,
    # and _create_cropped_images in the VisionAgentAbstract class, orchestrated by generate_outputs.

import unittest.mock # For mocking
import traceback # For error details in __main__

if __name__ == '__main__':
    # Configure logging for the test
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    dummy_animal_image_path = "dummy_animal_image.png"
    try:
        pil_dummy_image = Image.new('RGB', (700, 500), color='lightcoral')
        draw = ImageDraw.Draw(pil_dummy_image)
        draw.text((20, 20), "Dummy Animal Test", fill=(0, 0, 0))
        draw.ellipse([(100, 100), (300, 300)], fill="sandybrown", outline="black") # "Cat"
        draw.rectangle([(400, 200), (600, 400)], fill="darkgray", outline="black") # "Dog"
        pil_dummy_image.save(dummy_animal_image_path)
        logging.info(f"Created dummy animal test image: {dummy_animal_image_path}")

        agent_config = {
            "output_folder": "output/animal_agent_main_test",
            "google_credentials_path": "mock_credentials.json"
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
        
        class MockGCPFaceAnnotation:
            def __init__(self, confidence, bounding_poly_vertices, joy=vision.Likelihood.UNKNOWN, sorrow=vision.Likelihood.UNKNOWN, anger=vision.Likelihood.UNKNOWN, surprise=vision.Likelihood.UNKNOWN):
                self.detection_confidence = confidence
                self.bounding_poly = unittest.mock.MagicMock()
                self.bounding_poly.vertices = [MockGCPVertex(v[0], v[1]) for v in bounding_poly_vertices] # Face bpoly is absolute
                self.joy_likelihood = joy
                self.sorrow_likelihood = sorrow
                self.anger_likelihood = anger
                self.surprise_likelihood = surprise


        # Mock response for object_localization
        mock_object_response = unittest.mock.MagicMock()
        mock_object_response.localized_object_annotations = [
            MockGCPlocalizedObjectAnnotation("Cat", 0.92, [(0.14, 0.2), (0.42, 0.2), (0.42, 0.6), (0.14, 0.6)]), # Approx from ellipse
            MockGCPlocalizedObjectAnnotation("Dog", 0.85, [(0.57, 0.4), (0.85, 0.4), (0.85, 0.8), (0.57, 0.8)])  # Approx from rectangle
        ]
        mock_client_instance.object_localization.return_value = mock_object_response

        # Mock response for face_detection
        mock_face_response = unittest.mock.MagicMock()
        # Simulate one face found within the "Cat" object
        # Note: face bounding_poly vertices are absolute pixel coordinates
        mock_face_response.face_annotations = [
             MockGCPFaceAnnotation(0.98,
                                   [(150, 150), (250, 150), (250, 250), (150, 250)], # Absolute pixel coords for face
                                   joy=vision.Likelihood.VERY_LIKELY)
        ]
        mock_client_instance.face_detection.return_value = mock_face_response
        
        animal_agent = VisionAgentAnimal(config=agent_config)
        animal_agent.vision_client = mock_client_instance

        logging.info(f"{animal_agent.__class__.__name__} initialized with MOCKED Google Cloud Vision client for testing.")

        try:
            research_info = {"primary_subject": "Cat", "secondary_subjects": ["Dog"]}
            
            logging.info("\n--- Testing process method (Animal Agent) ---")
            full_process_result = animal_agent.process(
                image_path=dummy_animal_image_path,
                artist_name="AI Test Artist",
                title="Mock Digital Menagerie",
                research_data=research_info
            )
            
            logging.info(f"{animal_agent.__class__.__name__} (mocked) process result: ")
            specific_analysis = full_process_result.get('specific_analysis', {})
            logging.info(f"  Description: {specific_analysis.get('description')}")
            logging.info(f"  Important Objects: {specific_analysis.get('important_objects')}")
            logging.info(f"  Bounding Boxes Count: {len(specific_analysis.get('bounding_boxes', []))}")
            if specific_analysis.get('bounding_boxes'):
                for bbox_idx, bbox_item in enumerate(specific_analysis['bounding_boxes']):
                    logging.info(f"    - BBox {bbox_idx}: Label: {bbox_item.label}, Score: {bbox_item.score:.2f}")
            
            animal_details = specific_analysis.get('animal_specific_details', {})
            logging.info(f"  Animal Specific Details: Face Count: {animal_details.get('face_count')}")
            
            logging.info(f"  Output Image Paths: {full_process_result.get('output_image_paths')}")

            assert len(specific_analysis.get('bounding_boxes', [])) == 2, "Expected 2 object bounding boxes"
            assert "Cat" in specific_analysis.get('important_objects', []), "Expected 'Cat' in important objects"
            assert animal_details.get('face_count') == 1, "Expected 1 face detected"
            assert full_process_result.get('output_image_paths', {}).get('bounding_box_visualization') is not None, "Bounding box visualization should be created"
            logging.info("Basic assertions for Animal Agent process method passed.")

        except (CorruptedImageError, UnsupportedImageFormatError, InvalidMaskError) as e:
            logging.error(f"Image processing error for dummy animal: {e}")
        except Exception as e:
            logging.error(f"Error during {animal_agent.__class__.__name__} (mocked) processing: {e}", exc_info=True)
        finally:
            if os.path.exists(dummy_animal_image_path):
                os.remove(dummy_animal_image_path)
                logging.info(f"Removed dummy animal test image: {dummy_animal_image_path}")
            output_dir_to_clean = agent_config["output_folder"]
            if os.path.exists(output_dir_to_clean) and "animal_agent_main_test" in output_dir_to_clean:
                # import shutil
                # shutil.rmtree(output_dir_to_clean)
                logging.info(f"Test output generated in: {output_dir_to_clean}. Manual cleanup might be needed.")

    except ImportError as e:
        logging.warning(f"Pillow (PIL), unittest.mock or other dependencies might not be installed. Dummy agent test may not run fully: {e}")
    except Exception as e:
        logging.error(f"Error in __main__ setup for {VisionAgentAnimal.__name__}: {e}", exc_info=True)
