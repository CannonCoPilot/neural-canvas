import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor # Added for drawing
import io # For image bytes
import os # For path manipulation in __main__
from typing import Dict, Any, Optional, List

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
#     # Minimal BoundingBoxRegion definition as a fallback for now
#     from dataclasses import dataclass
#     @dataclass(frozen=True)
#     class BoundingBoxRegion:
#         label: str
#         score: float
#         normalized_vertices: List[Any] # Simplified for fallback

from google.cloud import vision
import google.auth
import google.oauth2.service_account


# Configure logging - This might be already configured by the calling script or abstract class
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VisionAgentLandscape(VisionAgentAbstract):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        logging.info(f"Initializing {self.__class__.__name__} (Landscape Specific)")
        self.output_folder = self.config.get('output_folder', 'output/landscape_analysis') # Ensure output folder is configured
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
        self.colors = ['red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple']


    def _perform_specific_analysis(self,
                                   image_path: str,
                                   pil_image: Image.Image,
                                   artist_name: Optional[str] = None,
                                   art_movement: Optional[str] = None,
                                   title: Optional[str] = None,
                                   year: Optional[str] = None,
                                   research_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform landscape-specific analysis using Google Cloud Vision API.
        Focuses on label detection, landmark detection, and image properties.
        """
        logging.info(f"{self.__class__.__name__} performing specific analysis on landscape image: {image_path}")

        if self.vision_client is None:
            logging.error(f"{self.__class__.__name__}: Google Cloud Vision client not initialized. Cannot analyze image {image_path}.")
            # Return structure expected by abstract class on error
            return {
                "description": "Google Cloud Vision client not initialized.",
                "bounding_boxes": [], "segmentation_masks": [],
                "raw_output": {"error": "Google Cloud Vision client not initialized."}
            }
        
        if research_data is None: research_data = {} # Ensure it's a dict

        try:
            with open(image_path, "rb") as image_file:
                content = image_file.read()
            gcp_image = vision.Image(content=content)
            img_width, img_height = pil_image.size

            features = [
                vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION, max_results=15),
                vision.Feature(type_=vision.Feature.Type.LANDMARK_DETECTION, max_results=5),
                vision.Feature(type_=vision.Feature.Type.IMAGE_PROPERTIES),
            ]
            request = vision.AnnotateImageRequest(image=gcp_image, features=features)
            response = self.vision_client.annotate_image(request=request)

            if response.error.message:
                logging.error(f"GCP Vision API error for {image_path}: {response.error.message}")
                # Return error structure
                return {
                    "description": f"GCP Vision API error: {response.error.message}",
                    "bounding_boxes": [], "segmentation_masks": [],
                    "raw_output": {"error": response.error.message, "details": str(response.error)}
                }

            identified_elements = [label.description for label in response.label_annotations]
            label_details = {label.description: label.score for label in response.label_annotations}
            logging.info(f"{self.__class__.__name__}: Identified elements (labels): {identified_elements}")

            landmarks_data_raw = []
            parsed_landmark_bounding_boxes: List[BoundingBoxRegion] = []
            for landmark in response.landmark_annotations:
                landmarks_data_raw.append({
                    "name": landmark.description, "score": landmark.score, "mid": landmark.mid,
                    "bounding_poly_pixel_vertices": [{"x": v.x, "y": v.y} for v in landmark.bounding_poly.vertices],
                    "locations": [{"latitude": loc.latitude, "longitude": loc.longitude} for loc in landmark.locations]
                })
                
                # Normalize landmark vertices (they are absolute pixel coordinates)
                normalized_vertices = []
                if landmark.bounding_poly and landmark.bounding_poly.vertices:
                    if img_width > 0 and img_height > 0: # Avoid division by zero
                        normalized_vertices = [
                            vision.NormalizedVertex(x=v.x / img_width, y=v.y / img_height)
                            for v in landmark.bounding_poly.vertices
                        ]
                    else: # Should not happen if pil_image is valid
                        logging.warning(f"Image dimensions are zero for {image_path}, cannot normalize landmark vertices.")
                
                if normalized_vertices: # Only add if we have valid normalized vertices
                    parsed_landmark_bounding_boxes.append(
                        BoundingBoxRegion(
                            label=landmark.description,
                            score=landmark.score,
                            normalized_vertices=normalized_vertices
                        )
                    )
            logging.info(f"{self.__class__.__name__}: Detected {len(parsed_landmark_bounding_boxes)} landmarks with bounding boxes.")

            color_palette_rgb_details = []
            if response.image_properties_annotation and response.image_properties_annotation.dominant_colors:
                colors = response.image_properties_annotation.dominant_colors.colors
                for color_info in colors:
                    color_palette_rgb_details.append({
                        "rgb": (int(color_info.color.red), int(color_info.color.green), int(color_info.color.blue)),
                        "score": color_info.score, "pixel_fraction": color_info.pixel_fraction
                    })
            color_palette_hex = [f"#{c['rgb'][0]:02x}{c['rgb'][1]:02x}{c['rgb'][2]:02x}" for c in color_palette_rgb_details]
            logging.info(f"{self.__class__.__name__}: Dominant colors (hex): {color_palette_hex}")

            description_text = f"Landscape analysis for '{title if title else os.path.basename(image_path)}'. "
            if identified_elements: description_text += f"Key elements: {', '.join(identified_elements[:5])}. "
            if landmarks_data_raw: description_text += f"Landmarks: {', '.join([l['name'] for l in landmarks_data_raw])}. "
            description_text += f"Colors: {', '.join(color_palette_hex[:3])}."

            raw_output_data = {
                "labels_raw": [{"description": l.description, "score": l.score, "mid": l.mid} for l in response.label_annotations],
                "landmarks_raw": landmarks_data_raw,
                "image_properties_raw": {
                    "dominant_colors": [{"color": {"red": c.color.red, "green": c.color.green, "blue": c.color.blue, "alpha": c.color.alpha.value if c.color.alpha else None},
                                         "score": c.score, "pixel_fraction": c.pixel_fraction}
                                        for c in response.image_properties_annotation.dominant_colors.colors]
                } if response.image_properties_annotation else None
            }

            return {
                "description": description_text,
                "bounding_boxes": parsed_landmark_bounding_boxes, # List[BoundingBoxRegion]
                "segmentation_masks": [], # No segmentation from this API
                "raw_output": raw_output_data,
                # Landscape specific keys:
                "identified_elements_list": identified_elements,
                "label_scores": label_details,
                "dominant_color_palette_rgb": color_palette_rgb_details,
                "dominant_color_palette_hex": color_palette_hex,
                "genre_confidence_landscape": 0.80, # Placeholder
            }
        except Exception as e:
            logging.error(f"{self.__class__.__name__}: Error during specific analysis for {image_path}: {e}", exc_info=True)
            import traceback
            return {
                "description": f"Error during landscape analysis: {str(e)}",
                "bounding_boxes": [],
                "segmentation_masks": [],
                "raw_output": {"error": str(e), "details": traceback.format_exc()}
            }

# _save_landscape_labeled_image is removed. Visualization is handled by VisionAgentAbstract.generate_outputs
# using the _create_bounding_box_image method.

    # The main 'process' method is inherited from VisionAgentAbstract.
    # It will call _perform_specific_analysis and then generate_outputs.

if __name__ == '__main__':
    # Example usage (optional)
    # Create a dummy image for testing
    dummy_landscape_image_path = "dummy_landscape_image.png"
    try:
        img = Image.new('RGB', (800, 600), color = 'lightgreen')
        d = ImageDraw.Draw(img)
        d.text((20,20), "Dummy Landscape Test", fill=(0,0,0))
        # Add some shapes to represent landscape elements
        d.rectangle([(50, 200), (300, 400)], fill="saddlebrown", outline="black") # "mountains"
        d.ellipse([(400, 300), (700, 550)], fill="darkgreen", outline="black") # "forest"
        img.save(dummy_landscape_image_path)
        logging.info(f"Created dummy landscape test image: {dummy_landscape_image_path}")

        agent_config = {"output_folder": "output/landscape_agent_outputs"}
        landscape_agent = VisionAgentLandscape(config=agent_config)
        
        try:
            # The process method from the abstract class will be called
            result = landscape_agent.process(
                dummy_landscape_image_path, 
                artist_name="Claude Monet (Dummy)", 
                title="Impression, soleil levant (Dummy)", 
                year="1872 (Dummy)"
            )
            logging.info(f"Landscape Agent processing result: {result}")
            if result and result.get('output_image_paths'):
                logging.info("Generated output images:")
                for key, path in result['output_image_paths'].items():
                    if path: # Path can be None if no bboxes/masks
                        logging.info(f"  {key}: {path}")
        except (CorruptedImageError, UnsupportedImageFormatError, InvalidMaskError) as e:
            logging.error(f"Image processing error for dummy landscape: {e}")
        except Exception as e:
            logging.error(f"Error during VisionAgentLandscape processing: {e}", exc_info=True)
        finally:
            # Clean up dummy image
            if os.path.exists(dummy_landscape_image_path):
                os.remove(dummy_landscape_image_path)
                logging.info(f"Removed dummy landscape test image: {dummy_landscape_image_path}")
                
    except ImportError:
        logging.warning("Pillow (PIL) might not be installed. Dummy agent test may not run fully.")
    except Exception as e:
        logging.error(f"Error in __main__ setup for VisionAgentLandscape: {e}", exc_info=True)
