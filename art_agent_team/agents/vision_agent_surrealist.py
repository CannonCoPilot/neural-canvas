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
from dataclasses import dataclass
from art_agent_team.agents.vision_agent_abstract import VisionAgentAbstract
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

# Custom Exceptions for Image Processing Errors
class CorruptedImageError(Exception):
    """Custom exception for corrupted image files."""
    pass

class UnsupportedImageFormatError(Exception):
    """Custom exception for unsupported image formats."""
    pass

@dataclass(frozen=True)
class SegmentationMask:
    """Data class for storing segmentation mask information."""
    y0: int  # in [0..height - 1]
    x0: int  # in [0..width - 1]
    y1: int  # in [0..height - 1]
    x1: int  # in [0..width - 1]
    mask: np.ndarray  # [img_height, img_width] with values 0..255
    label: str

class VisionAgentSurrealist(VisionAgentAbstract):
    """Agent for analyzing surrealist art using Vertex AI Vision APIs."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the VisionAgentSurrealist with configuration."""
        super().__init__(config)
        self.output_folder = config.get('output_folder', 'output')

        # Initialize Vertex AI
        try:
            aiplatform.init(
                project=self.config.get('gcp_project_id'),
                location=self.config.get('gcp_location', 'us-central1')
            )
            self.vertex_model = aiplatform.Model(self.config.get('vertex_model_id'))
            logging.info("VisionAgentSurrealist: Vertex AI model initialized successfully")
        except Exception as e:
            logging.error(f"VisionAgentSurrealist: Failed to initialize Vertex AI model: {e}")
            raise

        # Colors for visualization
        self.colors = ['red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple',
                      'brown', 'gray', 'beige', 'turquoise', 'cyan', 'magenta',
                      'lime', 'navy', 'maroon', 'teal', 'olive', 'coral', 'lavender',
                      'violet', 'gold', 'silver'] + [c for c in ImageColor.colormap.keys()]

        # Store research data for use in methods
        self.primary_subject: Optional[str] = None
        self.secondary_subjects: List[str] = []

    def analyze_image(self, image_path: str, research_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Analyzes surrealist image using Vertex AI, incorporating research data.
        Focuses on identifying dream-like elements, symbolism, and unusual juxtapositions.
        """
        try:
            # Load and validate image
            with Image.open(image_path) as img:
                if img.format not in ['JPEG', 'PNG', 'WEBP']:
                    raise UnsupportedImageFormatError(f"Unsupported image format: {img.format}")
                
                # Convert image to bytes for Vertex AI
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_bytes = img_byte_arr.getvalue()
                
                # Prepare prompt for surrealist analysis
                prompt = """Analyze this surrealist artwork focusing on:
                1. Dream-like elements and symbolism
                2. Unusual juxtapositions of objects
                3. Distortions of reality and space
                4. Hidden meanings and psychological elements
                5. Identify and segment key surrealist objects and symbols
                
                For each identified object or element, provide:
                - Symbolic meaning
                - Relationship to surrealist themes
                - Spatial relationship to other elements
                - Any distortions or transformations
                """

                # Call Vertex AI for analysis
                instance = json_format.ParseDict({
                    "prompt": prompt,
                    "image": {"bytesBase64Encoded": base64.b64encode(img_bytes).decode()},
                }, Value())

                prediction = self.vertex_model.predict([instance])
                analysis_result = json.loads(prediction.predictions[0])

                # Extract segmentation masks and metadata
                masks = []
                for obj in analysis_result.get('objects', []):
                    if 'mask' in obj and 'label' in obj:
                        mask_array = np.array(obj['mask'])
                        bbox = obj.get('bbox', [0, 0, mask_array.shape[1], mask_array.shape[0]])
                        masks.append(SegmentationMask(
                            y0=bbox[0], x0=bbox[1],
                            y1=bbox[2], x1=bbox[3],
                            mask=mask_array,
                            label=obj['label']
                        ))

                # Save visualization of analysis
                output_img = self._save_labeled_version(image_path, masks, analysis_result)
                masked_img = self._save_masked_version(image_path, masks)

                # Calculate importance scores for surrealist elements
                object_importance = self._calculate_object_importance(masks, analysis_result, research_data)
                important_objects = self._threshold_important_objects(object_importance)

                return {
                    'analysis': analysis_result,
                    'important_objects': important_objects,
                    'segmentation_masks': masks,
                    'visualization_path': output_img,
                    'masked_version_path': masked_img
                }

        except (UnidentifiedImageError, OSError) as e:
            raise CorruptedImageError(f"Could not process image {image_path}: {str(e)}")
        except Exception as e:
            logging.error(f"Error analyzing image {image_path}: {str(e)}")
            raise

    def _calculate_object_importance(self, masks: List[SegmentationMask], 
                                   analysis: Dict[str, Any],
                                   research_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate importance scores for objects in surrealist context.
        Prioritizes objects that exhibit surrealist characteristics:
        - Unusual juxtapositions
        - Dream-like qualities
        - Symbolic significance
        - Distortions of reality
        """
        importance_scores = {}
        
        for mask in masks:
            score = 0.0
            label = mask.label.lower()
            
            # Base score from mask size
            area = (mask.y1 - mask.y0) * (mask.x1 - mask.x0)
            size_score = min(area / (analysis.get('image_area', 1000000)), 0.5)
            score += size_score

            # Boost score for surrealist elements
            if any(keyword in label for keyword in ['dream', 'symbol', 'distort', 'transform']):
                score += 0.3

            # Boost for objects mentioned in research data
            if label in research_data.get('symbolic_elements', []):
                score += 0.2

            # Boost for unusual combinations
            if any(other.label.lower() != label for other in masks):
                score += 0.2

            importance_scores[label] = min(score, 1.0)

        return importance_scores

    def _threshold_important_objects(self, object_importance: Dict[str, float]) -> List[str]:
        """
        Filter objects based on importance scores.
        Uses a dynamic threshold for surrealist works, considering symbolic significance.
        """
        if not object_importance:
            return []

        # Dynamic threshold based on score distribution
        scores = list(object_importance.values())
        threshold = np.mean(scores) + 0.5 * np.std(scores)
        
        return [obj for obj, score in object_importance.items() if score >= threshold]

    def _save_labeled_version(self, image_path: str, masks: List[SegmentationMask],
                            analysis: Dict[str, Any]) -> str:
        """Save a version of the image with surrealist elements labeled and annotated."""
        with Image.open(image_path) as img:
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 20)
            except IOError:
                font = ImageFont.load_default()

            for idx, mask in enumerate(masks):
                color = self.colors[idx % len(self.colors)]
                label = mask.label
                
                # Draw bounding box
                draw.rectangle(
                    [(mask.x0, mask.y0), (mask.x1, mask.y1)],
                    outline=color,
                    width=2
                )
                
                # Add label with surrealist interpretation
                interpretation = analysis.get('interpretations', {}).get(label, '')
                text = f"{label}: {interpretation[:30]}..." if interpretation else label
                draw.text((mask.x0, mask.y0 - 25), text, fill=color, font=font)

            output_path = os.path.join(
                self.output_folder,
                f"labeled_{os.path.basename(image_path)}"
            )
            img.save(output_path)
            return output_path

    def _save_masked_version(self, image_path: str, masks: List[SegmentationMask]) -> str:
        """Save a version of the image with surrealist elements highlighted using masks."""
        with Image.open(image_path) as img:
            mask_overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(mask_overlay)

            for idx, mask in enumerate(masks):
                color = self.colors[idx % len(self.colors)]
                mask_img = Image.fromarray(mask.mask.astype('uint8') * 255)
                
                # Create semi-transparent overlay
                overlay_color = ImageColor.getrgb(color) + (128,)  # Add alpha channel
                draw.bitmap((0, 0), mask_img, fill=overlay_color)

            # Combine original image with mask overlay
            combined = Image.alpha_composite(
                img.convert('RGBA'),
                mask_overlay
            )

            output_path = os.path.join(
                self.output_folder,
                f"masked_{os.path.basename(image_path)}"
            )
            combined.save(output_path)
            return output_path
