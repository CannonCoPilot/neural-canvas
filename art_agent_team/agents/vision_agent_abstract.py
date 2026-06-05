import os
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor, UnidentifiedImageError
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import abc # Abstract Base Classes

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CorruptedImageError(Exception):
    """Custom exception for corrupted images."""
    pass

class UnsupportedImageFormatError(Exception):
    """Custom exception for unsupported image formats."""
    pass

class InvalidMaskError(ValueError):
    """Custom exception for invalid segmentation masks."""
    pass

@dataclass(frozen=True)
class BoundingBoxRegion:
    """
    Data class for storing object bounding box information, typically with normalized vertices.
    """
    label: str
    score: float # Confidence score
    # Normalized vertices of the bounding box (0.0 to 1.0)
    # Expected as a list of vision.NormalizedVertex objects or similar (e.g., list of tuples (x,y))
    # Example: [(v.x, v.y) for v in google_cloud_vision_object.bounding_poly.normalized_vertices]
    normalized_vertices: List[Any] # List of tuples or objects with x, y attributes

    def get_absolute_vertices(self, img_width: int, img_height: int) -> List[Tuple[int, int]]:
        """Converts normalized vertices to absolute pixel coordinates."""
        abs_verts = []
        for v_point in self.normalized_vertices:
            try:
                # Attempt to access .x and .y if vertices are objects (like vision.NormalizedVertex)
                x = v_point.x
                y = v_point.y
            except AttributeError:
                # Assume v_point is a tuple (x, y)
                x, y = v_point
            abs_verts.append((int(x * img_width), int(y * img_height)))
        return abs_verts

    def get_simple_bounding_box(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """
        Calculates a simple (min_x, min_y, max_x, max_y) bounding box
        from the absolute vertices.
        """
        if not self.normalized_vertices:
            return (0,0,0,0)
        
        abs_verts = self.get_absolute_vertices(img_width, img_height)
        all_x = [v[0] for v in abs_verts]
        all_y = [v[1] for v in abs_verts]
        if not all_x or not all_y: # Should not happen if normalized_vertices is not empty
             return (0,0,0,0)
        
        x0 = min(all_x)
        y0 = min(all_y)
        x1 = max(all_x)
        y1 = max(all_y)
        return x0, y0, x1, y1


@dataclass(frozen=True)
class SegmentationMask:
    """Data class for storing segmentation mask information."""
    # Coordinates define the bounding box of the mask for efficient cropping/processing
    y0: int  # Top-left y of the mask's bounding box within the original image
    x0: int  # Top-left x of the mask's bounding box within the original image
    y1: int  # Bottom-right y of the mask's bounding box
    x1: int  # Bottom-right x of the mask's bounding box
    mask_data: np.ndarray  # Binary or grayscale mask [height, width] relative to the mask's bounding box
    label: str
    confidence: Optional[float] = None

    def __post_init__(self):
        if not (0 <= self.y0 < self.y1 and 0 <= self.x0 < self.x1):
            raise InvalidMaskError(f"Invalid bounding box coordinates for mask '{self.label}': ({self.x0},{self.y0})-({self.x1},{self.y1})")
        if self.mask_data.ndim != 2:
            raise InvalidMaskError(f"Mask data for '{self.label}' must be 2D, got {self.mask_data.ndim}D.")
        if self.mask_data.shape[0] != (self.y1 - self.y0) or self.mask_data.shape[1] != (self.x1 - self.x0):
            # This check assumes mask_data is already cropped to its bounding box.
            # If mask_data is full image size, this check needs adjustment or removal.
            # For now, let's assume mask_data is cropped.
            expected_shape = (self.y1 - self.y0, self.x1 - self.x0)
            # Relaxing this check for now as it depends on how mask_data is provided.
            # logging.warning(f"Mask data shape {self.mask_data.shape} for '{self.label}' does not match its bounding box size {expected_shape}. This might be intended if mask_data is not pre-cropped.")


class VisionAgentAbstract(abc.ABC):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config if config else {}
        self.output_folder = self.config.get('output_folder', 'output/vision_outputs')
        os.makedirs(self.output_folder, exist_ok=True)
        logging.info(f"Initializing {self.__class__.__name__}. Output folder: {self.output_folder}")

        # Standardized colors for visualizations
        self.colors = ['red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple',
                       'brown', 'gray', 'beige', 'turquoise', 'cyan', 'magenta',
                       'lime', 'navy', 'maroon', 'teal', 'olive', 'coral', 'lavender',
                       'violet', 'gold', 'silver'] + [c for c in ImageColor.colormap.keys() if c not in ['black', 'white']]
        
        try:
            self.font = ImageFont.truetype("DejaVuSans.ttf", 15)
        except IOError:
            self.font = ImageFont.load_default()

    @abc.abstractmethod
    def _perform_specific_analysis(self,
                                   image_path: str,
                                   image: Image.Image,
                                   artist_name: Optional[str] = None,
                                   art_movement: Optional[str] = None,
                                   title: Optional[str] = None,
                                   year: Optional[str] = None,
                                   research_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Abstract method for domain-specific image analysis.
        Subclasses must implement this to perform detection, segmentation, etc.
        Should return a dictionary containing 'bounding_boxes' (List[BoundingBoxRegion]),
        'segmentation_masks' (List[SegmentationMask]), and any other agent-specific analysis results.
        - 'bounding_boxes': List[BoundingBoxRegion]
        - 'segmentation_masks': List[SegmentationMask] (Note: Cloud Vision API may not provide these directly)
        - 'raw_output': Dict[str, Any] (Raw response from the vision API)
        - Other agent-specific keys.
        """
        pass

    def process(self,
                image_path: str,
                artist_name: Optional[str] = None,
                art_movement: Optional[str] = None,
                title: Optional[str] = None,
                year: Optional[str] = None,
                research_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main processing pipeline for an image.
        Loads the image, performs specific analysis, generates outputs, and returns results.
        """
        logging.info(f"{self.__class__.__name__} received image: {image_path}")
        try:
            with Image.open(image_path) as img:
                img_rgb = img.convert("RGB") # Ensure consistent format
                
                # Perform domain-specific analysis
                analysis_results = self._perform_specific_analysis(
                    image_path, img_rgb,
                    artist_name, art_movement, title, year,
                    research_data=research_data
                )
                
                # Generate standard outputs
                output_paths = self.generate_outputs(image_path, img_rgb,
                                                     analysis_results.get('bounding_boxes', []),
                                                     analysis_results.get('segmentation_masks', []))
                
                # Combine analysis results with output paths
                final_result = {
                    "description": f"{self.__class__.__name__} processing complete.",
                    "specific_analysis": analysis_results,
                    "output_image_paths": output_paths,
                    "raw_output": analysis_results.get("raw_output", {}) # Include raw output if provided by subclass
                }
                return final_result

        except FileNotFoundError:
            logging.error(f"Image file not found: {image_path}")
            raise
        except (UnidentifiedImageError, OSError) as e:
            logging.error(f"Corrupted or unsupported image {image_path}: {e}")
            raise CorruptedImageError(f"Corrupted or unsupported image {image_path}: {e}")
        except InvalidMaskError as e:
            logging.error(f"Invalid mask data for {image_path}: {e}")
            raise # Re-raise to be handled by caller or higher-level error handling
        except Exception as e:
            logging.error(f"Unexpected error processing image {image_path} in {self.__class__.__name__}: {e}")
            raise


    def _create_bounding_box_image(self, original_image: Image.Image, bounding_box_regions: List[BoundingBoxRegion], base_filename: str) -> Optional[str]:
        """Creates and saves an image with bounding boxes drawn using BoundingBoxRegion."""
        if not bounding_box_regions:
            return None
        
        img_copy = original_image.copy().convert("RGBA") # Ensure RGBA for drawing
        draw = ImageDraw.Draw(img_copy)
        img_width, img_height = original_image.size
        
        for i, bbox_region in enumerate(bounding_box_regions):
            color = self.colors[i % len(self.colors)]
            
            abs_vertices = bbox_region.get_absolute_vertices(img_width, img_height)

            if not abs_vertices or len(abs_vertices) < 2:
                logging.warning(f"Skipping drawing for {bbox_region.label} due to insufficient vertices.")
                continue

            # Draw polygon if more than 2 distinct points, otherwise a line or point (or simple box)
            # For simplicity, if it's 4 vertices and they form a rectangle, draw.rectangle is fine.
            # Otherwise, draw.polygon. For landmarks, it's usually a polygon.
            # For object localization, it's often a rectangle represented by 4 vertices.
            
            # Use get_simple_bounding_box for text positioning and fallback rectangle
            x0_simple, y0_simple, x1_simple, y1_simple = bbox_region.get_simple_bounding_box(img_width, img_height)

            if x0_simple >= x1_simple or y0_simple >= y1_simple: # Skip invalid boxes
                logging.warning(f"Skipping drawing invalid bounding box for {bbox_region.label}: ({x0_simple},{y0_simple})-({x1_simple},{y1_simple})")
                continue

            # Check if vertices form a clear polygon or just a rectangle
            # A simple check: if number of vertices is 4 and they are axis-aligned (min/max x,y match corners)
            # For now, let's always use polygon for flexibility, it handles rectangles too.
            if len(abs_vertices) >= 3: # Need at least 3 points for a polygon
                draw.polygon(abs_vertices, outline=color, width=3)
            elif len(abs_vertices) == 2: # Draw a line if only two points
                draw.line(abs_vertices, fill=color, width=3)
            else: # Fallback to simple rectangle if not enough points for polygon/line
                 draw.rectangle([(x0_simple, y0_simple), (x1_simple, y1_simple)], outline=color, width=3)

            label_text = f"{bbox_region.label}"
            if bbox_region.score is not None: # score is the confidence
                label_text += f" ({bbox_region.score:.2f})"
            
            # Position text based on the simple bounding box's top-left corner
            text_y_pos = y0_simple - self.font.getbbox(label_text)[3] - 2
            if text_y_pos < 2 :
                text_y_pos = y0_simple + 2 if (y1_simple - y0_simple) > (self.font.getbbox(label_text)[3] + 4) else y1_simple + 2
            text_position = (x0_simple + 2, text_y_pos)
            
            # Simple background for text for better readability
            try:
                text_render_bbox = draw.textbbox(text_position, label_text, font=self.font)
                draw.rectangle(text_render_bbox, fill=ImageColor.getrgb(color) + (100,)) # Semi-transparent background
            except Exception as e_text_bg:
                logging.debug(f"Could not draw text background for {label_text}: {e_text_bg}")

            draw.text(text_position, label_text, fill="black", font=self.font) # Draw text on top
            
        output_path = os.path.join(self.output_folder, f"{base_filename}_bboxes.png")
        img_copy.convert("RGB").save(output_path) # Save as RGB
        logging.info(f"Saved bounding box image to {output_path}")
        return output_path

    def _create_segmentation_mask_image(self, original_image: Image.Image, segmentation_masks: List[SegmentationMask], base_filename: str) -> Optional[str]:
        """Creates and saves an image with segmentation masks overlaid."""
        if not segmentation_masks:
            return None

        img_copy_rgba = original_image.copy().convert("RGBA")
        overlay = Image.new("RGBA", img_copy_rgba.size, (0,0,0,0))
        draw_overlay = ImageDraw.Draw(overlay)

        for i, seg_mask in enumerate(segmentation_masks):
            # Validate mask shape against its bounding box
            expected_mask_shape = (seg_mask.y1 - seg_mask.y0, seg_mask.x1 - seg_mask.x0)
            if seg_mask.mask_data.shape != expected_mask_shape:
                logging.warning(f"Mask data shape {seg_mask.mask_data.shape} for '{seg_mask.label}' does not match its bounding box size {expected_mask_shape}. Resizing mask data to fit bounding box.")
                try:
                    mask_pil = Image.fromarray(seg_mask.mask_data.astype(np.uint8))
                    resized_mask_pil = mask_pil.resize(expected_mask_shape[::-1], Image.NEAREST) # (width, height)
                    mask_data_resized = np.array(resized_mask_pil)
                except Exception as e:
                    logging.error(f"Failed to resize mask data for '{seg_mask.label}': {e}. Skipping this mask.")
                    continue # Skip this mask if resizing fails
            else:
                mask_data_resized = seg_mask.mask_data


            # Ensure mask_data is boolean or 0-255 uint8 for proper PIL processing
            if mask_data_resized.dtype == bool:
                pil_mask = Image.fromarray(mask_data_resized) # Creates '1' mode image
            elif mask_data_resized.dtype == np.uint8:
                pil_mask = Image.fromarray(mask_data_resized, mode='L') # 'L' for grayscale
            else:
                logging.warning(f"Unsupported mask data type {mask_data_resized.dtype} for '{seg_mask.label}'. Converting to uint8. Data loss may occur.")
                pil_mask = Image.fromarray(mask_data_resized.astype(np.uint8), mode='L')

            color = self.colors[i % len(self.colors)]
            # Create a colored version of the mask for the overlay
            # The mask itself (pil_mask) is used as the alpha channel for the colored region
            colored_mask_fill = Image.new("RGBA", (seg_mask.x1 - seg_mask.x0, seg_mask.y1 - seg_mask.y0), color)
            
            # Paste the colored mask onto the overlay, using pil_mask as the alpha.
            # The position for pasting is (seg_mask.x0, seg_mask.y0)
            overlay.paste(colored_mask_fill, (seg_mask.x0, seg_mask.y0), pil_mask.convert('L')) # Ensure mask is 'L' for alpha

            # Optionally draw label
            label_text = f"{seg_mask.label}"
            if seg_mask.confidence is not None:
                label_text += f" ({seg_mask.confidence:.2f})"
            text_position = (seg_mask.x0 + 2, seg_mask.y0 + 2) # Adjust as needed
            draw_overlay.text(text_position, label_text, fill=color, font=self.font)


        # Composite the overlay with the original image
        combined_image = Image.alpha_composite(img_copy_rgba, overlay)
        output_path = os.path.join(self.output_folder, f"{base_filename}_segmentation.png")
        combined_image.convert("RGB").save(output_path) # Save as RGB
        logging.info(f"Saved segmentation mask image to {output_path}")
        return output_path

    def _create_cropped_images(self, original_image: Image.Image, items_to_crop: List[Any], base_filename: str, crop_type: str) -> List[str]:
        """
        Creates and saves cropped images based on BoundingBoxRegion or SegmentationMask.
        `items_to_crop` can be List[BoundingBoxRegion] or List[SegmentationMask].
        `crop_type` is 'bbox' or 'segmask' for filename differentiation.
        """
        cropped_image_paths = []
        if not items_to_crop:
            return cropped_image_paths
        
        img_width, img_height = original_image.size

        for i, item in enumerate(items_to_crop):
            label = "unknown"
            crop_coords = None

            if isinstance(item, BoundingBoxRegion):
                # Use get_simple_bounding_box to get absolute pixel coordinates (x0,y0,x1,y1)
                crop_coords = item.get_simple_bounding_box(img_width, img_height)
                label = item.label
            elif isinstance(item, SegmentationMask):
                crop_coords = (item.x0, item.y0, item.x1, item.y1) # SegmentationMask uses absolute coords for its bbox
                label = item.label
            else:
                logging.warning(f"Unsupported item type for cropping: {type(item)}. Skipping.")
                continue
            
            safe_label = "".join(c if c.isalnum() else "_" for c in label)
            
            try:
                x0, y0, x1, y1 = crop_coords
                
                # Validate and clamp crop coordinates
                clamped_x0 = max(0, x0)
                clamped_y0 = max(0, y0)
                clamped_x1 = min(img_width, x1)
                clamped_y1 = min(img_height, y1)

                if clamped_x0 >= clamped_x1 or clamped_y0 >= clamped_y1:
                    logging.warning(f"Skipping crop for '{label}' due to invalid/clamped coordinates: original ({x0},{y0})-({x1},{y1}), clamped ({clamped_x0},{clamped_y0})-({clamped_x1},{clamped_y1}) on image size ({img_width},{img_height}).")
                    continue

                cropped_img = original_image.crop((clamped_x0, clamped_y0, clamped_x1, clamped_y1))
                
                cropped_img_to_save = cropped_img # Default
                
                # If it's a segmentation mask, apply the mask to make background transparent
                if isinstance(item, SegmentationMask) and self.config.get('apply_mask_to_cropped_segmentation', True):
                    # Mask data in SegmentationMask is relative to its own bounding box (item.x0, item.y0, item.x1, item.y1)
                    # The size of cropped_img should match the size of item.mask_data
                    expected_mask_shape = (clamped_y1 - clamped_y0, clamped_x1 - clamped_x0)
                    
                    if item.mask_data.shape != expected_mask_shape:
                         logging.warning(f"Cropped segment mask data shape {item.mask_data.shape} for '{item.label}' does not match its target crop size {expected_mask_shape}. Resizing mask data.")
                         mask_pil_temp = Image.fromarray(item.mask_data.astype(np.uint8))
                         # Resize to the actual dimensions of the cropped_img
                         resized_mask_pil_temp = mask_pil_temp.resize((cropped_img.width, cropped_img.height), Image.NEAREST)
                         mask_data_for_alpha = np.array(resized_mask_pil_temp)
                    else:
                        mask_data_for_alpha = item.mask_data

                    if mask_data_for_alpha.dtype == bool:
                        pil_alpha_mask = Image.fromarray(mask_data_for_alpha) # '1' mode
                    else: # Assuming uint8
                        pil_alpha_mask = Image.fromarray(mask_data_for_alpha.astype(np.uint8), mode='L')
                    
                    cropped_img_rgba = cropped_img.convert("RGBA")
                    cropped_img_rgba.putalpha(pil_alpha_mask)
                    cropped_img_to_save = cropped_img_rgba
                
                crop_filename = f"{base_filename}_{crop_type}_{i}_{safe_label}.png"
                output_path = os.path.join(self.output_folder, crop_filename)
                cropped_img_to_save.save(output_path)
                cropped_image_paths.append(output_path)
            except Exception as e:
                logging.error(f"Error cropping for label '{label}' ({crop_type}): {e}", exc_info=True)
        
        if cropped_image_paths:
            logging.info(f"Saved {len(cropped_image_paths)} cropped {crop_type} images to {self.output_folder}")
        return cropped_image_paths

    def generate_outputs(self, image_path: str, original_image: Image.Image,
                         bounding_boxes: List[BoundingBoxRegion], # Updated type hint
                         segmentation_masks: List[SegmentationMask]) -> Dict[str, Any]:
        """
        Coordinates the creation of all standard image outputs.
        - Image with bounding boxes.
        - Image with segmentation masks.
        - Cropped images from bounding boxes.
        - Cropped images from segmentation masks.
        """
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        output_paths = {}

        # 1. Bounding Box Image & Cropped BBoxes
        if bounding_boxes:
            output_paths['bounding_box_visualization'] = self._create_bounding_box_image(original_image, bounding_boxes, base_filename)
            output_paths['cropped_from_bboxes'] = self._create_cropped_images(original_image, bounding_boxes, base_filename, "bbox")

        # 2. Segmentation Mask Image & Cropped Segments
        if segmentation_masks:
            # Validate segmentation masks before processing
            valid_masks = []
            for mask in segmentation_masks:
                try:
                    # Basic validation: mask_data should be 2D numpy array
                    if not isinstance(mask.mask_data, np.ndarray) or mask.mask_data.ndim != 2:
                        raise InvalidMaskError(f"Mask data for '{mask.label}' is not a 2D numpy array.")
                    # Check if mask dimensions match image dimensions if it's a full mask,
                    # or if it matches its own bounding box if it's a cropped mask.
                    # This logic is now more integrated into _create_segmentation_mask_image and _create_cropped_images
                    valid_masks.append(mask)
                except InvalidMaskError as e:
                    logging.error(f"Skipping invalid segmentation mask for '{mask.label}': {e}")
            
            if valid_masks:
                output_paths['segmentation_visualization'] = self._create_segmentation_mask_image(original_image, valid_masks, base_filename)
                output_paths['cropped_from_segmentation'] = self._create_cropped_images(original_image, valid_masks, base_filename, "segmask")
        
        return output_paths

if __name__ == '__main__':
    # This is an abstract class and cannot be instantiated directly for a full run.
    # Example of how a subclass might be structured:

    class DummyVisionAgent(VisionAgentAbstract):
        def _perform_specific_analysis(self, image_path: str, image: Image.Image, artist_name: Optional[str] = None, art_movement: Optional[str] = None, title: Optional[str] = None, year: Optional[str] = None) -> Dict[str, Any]:
            logging.info(f"DummyVisionAgent performing analysis on {image_path}")
            img_width, img_height = image.size
            
            # Dummy bounding boxes
            bboxes = [
                BoundingBox(x0=int(img_width*0.1), y0=int(img_height*0.1), x1=int(img_width*0.4), y1=int(img_height*0.4), label="object1", confidence=0.9),
                BoundingBox(x0=int(img_width*0.5), y0=int(img_height*0.5), x1=int(img_width*0.8), y1=int(img_height*0.8), label="object2", confidence=0.8)
            ]
            
            # Dummy segmentation masks (mask_data should be cropped to the bbox of the mask)
            mask1_y0, mask1_x0, mask1_y1, mask1_x1 = int(img_height*0.15), int(img_width*0.15), int(img_height*0.35), int(img_width*0.35)
            mask1_data = np.zeros((mask1_y1-mask1_y0, mask1_x1-mask1_x0), dtype=bool)
            mask1_data[int((mask1_y1-mask1_y0)*0.2):int((mask1_y1-mask1_y0)*0.8), int((mask1_x1-mask1_x0)*0.2):int((mask1_x1-mask1_x0)*0.8)] = True # A smaller square within

            mask2_y0, mask2_x0, mask2_y1, mask2_x1 = int(img_height*0.55), int(img_width*0.55), int(img_height*0.75), int(img_width*0.75)
            mask2_data = np.zeros((mask2_y1-mask2_y0, mask2_x1-mask2_x0), dtype=np.uint8) # Example with uint8
            cv2.ellipse(mask2_data, (int((mask2_x1-mask2_x0)/2), int((mask2_y1-mask2_y0)/2)), (int((mask2_x1-mask2_x0)*0.4), int((mask2_y1-mask2_y0)*0.3)), 0, 0, 360, 255, -1) # Ellipse, requires cv2

            seg_masks = []
            try:
                import cv2 # For dummy ellipse mask
                seg_masks.append(SegmentationMask(y0=mask1_y0, x0=mask1_x0, y1=mask1_y1, x1=mask1_x1, mask_data=mask1_data, label="segment1", confidence=0.95))
                seg_masks.append(SegmentationMask(y0=mask2_y0, x0=mask2_x0, y1=mask2_y1, x1=mask2_x1, mask_data=mask2_data, label="segment2", confidence=0.85))
            except ImportError:
                logging.warning("cv2 not installed, skipping ellipse dummy mask for SegmentationMask example.")
                seg_masks.append(SegmentationMask(y0=mask1_y0, x0=mask1_x0, y1=mask1_y1, x1=mask1_x1, mask_data=mask1_data, label="segment1", confidence=0.95))


            return {
                "description": "Dummy analysis complete.",
                "genre_confidence": 0.75,
                "identified_elements": ["elementA", "elementB"],
                "bounding_boxes": bboxes,
                "segmentation_masks": seg_masks,
                "raw_output": {"dummy_key": "dummy_value"}
            }

    # Example usage (requires a test image)
    if __name__ == '__main__':
        # Create a dummy image for testing
        dummy_image_path = "dummy_test_image.png"
        try:
            img = Image.new('RGB', (600, 400), color = 'skyblue')
            d = ImageDraw.Draw(img)
            d.text((10,10), "Test Image", fill=(255,255,0))
            d.rectangle([(50,50), (200,150)], fill="red", outline="black")
            d.ellipse([(250,200), (350,300)], fill="green", outline="blue")
            img.save(dummy_image_path)
            logging.info(f"Created dummy test image: {dummy_image_path}")

            # Test with DummyVisionAgent
            # Note: You might need to pip install opencv-python for the cv2 ellipse in dummy data
            agent_config = {"output_folder": "output/dummy_agent_outputs"}
            dummy_agent = DummyVisionAgent(config=agent_config)
            
            try:
                result = dummy_agent.process(dummy_image_path, artist_name="Dummy Artist", title="Dummy Artwork")
                logging.info(f"Dummy Agent processing result: {result}")
            except Exception as e:
                logging.error(f"Error during DummyVisionAgent processing: {e}", exc_info=True)
            finally:
                # Clean up dummy image
                if os.path.exists(dummy_image_path):
                    os.remove(dummy_image_path)
                    logging.info(f"Removed dummy test image: {dummy_image_path}")

        except ImportError:
            logging.warning("Pillow (PIL) or OpenCV (cv2) might not be installed. Dummy agent test may not run fully.")
        except Exception as e:
            logging.error(f"Error in __main__ setup for VisionAgentAbstract: {e}", exc_info=True)
