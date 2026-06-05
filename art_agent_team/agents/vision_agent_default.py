import logging
import os # Added for makedirs
from typing import Optional, Dict, Any
from art_agent_team.agents.vision_agent_abstract import VisionAgentAbstract, CorruptedImageError, UnsupportedImageFormatError
from PIL import Image, UnidentifiedImageError # Added Image for local import consistency

class DefaultVisionAgent(VisionAgentAbstract):
    """
    Default VisionAgent used as a fallback when no genre-specific agent is found.
    Provides basic image processing and cropping logic.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        logging.info("DefaultVisionAgent initialized.")

    def process_image(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Basic vision processing: returns minimal analysis results.
        """
        logging.info(f"[DefaultVisionAgent] Processing image: {image_path}")
        try:
            # Minimal analysis: just return image path and size
            img = Image.open(image_path) # Uses PIL.Image
            results = {
                "image_path": image_path,
                "image_size": img.size,
                "note": "DefaultVisionAgent performed minimal processing."
            }
            return results
        except FileNotFoundError as e:
            logging.error(f"[DefaultVisionAgent] Image file not found in process_image: {image_path}. Error: {e}")
            raise
        except UnidentifiedImageError as e:
            logging.error(f"[DefaultVisionAgent] Cannot identify image file (corrupted/unsupported) in process_image: {image_path}. Error: {e}")
            raise CorruptedImageError(f"Corrupted or unsupported image file: {image_path}") from e
        except IOError as e:
            logging.error(f"[DefaultVisionAgent] IOError opening image in process_image: {image_path}. Error: {e}")
            raise UnsupportedImageFormatError(f"IOError, possibly unsupported image format: {image_path}") from e
        except Exception as e:
            logging.error(f"[DefaultVisionAgent] Failed to process image: {e}")
            # Consider re-raising a generic error or returning None based on desired strictness
            raise RuntimeError(f"Failed to process image {image_path} with DefaultVisionAgent") from e

    def analyze_image(self, image_path: str, research_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Basic analyze_image implementation for compatibility.
        """
        logging.info(f"[DefaultVisionAgent] Analyzing image: {image_path}")
        # No advanced analysis; just return image size and research_data passthrough
        try:
            img = Image.open(image_path) # Uses PIL.Image
            return {
                "image_size": img.size,
                "research_data": research_data,
                "note": "DefaultVisionAgent performed basic analysis."
            }
        except FileNotFoundError as e:
            logging.error(f"[DefaultVisionAgent] Image file not found in analyze_image: {image_path}. Error: {e}")
            raise
        except UnidentifiedImageError as e:
            logging.error(f"[DefaultVisionAgent] Cannot identify image file (corrupted/unsupported) in analyze_image: {image_path}. Error: {e}")
            raise CorruptedImageError(f"Corrupted or unsupported image file: {image_path}") from e
        except IOError as e:
            logging.error(f"[DefaultVisionAgent] IOError opening image in analyze_image: {image_path}. Error: {e}")
            raise UnsupportedImageFormatError(f"IOError, possibly unsupported image format: {image_path}") from e
        except Exception as e:
            logging.error(f"[DefaultVisionAgent] Failed to analyze image: {e}")
            raise RuntimeError(f"Failed to analyze image {image_path} with DefaultVisionAgent") from e

    def copy_and_crop_image(self, input_path: str, output_path: str, analysis_results: Optional[Dict[str, Any]] = None) -> str:
        """
        Basic center crop to 16:9 aspect ratio as a fallback.
        Returns the output_path (str) on success.
        Raises CorruptedImageError, UnsupportedImageFormatError, or FileNotFoundError on failure.
        """
        logging.info(f"[DefaultVisionAgent] Cropping image: {input_path} -> {output_path}")
        try:
            # Attempt to open the image first to catch format/corruption errors early
            try:
                img = Image.open(input_path) # Uses PIL.Image
            except UnidentifiedImageError as e:
                error_msg = f"Cannot identify image file (corrupted or unsupported format) for default crop: {input_path}. Error: {e}"
                logging.error(f"[DefaultVisionAgent] {error_msg}")
                raise CorruptedImageError(error_msg) from e
            except FileNotFoundError as e:
                error_msg = f"Input file not found for default cropping: {input_path}. Error: {e}"
                logging.error(f"[DefaultVisionAgent] {error_msg}")
                raise # Re-raise FileNotFoundError
            except IOError as e:
                error_msg = f"IOError opening image for default cropping (possibly unsupported format): {input_path}. Error: {e}"
                logging.error(f"[DefaultVisionAgent] {error_msg}")
                raise UnsupportedImageFormatError(error_msg) from e
                
            width, height = img.size
            target_ratio = 16 / 9
            current_ratio = width / height

            if current_ratio > target_ratio:
                new_width = int(height * target_ratio)
                new_height = height
            else:
                new_width = width
                new_height = int(width / target_ratio)

            if new_width <= 0 or new_height <= 0:
                error_msg = f"Invalid target dimensions calculated for default crop: {new_width}x{new_height}"
                logging.error(f"[DefaultVisionAgent] {error_msg}")
                raise ValueError(error_msg) # Consistent error raising

            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = left + new_width
            bottom = top + new_height

            if left < 0 or top < 0 or right > width or bottom > height or new_width <= 0 or new_height <= 0 :
                error_msg = f"Invalid crop dimensions for default crop: L{left} T{top} R{right} B{bottom} for image {width}x{height}. Saving original."
                logging.error(f"[DefaultVisionAgent] {error_msg}")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                img_rgb = img.convert('RGB') if img.mode == 'RGBA' else img
                img_rgb.save(output_path, format='JPEG', quality=95)
                # Still return output_path, but it's the original.
                # Consider if this specific case should also be an exception.
                return output_path

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cropped = img.crop((left, top, right, bottom))
            cropped = cropped.convert('RGB') if cropped.mode == 'RGBA' else cropped
            cropped.save(output_path, format='JPEG', quality=95)
            logging.info(f"[DefaultVisionAgent] Cropped image saved: {output_path}")
            return output_path
            
        except FileNotFoundError: # Already handled by the initial Image.open try-except
            raise
        except (CorruptedImageError, UnsupportedImageFormatError): # Re-raise custom exceptions
            raise
        except IOError as e: # Catch other IOErrors during save, etc.
            error_msg = f"IOError during default image cropping/saving: {input_path}. Error: {e}"
            logging.error(f"[DefaultVisionAgent] {error_msg}")
            raise UnsupportedImageFormatError(error_msg) from e
        except Exception as e: # General fallback
            error_msg = f"Unexpected error cropping with DefaultVisionAgent: {input_path}. Error: {e}"
            logging.error(f"[DefaultVisionAgent] {error_msg}", exc_info=True)
            raise RuntimeError(error_msg) from e