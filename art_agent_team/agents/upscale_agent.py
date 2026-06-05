from typing import Optional, Dict, Any
import os
import io # Needed for in-memory image manipulation
from PIL import Image # Needed for image resizing
import logging
# from torchvision import transforms # No longer needed for ESRGAN

import time
import urllib.parse
import requests
from image_upscaling_api import upload_image, get_uploaded_images # Assuming this package is installed
import math # For sqrt

class UpscaleAgent:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the UpscaleAgent.
        Loads configuration for image-upscaling.net and Stability AI.
        Requires 'image_upscaling_id' for image-upscaling.net.
        Requires 'stablediffusion_api_key' for Stability AI.
        'upscaler_preference' in config can be 'image_upscaling_net', 'stability_ai', or 'fallback'.
        """
        self.image_upscaling_net_ready = False
        self.stability_ai_ready = False
        self.client_id = None
        self.stability_api_key = None
        self.upscaler_preference = "fallback" # Default preference

        if not config:
            config = {}  # Ensure config is a dict
            logging.warning("[UpscaleAgent] Config not provided during initialization.")

        # Configure image-upscaling.net
        self.client_id = config.get('image_upscaling_id')
        if self.client_id:
            self.image_upscaling_net_ready = True
            logging.info(f"[UpscaleAgent] Initialized for image-upscaling.net with client ID.")
        else:
            logging.warning("[UpscaleAgent] 'image_upscaling_id' not found in config. image-upscaling.net API will be unavailable.")

        # Configure Stability AI
        self.stability_api_key = config.get('stablediffusion_api_key') # Assuming key name from provided config
        if self.stability_api_key:
            self.stability_ai_ready = True
            logging.info(f"[UpscaleAgent] Initialized for Stability AI with API key.")
        else:
            logging.warning("[UpscaleAgent] 'stablediffusion_api_key' not found in config. Stability AI upscaling will be unavailable.")

        self.upscaler_preference = config.get('upscaler_preference', 'fallback').lower()
        if self.upscaler_preference not in ['image_upscaling_net', 'stability_ai', 'fallback']:
            logging.warning(f"[UpscaleAgent] Invalid 'upscaler_preference' value: {self.upscaler_preference}. Defaulting to 'fallback'.")
            self.upscaler_preference = 'fallback'
        
        logging.info(f"[UpscaleAgent] Upscaler preference set to: {self.upscaler_preference}")

        if not self.image_upscaling_net_ready and not self.stability_ai_ready:
            logging.error("[UpscaleAgent] ERROR: No upscaling APIs are configured. Upscaling will be unavailable.")
            return

    # def _load_esrgan_model(self):
    #     """
    #     Load the ESRGAN model from the specified path.
    #     This is a placeholder for the actual model loading logic, which should be implemented
    #     based on the specific ESRGAN architecture and weights format.
    #     """
    #     # Placeholder for loading ESRGAN model
    #     # In a real implementation, this would load the model weights into a suitable architecture
    #     # For example, using a library like https://github.com/xinntao/ESRGAN
    #     # logging.info(f"[UpscaleAgent] Loading ESRGAN model (placeholder) from '{self.model_path}'")
    #     # model = torch.load(self.model_path, map_location=self.device)
    #     # model.eval()
    #     # return model

    def _upscale_with_image_upscaling_net(self, input_path: str, output_path: str) -> Optional[str]:
        """
        Upscales an image using the image-upscaling.net API.
        """
        if not self.image_upscaling_net_ready:
            logging.warning(f"[UpscaleAgent] image-upscaling.net API not ready. Skipping.")
            return None

        logging.info(f"[UpscaleAgent] Starting image-upscaling.net API upscale for '{input_path}'...")
        original_filename_base = os.path.splitext(os.path.basename(input_path))[0]

        try:
            logging.info(f"[UpscaleAgent] Uploading '{input_path}' to image-upscaling.net API.")
            upload_image(
                input_path,
                self.client_id,
                scale=4,  # For 4K upscaling
                use_face_enhance=False
            )
            logging.info(f"[UpscaleAgent] Image upload initiated for '{input_path}' via image-upscaling.net.")

            polling_start_time = time.time()
            polling_timeout = 300 # 5 minutes timeout for polling

            logging.info("[UpscaleAgent] Polling image-upscaling.net for upscaled image completion...")
            while True:
                if time.time() - polling_start_time > polling_timeout:
                    logging.error(f"[UpscaleAgent] Timeout waiting for image-upscaling.net processing for '{input_path}'.")
                    return None

                waiting, completed, in_progress = get_uploaded_images(self.client_id)
                logging.debug(f"[UpscaleAgent] image-upscaling.net Poll status - Waiting: {len(waiting)}, In Progress: {len(in_progress)}, Completed: {len(completed)}")

                processed_url = None
                for url_info in completed:
                    url_str = url_info if isinstance(url_info, str) else url_info.get('url', '')
                    if original_filename_base in url_str and "_preview_" not in url_str:
                        processed_url = url_str
                        break
                
                if processed_url:
                    logging.info(f"[UpscaleAgent] image-upscaling.net processing completed. Found URL: {processed_url}")
                    filename_from_url = processed_url.split("/")[-1]
                    download_url = (
                        "https://image-upscaling.net/imageupscaling/download_and_delete_image.php?clientid="
                        + self.client_id
                        + "&file="
                        + urllib.parse.quote(filename_from_url)
                    )
                    logging.info(f"[UpscaleAgent] Downloading upscaled image from image-upscaling.net: {download_url}")
                    
                    response = requests.get(download_url, stream=True, verify=False) # verify=False due to known SSL issues
                    response.raise_for_status()

                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with open(output_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    logging.info(f"[UpscaleAgent] Successfully saved upscaled image from image-upscaling.net to '{output_path}'.")
                    return output_path

                if not waiting and not in_progress and not processed_url:
                    logging.warning(f"[UpscaleAgent] image-upscaling.net Polling: No images waiting or in progress, and target not found. Continuing poll.")

                time.sleep(10) # Polling interval increased slightly

        except requests.exceptions.RequestException as e:
            logging.error(f"[UpscaleAgent] ERROR: Network error during image-upscaling.net interaction for '{input_path}': {e}")
            return None
        except Exception as e:
            logging.error(f"[UpscaleAgent] ERROR: An unexpected error occurred during image-upscaling.net upscaling for '{input_path}': {e}")
            return None
        return None # Should be unreachable if logic is correct, but as a safeguard.

    def _upscale_with_stability_ai(self, input_path: str, output_path: str) -> Optional[str]:
        """
        Upscales an image using the Stability AI API (fast upscaler).
        """
        if not self.stability_ai_ready:
            logging.warning(f"[UpscaleAgent] Stability AI API not ready. Skipping.")
            return None

        if not self.stability_api_key:
            logging.error("[UpscaleAgent] Stability AI API key not configured.")
            return None

        logging.info(f"[UpscaleAgent] Starting Stability AI upscale for '{input_path}'...")
        
        output_format = os.path.splitext(output_path)[1].lower().replace('.', '')
        if output_format not in ['png', 'jpeg', 'jpg', 'webp']:
            logging.warning(f"[UpscaleAgent] Unsupported output format '{output_format}' for Stability AI. Defaulting to 'png'.")
            output_format = 'png'
            # Adjust output_path to reflect the actual format being requested if it was changed.
            output_path = os.path.splitext(output_path)[0] + '.png'


        if output_format == 'jpg': # Stability AI uses 'jpeg'
            output_format = 'jpeg'

        try:
            max_pixels = 1048576  # Stability AI API limit
            img = Image.open(input_path)
            width, height = img.size
            current_pixels = width * height

            image_data_to_send = None

            if current_pixels > max_pixels:
                logging.info(f"[UpscaleAgent] Image '{input_path}' ({width}x{height}, {current_pixels} pixels) exceeds Stability AI limit of {max_pixels} pixels. Resizing.")
                scale_ratio = math.sqrt(max_pixels / current_pixels)
                new_width = int(width * scale_ratio)
                new_height = int(height * scale_ratio)
                
                # Ensure at least 1x1 dimensions after scaling
                new_width = max(1, new_width)
                new_height = max(1, new_height)

                img_resized = img.resize((new_width, new_height), Image.LANCZOS)
                logging.info(f"[UpscaleAgent] Resized image to {new_width}x{new_height} ({new_width * new_height} pixels).")
                
                # Save resized image to an in-memory buffer
                buffer = io.BytesIO()
                # Determine format for saving to buffer, use original if possible, else PNG
                original_format = img.format if img.format else 'PNG'
                if original_format.upper() == 'JPEG':
                    # For JPEG, ensure RGB mode if it was RGBA (to avoid issues with saving alpha)
                    if img_resized.mode == 'RGBA':
                        img_resized = img_resized.convert('RGB')
                img_resized.save(buffer, format=original_format)
                buffer.seek(0)
                image_data_to_send = buffer
            else:
                logging.info(f"[UpscaleAgent] Image '{input_path}' ({width}x{height}, {current_pixels} pixels) is within Stability AI limits. No resize needed.")
                # Use original image directly
                # image_data_to_send will be opened from input_path below

            files_payload = {}
            if image_data_to_send: # If image was resized and is in buffer
                files_payload["image"] = ("resized_image." + (img.format.lower() if img.format else "png"), image_data_to_send, img.get_format_mimetype() or f"image/{img.format.lower() or 'png'}")
                
                response = requests.post(
                    "https://api.stability.ai/v2beta/stable-image/upscale/fast",
                    headers={
                        "authorization": f"Bearer {self.stability_api_key}",
                        "accept": "image/*"
                    },
                    files=files_payload,
                    data={
                        "output_format": output_format
                    }
                )
            else: # Use original image from path
                with open(input_path, "rb") as f_image:
                    files_payload["image"] = f_image
                    response = requests.post(
                        "https://api.stability.ai/v2beta/stable-image/upscale/fast",
                        headers={
                            "authorization": f"Bearer {self.stability_api_key}",
                            "accept": "image/*"
                        },
                        files=files_payload,
                        data={
                            "output_format": output_format
                        }
                    )

            response.raise_for_status() # Raise an exception for HTTP errors

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logging.info(f"[UpscaleAgent] Successfully saved Stability AI upscaled image to '{output_path}'.")
            return output_path

        except requests.exceptions.RequestException as e:
            logging.error(f"[UpscaleAgent] ERROR: Network error during Stability AI interaction for '{input_path}': {e}")
            if e.response is not None:
                logging.error(f"[UpscaleAgent] Stability AI Response: {e.response.text}")
            return None
        except Exception as e:
            logging.error(f"[UpscaleAgent] ERROR: An unexpected error occurred during Stability AI upscaling for '{input_path}': {e}")
            return None

    def upscale_image(self, input_path: str, output_path: str, target_width: int = 3840, target_height: int = 2160) -> Optional[str]:
        """
        Upscale the input image using the configured API.
        Saves the upscaled image to output_path.
        target_width and target_height are for interface consistency, actual scaling depends on API.
        """
        logging.info(f"[UpscaleAgent] Received request to upscale '{input_path}' to '{output_path}'. Target dimensions: {target_width}x{target_height}.")
        if not os.path.exists(input_path):
            logging.error(f"[UpscaleAgent] ERROR: Input file not found at '{input_path}'. Upscaling aborted.")
            return None

        if not self.image_upscaling_net_ready and not self.stability_ai_ready:
            logging.error("[UpscaleAgent] No upscaling APIs are configured or ready. Cannot upscale. Upscaling aborted.")
            return None
        
        logging.info(f"[UpscaleAgent] Attempting to upscale '{input_path}' to '{output_path}' using preference: {self.upscaler_preference}")
        
        result_path = None
        if self.upscaler_preference == "image_upscaling_net":
            if self.image_upscaling_net_ready:
                result_path = self._upscale_with_image_upscaling_net(input_path, output_path)
            else:
                logging.warning("[UpscaleAgent] image-upscaling.net preferred but not ready. No fallback configured for this preference. Upscaling aborted for this preference.")
                result_path = None
        
        elif self.upscaler_preference == "stability_ai":
            if self.stability_ai_ready:
                result_path = self._upscale_with_stability_ai(input_path, output_path)
            else:
                logging.warning("[UpscaleAgent] Stability AI preferred but not ready. No fallback configured for this preference. Upscaling aborted for this preference.")
                result_path = None

        elif self.upscaler_preference == "fallback":
            # result = None # Renamed to result_path for consistency
            if self.image_upscaling_net_ready:
                logging.info("[UpscaleAgent] Fallback: Trying image-upscaling.net API first.")
                result_path = self._upscale_with_image_upscaling_net(input_path, output_path)
                if result_path:
                    logging.info(f"[UpscaleAgent] Upscaling successful with image-upscaling.net. Output: {result_path}")
                    return result_path # Return early on success
                logging.warning("[UpscaleAgent] Fallback: image-upscaling.net API failed or not configured. Trying Stability AI.")
            
            if self.stability_ai_ready:
                logging.info("[UpscaleAgent] Fallback: Trying Stability AI API.")
                result_path = self._upscale_with_stability_ai(input_path, output_path)
                if result_path:
                    logging.info(f"[UpscaleAgent] Upscaling successful with Stability AI. Output: {result_path}")
                    return result_path # Return early on success
                logging.warning("[UpscaleAgent] Fallback: Stability AI API also failed or not configured.")
            
            if not self.image_upscaling_net_ready and not self.stability_ai_ready:
                 logging.error("[UpscaleAgent] Fallback: Neither API is available/ready. Upscaling failed.")
            elif not result_path: # Check result_path here
                 logging.error(f"[UpscaleAgent] Fallback: All available upscaling attempts failed for '{input_path}'. Upscaling failed.")
            # return result # Renamed to result_path

        else: # Should not happen due to __init__ check
            logging.error(f"[UpscaleAgent] Invalid upscaler_preference '{self.upscaler_preference}'. Upscaling aborted.")
            result_path = None

        if result_path:
            logging.info(f"[UpscaleAgent] Upscaling process completed for '{input_path}'. Output saved to '{result_path}'.")
        else:
            logging.error(f"[UpscaleAgent] Upscaling process failed for '{input_path}'.")
        return result_path

# Example usage for testing (Commented out as per transition to API)
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
#     # Ensure you have a config.yaml or provide the client_id directly for testing
#     # Example: Create a dummy config.yaml with:
#     # image_upscaling_id: "your_32_digit_hex_client_id"
    
#     import yaml
#     config_path = '../../art_agent_team/config/config.yaml' # Adjust path as needed
#     test_config = {}
#     try:
#         with open(config_path, 'r') as f:
#             test_config = yaml.safe_load(f)
#     except FileNotFoundError:
#         print(f"Test config file not found at {config_path}. Please create it with 'image_upscaling_id'.")
#         exit()
#     if 'image_upscaling_id' not in test_config:
#         print("'image_upscaling_id' not in test_config. Please add it.")
#         exit()

#     agent = UpscaleAgent(config=test_config)

#     if agent.api_ready:
#         # Create a dummy input image for testing
#         dummy_input_path = "test_input_upscale.png"
#         dummy_output_path = "test_output_api_upscaled.png"

#         if not os.path.exists(dummy_input_path):
#             try:
#                 from PIL import Image as PILImage
#                 dummy_img = PILImage.new('RGB', (100, 100), color = 'red')
#                 dummy_img.save(dummy_input_path)
#                 print(f"Created dummy input image: {dummy_input_path}")
#             except ImportError:
#                 print("Pillow not installed, cannot create dummy image. Please create test_input_upscale.png manually.")
#                 exit()
#             except Exception as e:
#                 print(f"Failed to create dummy image: {e}")
#                 exit()
        
#         print(f"Attempting to upscale '{dummy_input_path}' to '{dummy_output_path}'...")
#         output_file = agent.upscale_image(dummy_input_path, dummy_output_path)
        
#         if output_file:
#             print(f"Upscaled image saved to {output_file}")
#         else:
#             print("Upscaling failed.")
#     else:
#         print("UpscaleAgent not ready. Check config and client ID.")
#     pass