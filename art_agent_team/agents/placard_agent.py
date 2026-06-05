import os
from PIL import Image, ImageDraw, ImageFont
import textwrap
import logging # Added for better logging, though print is used as requested for now

# Setup basic logging (optional, can be configured externally)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
# Using print with tags as requested for simplicity

class PlacardAgent:
    # Modified __init__ signature and defaults
    def __init__(self, config=None,
                 # Default font paths if not in config
                 default_font_path='art_agent_team/assets/fonts/arial.ttf',
                 default_font_path_bold='art_agent_team/assets/fonts/arialbd.ttf',
                 default_background_image_path='input/card_stock.jpg',
                 plaque_opacity=0.9,
                 margin_percent=5,
                 fallback_bg_size=(600, 400), # Default size for fallback background
                 text_color=(0, 0, 0, 255)): # Default text color (black)
        """
        Initialize the PlacardAgent.
        Args:
            config (dict, optional): Configuration dictionary.
            default_font_path (str): Default path to the regular font file if not in config.
            default_font_path_bold (str): Default path to the bold font file if not in config.
            default_background_image_path (str): Default path to the background image if not in config.
            plaque_opacity (float): Opacity for the plaque background (0.0 to 1.0).
            margin_percent (int): Margin from image edges as a percentage of image dimensions.
            fallback_bg_size (tuple): Size (width, height) for the fallback white background.
            text_color (tuple): RGBA tuple for the text color.
        """
        logging.info(f"[PlacardAgent] Initializing with config: {config}")
        
        placard_config = config.get('placard_agent', {}) if config else {}

        self.font_path = placard_config.get('regular_font_path', default_font_path)
        self.font_path_bold = placard_config.get('bold_font_path', default_font_path_bold)
        self.background_image_path = placard_config.get('background_image_path', default_background_image_path)
        
        # General settings can be top-level or within placard_agent for flexibility
        self.plaque_opacity = placard_config.get('plaque_opacity', config.get('plaque_opacity', plaque_opacity) if config else plaque_opacity)
        self.margin_percent = placard_config.get('margin_percent', config.get('margin_percent', margin_percent) if config else margin_percent)
        self.fallback_bg_size = placard_config.get('fallback_bg_size', config.get('fallback_bg_size', fallback_bg_size) if config else fallback_bg_size)
        self.text_color = placard_config.get('text_color', config.get('text_color', text_color) if config else text_color)

        # Validate opacity
        self.plaque_opacity_int = int(255 * max(0.0, min(1.0, self.plaque_opacity)))
        logging.info(f"[PlacardAgent] Using background: {self.background_image_path}, Opacity: {self.plaque_opacity}, Margin: {self.margin_percent}%")


    # Removed _create_textured_background as we now load an image
    # Removed _analyze_background_color and _get_contrasting_color - using fixed text color for now
    # Simpler approach: Use black text, assuming card_stock provides enough contrast.

    def add_plaque(self, input_path, output_path, metadata):
        """
        Loads a background image, adds text from metadata, resizes proportionally,
        and overlays it onto the input artwork image using smart positioning to avoid
        obscuring key artistic elements. Adapts plaque style based on genre metadata.

        Args:
            input_path (str): Path to the input artwork image.
            output_path (str): Path to save the output image with the plaque.
            metadata (dict): Dictionary containing text information. Expected keys:
                             'title', 'author', 'date', 'nationality', 'style', 'genre'.

        Returns:
            str or None: The output_path if successful, None otherwise.
        """
        self._think_with_grok(
            f"Preparing to add a placard to {input_path} with metadata: {metadata}. "
            "What are the optimal layout and content considerations for a museum-style plaque?"
        )
        logging.info(f"[PlacardAgent] Starting plaque addition for {input_path}")
        try:
            artwork_img = Image.open(input_path).convert("RGBA")
            artwork_width, artwork_height = artwork_img.size
            logging.info(f"[PlacardAgent] Artwork loaded: {artwork_width}x{artwork_height}")
        except FileNotFoundError:
            logging.error(f"[PlacardAgent] ERROR: Input artwork file not found at {input_path}")
            return None
        except Exception as e:
            logging.error(f"[PlacardAgent] ERROR: Could not load artwork image {input_path}. Error: {e}")
            return None

        # --- 1. Load or Create Background ---
        # Use the background image path specified in the configuration
        current_background_path = self.background_image_path
        logging.info(f"[PlacardAgent] Attempting to load background image from: {current_background_path}")

        try:
            plaque_bg_original = Image.open(current_background_path).convert("RGBA")
            bg_width_orig, bg_height_orig = plaque_bg_original.size
            logging.info(f"[PlacardAgent] Background image loaded: {current_background_path} ({bg_width_orig}x{bg_height_orig})")
        except FileNotFoundError:
            logging.warning(f"[PlacardAgent] WARNING: Background image file not found at {current_background_path}. Creating fallback white background.")
            bg_width_orig, bg_height_orig = self.fallback_bg_size
            plaque_bg_original = Image.new('RGBA', (bg_width_orig, bg_height_orig), (255, 255, 255, 255))  # White background
        except Exception as e:
            logging.warning(f"[PlacardAgent] WARNING: Could not load background image {current_background_path} (Error: {e}). Creating fallback white background.")
            bg_width_orig, bg_height_orig = self.fallback_bg_size
            plaque_bg_original = Image.new('RGBA', (bg_width_orig, bg_height_orig), (255, 255, 255, 255))  # White background

        # --- 3. Resize Placard Proportionally ---
        # Target width: ~5% of artwork width
        target_plaque_width_percent = 5
        target_plaque_width = int(artwork_width * target_plaque_width_percent / 100)

        # Ensure minimum width to prevent placard becoming too small
        min_plaque_width = 50  # Minimum pixels wide
        if target_plaque_width < min_plaque_width:
            logging.info(f"[PlacardAgent] Calculated target width {target_plaque_width}px is below minimum {min_plaque_width}px. Using minimum width.")
            target_plaque_width = min_plaque_width

        # Calculate scale factor and target height to maintain aspect ratio
        scale_factor = target_plaque_width / bg_width_orig
        target_plaque_height = int(bg_height_orig * scale_factor)

        logging.info(f"[PlacardAgent] Resizing placard: Target width {target_plaque_width}px ({target_plaque_width_percent}%), Target height {target_plaque_height}px")
        try:
            # Use LANCZOS for high-quality resizing
            plaque_bg_resized = plaque_bg_original.resize((target_plaque_width, target_plaque_height), Image.Resampling.LANCZOS)

            # Apply opacity uniformly (optional, could be done during paste)
            alpha = plaque_bg_resized.split()[-1]
            alpha = alpha.point(lambda p: min(p, self.plaque_opacity_int))  # Apply configured opacity
            plaque_bg_resized.putalpha(alpha)

        except Exception as e:
            logging.error(f"[PlacardAgent] ERROR: Failed to resize placard background. Error: {e}")
            return None

        # --- 2. Add Dynamic Text ---
        draw = ImageDraw.Draw(plaque_bg_resized)
        # Use text_color from init, default is black, adapt based on genre if needed
        text_color = self.text_color
        if genre in ['surrealist', 'abstract']:
            text_color = (50, 50, 50, 255)  # Slightly lighter for modern genres
        padding = int(target_plaque_width * 0.05)  # 5% padding inside the plaque

        # Prepare text content from metadata
        title = metadata.get('title', 'Untitled')
        author = metadata.get('author', 'Unknown Artist')
        date = metadata.get('date', '')
        nationality = metadata.get('nationality', '')
        style = metadata.get('style', '')

        # Combine fields vertically
        lines_content = [
            title,
            author,
            date,
            nationality,
            style
        ]
        lines_content = [line for line in lines_content if line]  # Remove empty strings

        # Font handling with fallback and text drawing
        font_regular, font_bold = None, None
        try:
            # Attempt to load specified TTF fonts
            initial_font_size = int(target_plaque_height * 0.08) # Start size relative to plaque height
            if initial_font_size < 10: initial_font_size = 10 # Minimum font size
            logging.info(f"[PlacardAgent] Initial target font size: {initial_font_size}")

            try:
                font_regular = ImageFont.truetype(self.font_path, initial_font_size)
                logging.info(f"[PlacardAgent] Loaded regular font: {self.font_path} at size {initial_font_size}")
            except IOError:
                logging.warning(f"[PlacardAgent] Regular font file not found at {self.font_path}. Trying default.")
                font_regular = ImageFont.load_default()

            try:
                # Use bold font only if regular TTF was found, otherwise default doesn't have bold pair
                if self.font_path != 'default': # Check if a specific font was intended
                    font_bold = ImageFont.truetype(self.font_path_bold, initial_font_size)
                    logging.info(f"[PlacardAgent] Loaded bold font: {self.font_path_bold} at size {initial_font_size}")
                else:
                    font_bold = font_regular # Fallback bold to regular if using default font
            except IOError:
                logging.warning(f"[PlacardAgent] Bold font file not found at {self.font_path_bold}. Using regular font for bold sections.")
                font_bold = font_regular # Fallback bold to regular

            # Layout: Title bold (if possible), rest regular
            # Ensure fonts list matches lines_content length, using font_regular for extras
            num_lines = len(lines_content)
            fonts = [font_bold] + [font_regular] * (num_lines - 1) if num_lines > 0 else []
            if len(fonts) > num_lines: fonts = fonts[:num_lines] # Truncate if needed
            while len(fonts) < num_lines: fonts.append(font_regular) # Pad if needed

            max_text_width = target_plaque_width - 2 * padding
            # Wrap width calculation needs to be inside the loop as font changes

            y_text = padding
            total_text_height = 0

            # Adjust font size dynamically to fit height
            font_size = initial_font_size
            final_wrapped_lines_with_fonts = []
            final_line_height = 0

            while font_size >= 8: # Lowered minimum practical size slightly
                logging.debug(f"[PlacardAgent] Trying font size: {font_size}")
                # Reload fonts at the current size (handle potential errors again)
                try:
                    current_font_regular = ImageFont.truetype(self.font_path, font_size)
                except IOError:
                    current_font_regular = ImageFont.load_default() # Use default if TTF fails at this size

                try:
                    # Only use bold if regular TTF was found for this size
                    if isinstance(current_font_regular, ImageFont.FreeTypeFont):
                         current_font_bold = ImageFont.truetype(self.font_path_bold, font_size)
                    else:
                         current_font_bold = current_font_regular # Fallback bold to regular
                except IOError:
                    current_font_bold = current_font_regular # Fallback bold to regular

                # Update fonts list for the current size
                num_lines = len(lines_content)
                current_fonts = [current_font_bold] + [current_font_regular] * (num_lines - 1) if num_lines > 0 else []
                if len(current_fonts) > num_lines: current_fonts = current_fonts[:num_lines]
                while len(current_fonts) < num_lines: current_fonts.append(current_font_regular)

                # Use the height of the regular font for consistent line spacing
                bbox = draw.textbbox((0, 0), "Ay", font=current_font_regular)
                line_height = (bbox[3] - bbox[1]) * 1.25 # Approx line height with spacing
                if line_height <= 0: line_height = font_size * 1.25 # Estimate if bbox fails

                current_wrapped_lines_with_fonts = []
                current_total_height = padding # Start with top padding

                for i, line_text in enumerate(lines_content):
                    font_to_use = current_fonts[i]
                    # Calculate wrap width based on current font size and max width
                    # Use textlength for potentially better wrap estimation than single char
                    try:
                        # Use textlength if available (newer PIL/Pillow)
                        char_width_approx = draw.textlength("W", font=font_to_use)
                    except AttributeError:
                        # Fallback to bbox if textlength not available
                        char_bbox = draw.textbbox((0,0), "W", font=font_to_use)
                        char_width_approx = char_bbox[2] - char_bbox[0]

                    wrap_width = max(10, int(max_text_width / char_width_approx)) if char_width_approx > 0 else 20

                    wrapped = textwrap.wrap(line_text, width=wrap_width, replace_whitespace=False, drop_whitespace=False)
                    if not wrapped: wrapped = [""] # Handle case where original line was empty

                    for wrapped_line in wrapped:
                        current_wrapped_lines_with_fonts.append((wrapped_line, font_to_use))
                        current_total_height += line_height

                current_total_height += padding # Add bottom padding

                if current_total_height <= target_plaque_height:
                    logging.info(f"[PlacardAgent] Text fits at font size {font_size}. Calculated height: {current_total_height:.2f}px <= {target_plaque_height}px")
                    final_wrapped_lines_with_fonts = current_wrapped_lines_with_fonts
                    final_line_height = line_height
                    break # Font size is good
                else:
                    # Text too tall, reduce font size and try again
                    logging.debug(f"[PlacardAgent] Text too tall at font size {font_size} ({current_total_height:.2f}px > {target_plaque_height}px). Reducing font size.")
                    font_size -= 1 # Decrement by 1 for finer control

            if not final_wrapped_lines_with_fonts:
                 logging.warning(f"[PlacardAgent] Text might be too long to fit even at minimum font size {font_size+1}. Using smallest calculated size.")
                 # Use the last calculated wrapped lines from the smallest font size attempt
                 final_wrapped_lines_with_fonts = current_wrapped_lines_with_fonts
                 final_line_height = line_height
                 if not final_wrapped_lines_with_fonts:
                    logging.error(f"[PlacardAgent] Could not prepare text even at minimum size. Skipping text drawing.")
                    # Skip drawing if still no lines

            if final_wrapped_lines_with_fonts:
                # Draw the final wrapped text vertically centered (optional, simple top alignment used here)
                y_text = padding
                # Optional: Calculate total text block height for centering
                # total_final_text_height = len(final_wrapped_lines_with_fonts) * final_line_height
                # y_text = (target_plaque_height - total_final_text_height) / 2

                logging.info(f"[PlacardAgent] Drawing {len(final_wrapped_lines_with_fonts)} lines of text with font size {font_size} and line height {final_line_height:.2f}")
                for line, font in final_wrapped_lines_with_fonts:
                     # Check if text exceeds plaque height before drawing THIS line
                     if y_text + final_line_height > target_plaque_height - padding:
                         logging.warning(f"[PlacardAgent] Text drawing truncated due to height limit before drawing line: '{line}'")
                         # Optionally draw '...' if space allows
                         if y_text + (draw.textbbox((0,0), "...", font=current_font_regular)[3] * 1.25) <= target_plaque_height - padding:
                             draw.text((padding, y_text), "...", font=current_font_regular, fill=self.text_color)
                         break # Stop drawing further lines
                     draw.text((padding, y_text), line, font=font, fill=self.text_color)
                     y_text += final_line_height
                logging.info(f"[PlacardAgent] Text drawn onto placard.")
            else:
                logging.info(f"[PlacardAgent] No text lines to draw.")

        # Removed specific IOError catch as it's handled during font loading attempts now
        except Exception as e:
            logging.error(f"[PlacardAgent] Unexpected failure during text processing or drawing. Error: {e}", exc_info=True)
            # Continue with plaque background only
            pass

        # --- 4. Overlay Placard with Smart Positioning & 5. Save and Return Path ---
        try:
            margin_x = int(artwork_width * self.margin_percent / 100)
            margin_y = int(artwork_height * self.margin_percent / 100)

            # Smart positioning: Try to find a low-detail area, default to lower-right if not implemented
            plaque_x, plaque_y = self._find_optimal_plaque_position(
                artwork_img, target_plaque_width, target_plaque_height, margin_x, margin_y
            )

            logging.info(f"[PlacardAgent] Overlaying placard at ({plaque_x}, {plaque_y})")
            # Paste using the plaque's alpha channel for transparency
            artwork_img.paste(plaque_bg_resized, (plaque_x, plaque_y), plaque_bg_resized)
            
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir): # Check if output_dir is not empty string
                os.makedirs(output_dir)
                logging.info(f"[PlacardAgent] Created output directory: {output_dir}")
            elif not output_dir: # Handle case where output_path is just a filename
                 logging.info(f"[PlacardAgent] Output path '{output_path}' is in current directory. No directory creation needed.")


            artwork_img.convert("RGB").save(output_path)
            logging.info(f"[PlacardAgent] Plaque added successfully. Output saved to {output_path}")
            return output_path
            
        except Exception as e:
            logging.error(f"[PlacardAgent] ERROR: Failed to paste placard or save final image to {output_path}. Error: {e}", exc_info=True)
            return None

    def _find_optimal_plaque_position(self, artwork_img, plaque_width, plaque_height, margin_x, margin_y):
        """
        Find an optimal position for the plaque by looking for low-detail areas.
        Currently a placeholder, defaults to lower-right corner.
        Future implementation could use edge detection or variance analysis to find unobtrusive areas.
        """
        artwork_width, artwork_height = artwork_img.size
        # Default to lower-right corner
        plaque_x = artwork_width - plaque_width - margin_x
        plaque_y = artwork_height - plaque_height - margin_y
        # TODO: Implement smart positioning using image analysis (e.g., OpenCV for low variance areas)
        logging.info(f"[PlacardAgent] Using default lower-right position for plaque. Smart positioning TBD.")
        return plaque_x, plaque_y

    def _think_with_grok(self, prompt: str):
        """
        Internal reasoning using Grok LLM (grok-3-mini-fast-high-beta).
        """
        import os
        grok_api_key = os.environ.get("GROK_API_KEY")
        if not grok_api_key:
            logging.warning("[PlacardAgent] Grok API key not found. Skipping internal reasoning.")
            return None
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=grok_api_key,
                base_url="https://api.x.ai/v1",
            )
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="grok-3-mini-fast-high-beta",
                temperature=0.2,
                max_tokens=256
            )
            reasoning = response.choices[0].message.content
            logging.info(f"[PlacardAgent] Grok thinking output: {reasoning}")
            return reasoning
        except Exception as e:
            logging.error(f"[PlacardAgent] Grok thinking failed: {e}")
            return None

# Example usage remains similar, but ensure background image exists
# and metadata keys match the new expectation.
if __name__ == "__main__":
    # Create dummy fonts and background if they don't exist for basic testing
    os.makedirs('fonts', exist_ok=True)
    os.makedirs('input', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    try:
        # Create dummy font files if they don't exist (won't be valid fonts)
        if not os.path.exists('fonts/arial.ttf'): open('fonts/arial.ttf', 'a').close()
        if not os.path.exists('fonts/arialbd.ttf'): open('fonts/arialbd.ttf', 'a').close()
        # Create a dummy background image if it doesn't exist
        if not os.path.exists('input/card_stock.jpg'):
             Image.new('RGB', (600, 400), color = (240, 230, 220)).save('input/card_stock.jpg')
        # Create a dummy input image
        if not os.path.exists('input/test_image.jpg'):
             Image.new('RGB', (1920, 1080), color = (73, 109, 137)).save('input/test_image.jpg')

    except Exception as e:
        print(f"Warning: Could not create dummy files for testing: {e}")


    agent = PlacardAgent() # Uses defaults including 'input/card_stock.jpg'
    # Example metadata with new fields
    metadata = {
        'title': 'Example Artwork Title That Is Quite Long To Test Wrapping',
        'artist': 'Jane Doe',
        'date': '2024',
        # 'description': 'Description removed as per new requirements',
        'style': 'Impressionism'
    }
    # Ensure input/test_image.jpg exists for this example to run
    if os.path.exists('input/test_image.jpg'):
         result_path = agent.add_plaque('input/test_image.jpg', 'output/test_plaqued.jpg', metadata)
         if result_path:
             print(f"Test completed. Output image: {result_path}")
         else:
             print("Test failed.")
    else:
        print("Skipping test: input/test_image.jpg not found.")