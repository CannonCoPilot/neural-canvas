import os
import cv2
import numpy as np
import logging
from typing import Dict, Any
from PIL import Image
from skimage.metrics import structural_similarity as ssim

class FidelityMetrics:
    """
    A class to compute artistic fidelity metrics for processed artwork images,
    ensuring modifications respect composition, color balance, detail preservation,
    and emotional impact.
    """
    def __init__(self, config=None):
        """
        Initialize the FidelityMetrics with configuration parameters for thresholds.
        """
        if not config:
            config = {}
        self.composition_threshold = config.get('composition_threshold', 0.8)
        self.color_threshold = config.get('color_threshold', 0.85)
        self.detail_threshold = config.get('detail_threshold', 0.75)
        self.emotional_contrast_threshold = config.get('emotional_contrast_threshold', 0.7)
        logging.info("[FidelityMetrics] Initialized with thresholds: Composition={:.2f}, Color={:.2f}, Detail={:.2f}, Emotional Contrast={:.2f}".format(
            self.composition_threshold, self.color_threshold, self.detail_threshold, self.emotional_contrast_threshold))

    def validate_fidelity(self, original_path: str, processed_path: str, genre: str = "Default") -> Dict[str, Any]:
        """
        Compute fidelity metrics between original and processed images.
        Returns a dictionary of metric scores and pass/fail status based on thresholds.
        Adjusts thresholds slightly based on genre if necessary.

        Args:
            original_path (str): Path to the original image.
            processed_path (str): Path to the processed image.
            genre (str): Genre of the artwork to adjust thresholds if needed.

        Returns:
            Dict[str, Any]: Dictionary containing metric scores and pass/fail status.
        """
        if not os.path.exists(original_path) or not os.path.exists(processed_path):
            logging.error(f"[FidelityMetrics] Image not found. Original: {original_path}, Processed: {processed_path}")
            return {"error": "Image file not found", "passed": False}

        try:
            # Load images
            orig_img = cv2.imread(original_path, cv2.IMREAD_COLOR)
            proc_img = cv2.imread(processed_path, cv2.IMREAD_COLOR)

            if orig_img is None or proc_img is None:
                logging.error(f"[FidelityMetrics] Failed to load images. Original: {original_path}, Processed: {processed_path}")
                return {"error": "Failed to load images", "passed": False}

            # Resize processed image to match original dimensions if necessary for comparison
            if orig_img.shape != proc_img.shape:
                proc_img = cv2.resize(proc_img, (orig_img.shape[1], orig_img.shape[0]), interpolation=cv2.INTER_LANCZOS4)

            # Adjust thresholds based on genre (e.g., surrealist art may tolerate more deviation)
            adj_composition_threshold = self.composition_threshold
            adj_color_threshold = self.color_threshold
            if genre.lower() in ['surrealist', 'abstract']:
                adj_composition_threshold -= 0.1  # More tolerance for composition changes
                adj_color_threshold -= 0.05       # Slight tolerance for color changes

            # Compute metrics
            metrics = {
                "composition_balance": self._compute_composition_balance(orig_img, proc_img),
                "color_fidelity": self._compute_color_fidelity(orig_img, proc_img),
                "detail_preservation": self._compute_detail_preservation(orig_img, proc_img),
                "emotional_impact": self._compute_emotional_impact(orig_img, proc_img, genre)
            }

            # Determine pass/fail based on thresholds
            metrics["passed"] = (
                metrics["composition_balance"]["score"] >= adj_composition_threshold and
                metrics["color_fidelity"]["score"] >= adj_color_threshold and
                metrics["detail_preservation"]["score"] >= self.detail_threshold and
                metrics["emotional_impact"]["score"] >= self.emotional_contrast_threshold
            )

            metrics["thresholds"] = {
                "composition": adj_composition_threshold,
                "color": adj_color_threshold,
                "detail": self.detail_threshold,
                "emotional": self.emotional_contrast_threshold
            }

            logging.info(f"[FidelityMetrics] Validation complete for {os.path.basename(processed_path)}. Passed: {metrics['passed']}")
            return metrics

        except Exception as e:
            logging.error(f"[FidelityMetrics] Error during fidelity validation: {e}")
            return {"error": str(e), "passed": False}

    def _compute_composition_balance(self, orig_img: np.ndarray, proc_img: np.ndarray) -> Dict[str, Any]:
        """
        Measure composition balance by comparing structural similarity.
        High SSIM indicates preserved composition.
        """
        try:
            score, _ = ssim(orig_img, proc_img, multichannel=True, full=True)
            return {"score": float(score), "description": "Structural similarity index for composition balance"}
        except Exception as e:
            logging.error(f"[FidelityMetrics] Error computing composition balance: {e}")
            return {"score": 0.0, "description": "Error in computation", "error": str(e)}

    def _compute_color_fidelity(self, orig_img: np.ndarray, proc_img: np.ndarray) -> Dict[str, Any]:
        """
        Assess color fidelity by comparing color histograms.
        Uses histogram correlation for similarity measure.
        """
        try:
            orig_hist = cv2.calcHist([orig_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            proc_hist = cv2.calcHist([proc_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            score = cv2.compareHist(orig_hist, proc_hist, cv2.HISTCMP_CORREL)
            return {"score": float(score), "description": "Histogram correlation for color fidelity"}
        except Exception as e:
            logging.error(f"[FidelityMetrics] Error computing color fidelity: {e}")
            return {"score": 0.0, "description": "Error in computation", "error": str(e)}

    def _compute_detail_preservation(self, orig_img: np.ndarray, proc_img: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate detail preservation by comparing high-frequency components (edges).
        Uses Laplacian variance as a proxy for sharpness.
        """
        try:
            orig_lap = cv2.Laplacian(cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
            proc_lap = cv2.Laplacian(cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
            orig_var = orig_lap.var()
            proc_var = proc_lap.var()
            score = min(proc_var / orig_var if orig_var > 0 else 1.0, 1.0)  # Cap at 1.0 if processed has more detail
            return {"score": float(score), "description": "Laplacian variance ratio for detail preservation"}
        except Exception as e:
            logging.error(f"[FidelityMetrics] Error computing detail preservation: {e}")
            return {"score": 0.0, "description": "Error in computation", "error": str(e)}

    def _compute_emotional_impact(self, orig_img: np.ndarray, proc_img: np.ndarray, genre: str) -> Dict[str, Any]:
        """
        Use proxy metrics like contrast and brightness to estimate emotional impact retention.
        Adjusts expected contrast based on genre (e.g., high contrast for dramatic genres).
        """
        try:
            orig_gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
            proc_gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
            orig_contrast = orig_gray.std()
            proc_contrast = proc_gray.std()
            score = min(proc_contrast / orig_contrast if orig_contrast > 0 else 1.0, 1.0)
            # Adjust expectation based on genre
            if genre.lower() in ['religious_historical', 'portrait']:
                score = score * 0.9  # Slightly less strict for genres where contrast might be intentionally adjusted
            return {"score": float(score), "description": "Contrast ratio for emotional impact"}
        except Exception as e:
            logging.error(f"[FidelityMetrics] Error computing emotional impact: {e}")
            return {"score": 0.0, "description": "Error in computation", "error": str(e)}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    metrics = FidelityMetrics()
    # Placeholder for test usage
    # result = metrics.validate_fidelity('input/test_image.jpg', 'output/test_processed.jpg', 'Portrait')
    # print(result)