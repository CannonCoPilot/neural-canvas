import unittest
import os
import logging
import json
import numpy as np
from PIL import Image
# Import specialized VisionAgent classes
from art_agent_team.agents.vision_agent_abstract import VisionAgentAbstract
from art_agent_team.agents.vision_agent_animal import VisionAgentAnimal
from art_agent_team.agents.vision_agent_figurative import VisionAgentFigurative
from art_agent_team.agents.vision_agent_genre import VisionAgentGenre
from art_agent_team.agents.vision_agent_landscape import VisionAgentLandscape
from art_agent_team.agents.vision_agent_portrait import VisionAgentPortrait
from art_agent_team.agents.vision_agent_religious_historical import VisionAgentReligiousHistorical
from art_agent_team.agents.vision_agent_still_life import VisionAgentStillLife
from concurrent.futures import wait


class TestVisionWorkflowIntegration(unittest.TestCase):
    """Integration tests for the complete vision workflow."""

    def setUp(self):
        """Set up test environment with real paths and credentials, loading API keys only from config.yaml."""
        import yaml

        # Project base directory
        self.base_dir = '/Users/nathaniel.cannon/Documents/VScodeWork/Art_AI'

        # Use actual production paths
        self.input_folder = os.path.join(self.base_dir, 'test_images')
        self.output_folder = os.path.join(self.base_dir, 'art_agent_team/output')
        # Define multiple test images for broader coverage
        self.test_images = [
            'Frank Brangwyn, Swans, c.1921.jpg',
            'Emile Clause - Summer morning 1891.jpeg',
            'Santa Ynez California Hillside, Eyvind Earle, 1969.jpeg',
            'Tomioka Soichiro - Trees (1961).jpeg'
        ]

        # Load API keys from config.yaml only
        config_path = os.path.join(self.base_dir, 'art_agent_team/config/config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        google_api_key = config.get('google_api_key')
        grok_api_key = config.get('grok_api_key')

        if not google_api_key:
            raise ValueError("google_api_key not set in config.yaml")
        if not grok_api_key:
            raise ValueError("grok_api_key not set in config.yaml")

        # Test configuration with both API keys
        self.test_config = {
            'input_folder': self.input_folder,
            'output_folder': self.output_folder,
            'google_api_key': google_api_key,
            'grok_api_key': grok_api_key
        }

        # Create directories if needed
        os.makedirs(self.output_folder, exist_ok=True)

    def test_api_key_initialization(self):
        """Test that API keys are properly passed to and initialized in the agent."""
        vision_agent = VisionAgentAnimal(self.test_config)
        
        # Verify Gemini client initialization
        self.assertIsNotNone(vision_agent.gemini_pro, "Gemini Pro model not initialized")
        
        # Verify Grok API key is present and model name is set
        self.assertIsNotNone(getattr(vision_agent, "grok_api_key", None), "Grok API key not set in VisionAgentAnimal")
        self.assertEqual(vision_agent.grok_vision_model, "grok-2-vision-latest", "Incorrect Grok model specified")

    def run_vision_workflow_test(self, agent_class, agent_name, image_path, research_data):
        """Helper method to run vision workflow test for a given agent and image."""
        # Initialize the specialized vision agent
        vision_agent = agent_class(self.test_config)

        # Analyze image using the specialized agent
        analysis_results = vision_agent.analyze_image(image_path, research_data)

        # Verify analysis results
        self.assertIsNotNone(analysis_results, f"Analysis failed for {agent_name} with image {os.path.basename(image_path)}")
        self.assertIn("objects", analysis_results, f"No objects detected for {agent_name}")
        self.assertIn("segmentation_masks", analysis_results, f"No segmentation masks created for {agent_name}")

        # Verify objects have importance scores
        for obj in analysis_results["objects"]:
            self.assertIn("importance", obj, f"Object missing importance score in {agent_name}")
            self.assertGreaterEqual(obj["importance"], 0.0)
            self.assertLessEqual(obj["importance"], 1.0)

        # Save all versions
        basename = os.path.splitext(os.path.basename(image_path))[0]
        labeled_path = os.path.join(self.output_folder, f"{basename}_labeled.jpg")
        masked_path = os.path.join(self.output_folder, f"{basename}_masked.jpg")
        cropped_path = os.path.join(self.output_folder, f"{basename}_cropped.jpg")

        vision_agent.save_analysis_outputs(image_path, analysis_results, self.output_folder)

        # Verify all output files exist
        self.assertTrue(os.path.exists(labeled_path), f"Labeled image not created for {agent_name} with {basename}")
        self.assertTrue(os.path.exists(masked_path), f"Masked image not created for {agent_name} with {basename}")
        self.assertTrue(os.path.exists(cropped_path), f"Cropped image not created for {agent_name} with {basename}")

        # Verify cropped image aspect ratio (assuming 16:9 is the target)
        with Image.open(cropped_path) as img:
            width, height = img.size
            actual_ratio = width / height
            self.assertAlmostEqual(actual_ratio, 16/9, places=2, msg=f"Cropped image aspect ratio is not 16:9 for {agent_name} with {basename}")

        # Verify important objects have masks (threshold based on percentile)
        important_objects_for_masking = [obj for obj in analysis_results["objects"]
                                         if obj.get("importance", 0) >= np.percentile([o.get("importance", 0)
                                         for o in analysis_results["objects"]], 80)]
        self.assertLessEqual(len(analysis_results["segmentation_masks"]), len(important_objects_for_masking),
                           f"Too many segmentation masks created for {agent_name} with {basename}")

        return analysis_results

    def test_vision_workflow_animal(self):
        """Test the animal vision workflow with output verification for multiple images."""
        agent_name = "VisionAgentAnimal"
        for image in self.test_images:
            image_path = os.path.join(self.input_folder, image)
            # Define research data specific to the test image
            research_data = {
                "primary_subject": "animals",
                "secondary_subjects": ["water", "trees"],
                "paragraph_description": f"A painting of animals in a natural setting with {image}.",
                "structured_sentence": f"The painting depicts animals, possibly near water or trees, in {image}."
            }
            analysis_results = self.run_vision_workflow_test(VisionAgentAnimal, agent_name, image_path, research_data)
            
            # Verify that "animal" objects are detected and have high importance
            animal_objects = [obj for obj in analysis_results["objects"]
                              if obj.get("type", "").lower() in ["animal", "group_of_animals"] or "animal" in obj.get("label", "").lower()]
            self.assertGreater(len(animal_objects), 0, f"No animal objects detected in {image} by {agent_name}")
            self.assertTrue(any(obj.get("importance", 0) > 0.7 for obj in animal_objects), f"No important animal objects detected in {image} by {agent_name}")

    def test_vision_workflow_landscape(self):
        """Test the landscape vision workflow with output verification for multiple images."""
        agent_name = "VisionAgentLandscape"
        for image in self.test_images:
            image_path = os.path.join(self.input_folder, image)
            # Define research data specific to the test image
            research_data = {
                "primary_subject": "landscape",
                "secondary_subjects": ["mountains", "rivers"],
                "paragraph_description": f"A landscape painting featuring natural scenery in {image}.",
                "structured_sentence": f"The painting depicts a landscape with possible mountains or rivers in {image}."
            }
            analysis_results = self.run_vision_workflow_test(VisionAgentLandscape, agent_name, image_path, research_data)
            
            # Verify that "landscape" or "terrain" objects are detected and have high importance
            landscape_objects = [obj for obj in analysis_results["objects"]
                                 if obj.get("type", "").lower() in ["landscape", "terrain"] or "landscape" in obj.get("label", "").lower()]
            self.assertGreater(len(landscape_objects), 0, f"No landscape objects detected in {image} by {agent_name}")
            self.assertTrue(any(obj.get("importance", 0) > 0.7 for obj in landscape_objects), f"No important landscape objects detected in {image} by {agent_name}")

    def test_vision_workflow_portrait(self):
        """Test the portrait vision workflow with output verification for multiple images."""
        agent_name = "VisionAgentPortrait"
        for image in self.test_images:
            image_path = os.path.join(self.input_folder, image)
            # Define research data specific to the test image
            research_data = {
                "primary_subject": "portrait",
                "secondary_subjects": ["person", "face"],
                "paragraph_description": f"A portrait painting of a person in {image}.",
                "structured_sentence": f"The painting depicts a portrait of a person, focusing on their face in {image}."
            }
            analysis_results = self.run_vision_workflow_test(VisionAgentPortrait, agent_name, image_path, research_data)
            
            # Verify that "portrait" or "person" objects are detected and have high importance
            portrait_objects = [obj for obj in analysis_results["objects"]
                                if obj.get("type", "").lower() in ["portrait", "person", "face"] or "portrait" in obj.get("label", "").lower()]
            self.assertGreater(len(portrait_objects), 0, f"No portrait objects detected in {image} by {agent_name}")
            self.assertTrue(any(obj.get("importance", 0) > 0.7 for obj in portrait_objects), f"No important portrait objects detected in {image} by {agent_name}")

    def test_vision_workflow_religious_historical(self):
        """Test the religious/historical vision workflow with output verification for multiple images."""
        agent_name = "VisionAgentReligiousHistorical"
        for image in self.test_images:
            image_path = os.path.join(self.input_folder, image)
            # Define research data specific to the test image
            research_data = {
                "primary_subject": "religious historical",
                "secondary_subjects": ["figures", "symbols"],
                "paragraph_description": f"A religious or historical painting with significant figures in {image}.",
                "structured_sentence": f"The painting depicts a religious or historical scene with figures and symbols in {image}."
            }
            analysis_results = self.run_vision_workflow_test(VisionAgentReligiousHistorical, agent_name, image_path, research_data)
            
            # Verify that "religious" or "historical" objects are detected and have high importance
            religious_objects = [obj for obj in analysis_results["objects"]
                                 if obj.get("type", "").lower() in ["religious", "historical", "figure"] or "religious" in obj.get("label", "").lower() or "historical" in obj.get("label", "").lower()]
            self.assertGreater(len(religious_objects), 0, f"No religious/historical objects detected in {image} by {agent_name}")
            self.assertTrue(any(obj.get("importance", 0) > 0.7 for obj in religious_objects), f"No important religious/historical objects detected in {image} by {agent_name}")

    def test_vision_workflow_still_life(self):
        """Test the still life vision workflow with output verification for multiple images."""
        agent_name = "VisionAgentStillLife"
        for image in self.test_images:
            image_path = os.path.join(self.input_folder, image)
            # Define research data specific to the test image
            research_data = {
                "primary_subject": "still life",
                "secondary_subjects": ["objects", "food"],
                "paragraph_description": f"A still life painting featuring objects in {image}.",
                "structured_sentence": f"The painting depicts a still life arrangement of objects, possibly food, in {image}."
            }
            analysis_results = self.run_vision_workflow_test(VisionAgentStillLife, agent_name, image_path, research_data)
            
            # Verify that "still life" or "object" objects are detected and have high importance
            still_life_objects = [obj for obj in analysis_results["objects"]
                                  if obj.get("type", "").lower() in ["still_life", "object"] or "still life" in obj.get("label", "").lower()]
            self.assertGreater(len(still_life_objects), 0, f"No still life objects detected in {image} by {agent_name}")
            self.assertTrue(any(obj.get("importance", 0) > 0.7 for obj in still_life_objects), f"No important still life objects detected in {image} by {agent_name}")

if __name__ == '__main__':
    unittest.main()

import threading
import queue
from queue import Queue, Empty
from unittest.mock import MagicMock, patch
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from art_agent_team.docent_agent import DocentAgent
from art_agent_team.agents.research_agent import ResearchAgent
from art_agent_team.agents.upscale_agent import UpscaleAgent
from art_agent_team.agents.placard_agent import PlacardAgent
from art_agent_team.agents.vision_agent_animal import VisionAgentAnimal

class WorkflowError(Exception):
    """Custom exception for workflow errors with agent context."""
    def __init__(self, message, agent_name, queue_state=None):
        self.agent_name = agent_name
        self.queue_state = queue_state
        super().__init__(f"{agent_name}: {message}")

def threaded_agent_handoff_workflow(config, timeout=10):
    """
    Simulate threaded handoff with robust error handling and data verification.
    
    Args:
        config: Configuration dictionary with required settings
        timeout: Maximum time in seconds to wait for workflow completion
    
    Returns:
        dict: Workflow results including processing status and output files
    
    Raises:
        WorkflowError: If any agent fails or timeout occurs
    """
    # Setup queues with size limits to prevent memory issues
    queues = {
        'research': Queue(maxsize=10),
        'vision': Queue(maxsize=10),
        'upscale': Queue(maxsize=10),
        'placard': Queue(maxsize=10)
    }
    results = {'status': 'pending', 'output': [], 'errors': [], 'queue_sizes': {}}
    error_event = threading.Event()
    
    def handle_error(agent_name, error, queue_state=None):
        """Centralized error handling."""
        results['errors'].append({
            'agent': agent_name,
            'error': str(error),
            'queue_state': queue_state
        })
        error_event.set()
        
    def verify_output(agent_name, data):
        """Verify data integrity between agent handoffs."""
        if not data:
            raise WorkflowError("Empty output received", agent_name)
        results['output'].append(f"{agent_name} output verified")

    def docent_worker():
        """Initialize workflow with test image and metadata."""
        try:
            # Use a real image from test_images/ to avoid file not found error
            test_image_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'test_images',
                'Frank Brangwyn, Swans, c.1921.jpg'
            )
            metadata = {'title': 'Test Image', 'artist': 'Test Artist'}
            queues['research'].put((test_image_path, metadata), timeout=2)
            results['output'].append("DocentAgent queued")
            verify_output('DocentAgent', metadata)
        except Exception as e:
            handle_error('DocentAgent', e)

    def research_worker():
        """Process research data and hand off to vision analysis."""
        try:
            while not error_event.is_set():
                try:
                    image_path, metadata = queues['research'].get(timeout=2)
                    if not all(k in metadata for k in ['title', 'artist']):
                        raise WorkflowError("Incomplete metadata", 'ResearchAgent')
                    
                    vision_agent = VisionAgentAnimal(config)
                    analysis = vision_agent.analyze_image(image_path, metadata)
                    verify_output('ResearchAgent', analysis)
                    
                    queues['vision'].put((image_path, analysis), timeout=2)
                    queues['research'].task_done()
                    monitor_queues()
                except Empty:
                    if not error_event.is_set():
                        results['output'].append("ResearchAgent completed")
                    break
        except Exception as e:
            handle_error('ResearchAgent', e, queues['research'].qsize())

    def vision_worker():
        """Perform vision analysis and prepare for upscaling."""
        try:
            while not error_event.is_set():
                try:
                    image_path, analysis = queues['vision'].get(timeout=2)
                    verify_output('VisionAgent', analysis)
                    
                    # Process with real VisionAgent
                    cropped_path = os.path.join(config.get('output_dir', '.'), 'test_cropped.jpg')
                    vision_agent = VisionAgentAnimal(config)
                    analysis = vision_agent.analyze_image(image_path, analysis)
                    
                    queues['upscale'].put((cropped_path, analysis), timeout=2)
                    queues['vision'].task_done()
                    monitor_queues()
                except Empty:
                    if not error_event.is_set():
                        results['output'].append("VisionAgent completed")
                    break
        except Exception as e:
            handle_error('VisionAgent', e, queues['vision'].qsize())

    def upscale_worker():
        """Perform image upscaling with quality validation."""
        try:
            while not error_event.is_set():
                try:
                    image_path, analysis = queues['upscale'].get(timeout=2)
                    
                    # Use real UpscaleAgent implementation
                    upscale_agent = UpscaleAgent()
                    upscaled_path = os.path.join(config.get('output_dir', '.'), 'test_upscaled.jpg')
                    upscale_agent.upscale_image(image_path, upscaled_path)
                    
                    # Verify upscaled output
                    if not os.path.exists(upscaled_path):
                        raise WorkflowError("Upscaled image not created", 'UpscaleAgent')
                    
                    with Image.open(upscaled_path) as img:
                        if img.size != (3840, 2160):
                            raise WorkflowError("Invalid upscale resolution", 'UpscaleAgent')
                    
                    verify_output('UpscaleAgent', upscaled_path)
                    queues['placard'].put((upscaled_path, analysis), timeout=2)
                    queues['upscale'].task_done()
                    monitor_queues()
                except Empty:
                    if not error_event.is_set():
                        results['output'].append("UpscaleAgent completed")
                    break
        except Exception as e:
            handle_error('UpscaleAgent', e, queues['upscale'].qsize())

    def placard_worker():
        """Add placard to processed image with metadata."""
        try:
            while not error_event.is_set():
                try:
                    image_path, analysis = queues['placard'].get(timeout=2)
                    
                    # Use real PlacardAgent implementation
                    placard_agent = PlacardAgent()
                    output_path = os.path.join(config.get('output_dir', '.'), 'test_final.jpg')
                    metadata = {
                        'title': 'Test Image',
                        'artist': 'Test Artist',
                        'nationality': 'Test',
                        'date': '2025'
                    }
                    
                    placard_agent.add_plaque(image_path, output_path, metadata)
                    
                    if not os.path.exists(output_path):
                        raise WorkflowError("Final output not generated", 'PlacardAgent')
                    
                    verify_output('PlacardAgent', output_path)
                    results['output'].append("PlacardAgent completed")
                    queues['placard'].task_done()
                    monitor_queues()
                except Empty:
                    if not error_event.is_set():
                        results['output'].append("Workflow completed successfully")
                    break
        except Exception as e:
            handle_error('PlacardAgent', e, queues['placard'].qsize())

    def monitor_queues():
        """Monitor current queue sizes for debugging."""
        for name, q in queues.items():
            results['queue_sizes'][name] = q.qsize()

    # Execute workflow with timeout and proper cleanup
    with ThreadPoolExecutor(max_workers=5) as executor:
        try:
            # Start all workers
            futures = []
            for worker in [docent_worker, research_worker, vision_worker,
                         upscale_worker, placard_worker]:
                futures.append(executor.submit(worker))
            
            # Wait for completion or timeout
            done, pending = wait(futures, timeout=timeout)
            
            if pending:
                error_event.set()
                raise WorkflowError("Workflow timeout", "WorkflowManager")
            
            # Check for exceptions
            for future in done:
                if future.exception():
                    raise future.exception()
            
            # Verify all queues are empty
            for name, q in queues.items():
                if not q.empty():
                    raise WorkflowError(f"Queue '{name}' not fully processed", "WorkflowManager")
            
            results['status'] = 'completed'
            
        except Exception as e:
            results['status'] = 'failed'
            handle_error('WorkflowManager', e)
            raise
        finally:
            # Cleanup
            error_event.set()
            for q in queues.values():
                with q.mutex:
                    q.queue.clear()
    
    return results

class TestAgentThreadedIntegration(unittest.TestCase):
    """Test suite for threaded agent integration workflow."""
    
    def setUp(self):
        """Initialize test environment with required paths and configuration."""
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.test_data_dir = os.path.join(self.test_dir, '..', 'test_data')
        self.output_dir = os.path.join(self.test_data_dir, 'output')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.config = {
            'test_data_dir': self.test_data_dir,
            'output_dir': self.output_dir
        }

    def test_successful_workflow(self):
        """Test complete workflow execution with all agents."""
        results = threaded_agent_handoff_workflow(self.config)
        self.assertEqual(results['status'], 'completed')
        self.assertIn("PlacardAgent completed", results['output'])
        self.assertEqual(len(results['errors']), 0)

    def test_timeout_handling(self):
        """Test workflow timeout handling."""
        with self.assertRaises(WorkflowError) as context:
            results = threaded_agent_handoff_workflow(self.config, timeout=1)
        self.assertIn("timeout", str(context.exception))

    def test_error_propagation(self):
        """Test error handling and propagation through the workflow."""
        with patch('PIL.Image.open') as mock_open:
            mock_open.side_effect = IOError("Test error")
            with self.assertRaises(WorkflowError) as context:
                results = threaded_agent_handoff_workflow(self.config)
            self.assertIn("error", results['status'])
            self.assertGreater(len(results['errors']), 0)

    def test_queue_limits(self):
        """Test queue size limits and backpressure handling."""
        # Modify config to trigger queue limits
        config = self.config.copy()
        config['stress_test'] = True
        results = threaded_agent_handoff_workflow(config)
        self.assertLessEqual(max(q.qsize() for q in results.get('queue_sizes', {}).values()), 10)

    def test_minimal_agent_implementations(self):
        """Test integration of minimal UpscaleAgent and PlacardAgent implementations."""
        results = threaded_agent_handoff_workflow(self.config)
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'test_upscaled.jpg')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'test_final.jpg')))

    def tearDown(self):
        """Cleanup test outputs."""
        for f in os.listdir(self.output_dir):
            if f.startswith('test_'):
                os.remove(os.path.join(self.output_dir, f))
