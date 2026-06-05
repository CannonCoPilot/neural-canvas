# art_agent_team/agents/__init__.py

from .vision_agent_default import DefaultVisionAgent # Corrected class name
from .vision_agent_landscape import VisionAgentLandscape
from .vision_agent_animal import VisionAgentAnimal
from .vision_agent_portrait import VisionAgentPortrait
from .vision_agent_still_life import VisionAgentStillLife
from .vision_agent_figurative import VisionAgentFigurative
from .vision_agent_genre import VisionAgentGenre # Corrected class name
from .vision_agent_religious_historical import VisionAgentReligiousHistorical
from .vision_agent_surrealist import VisionAgentSurrealist
# Import other vision agents as needed

vision_agent_classes = {
    "Default": DefaultVisionAgent, # Corrected class name
    "Landscape": VisionAgentLandscape,
    "Animal": VisionAgentAnimal,
    "Portrait": VisionAgentPortrait,
    "Still Life": VisionAgentStillLife,
    "Figurative": VisionAgentFigurative,
    "Genre Scene": VisionAgentGenre, # Corrected class name
    "Religious/Historical": VisionAgentReligiousHistorical,
    "Surrealist": VisionAgentSurrealist,
    # Add other mappings here, e.g.
    # "Abstract": VisionAgentAbstract, # If you have VisionAgentAbstract
}

# Make other agents available for import if desired
# DocentAgent should be imported from art_agent_team.docent_agent, not here.
from .research_agent import ResearchAgent
from .upscale_agent import UpscaleAgent
from .placard_agent import PlacardAgent

__all__ = [
    "ResearchAgent",
    "UpscaleAgent",
    "PlacardAgent",
    "DefaultVisionAgent", # Corrected class name
    "VisionAgentLandscape",
    "VisionAgentAnimal",
    "VisionAgentPortrait",
    "VisionAgentStillLife",
    "VisionAgentFigurative",
    "VisionAgentGenre", # Corrected class name
    "VisionAgentReligiousHistorical",
    "VisionAgentSurrealist",
    "vision_agent_classes"
]