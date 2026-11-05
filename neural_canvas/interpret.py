"""Stage 2 — interpret the scene: genre, subjects and faces (boxed), intent, and the label's prose."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from PIL import Image

from .vlm import VLM

MAX_FACE_AREA = 0.35   # fraction of the image; a larger "face" box is a whole group
GENRES = ["landscape", "seascape", "cityscape", "portrait", "figure-group", "religious-historical",
          "genre-scene", "still-life", "animal", "interior", "abstract"]

PROMPT = """You are a museum curator preparing this artwork for display.
Verified identification (from the collection record — treat as fact): {facts}

1. Classify its genre as one of: {genres}.
2. Locate its subjects. For each give a TIGHT bounding box as [x0, y0, x1, y1] in coordinates
   normalised to 0-1000 over the full image width and height (x right, y down), an importance
   1-5, and "essential": true only for elements whose loss would change what the work is about
   (the protagonists, the focal event, faces of main figures). Background, sky, ground and
   framing elements are never essential. At most 6 essential subjects.
3. Locate every visible FACE, human or animal, with a TIGHT box around the head (same 0-1000
   coordinates), the label of the subject it belongs to, and "essential": true when that subject is
   essential. Include small and distant faces; give [] if there are none.
4. Give the focal point [x, y] (0-1000) and, if the work has one, the horizon height y (0-1000).
5. Describe the scene and the artist's evident intent.
6. Write a museum wall-label paragraph of 40-65 words about this work, as the curator who knows it:
   name its subjects the way the verified title names them (never "a figure" when the title says
   who it is), then what we see, how it is painted, and why it matters. Add no biographical or
   historical claim beyond the verified identification; everything else must be visible in the image.
Reply ONLY with JSON:
{{"genre": "...", "scene": "...", "artistic_intent": "...", "focal_point": [x, y], "horizon_y": null,
  "subjects": [{{"label": "...", "box_2d": [x0, y0, x1, y1], "importance": 3, "essential": false}}],
  "faces": [{{"of": "subject label", "box_2d": [x0, y0, x1, y1], "essential": true}}],
  "label_text": "..."}}"""


@dataclass
class Subject:
    label: str
    box: tuple[float, float, float, float]   # fractions of width/height, x0 y0 x1 y1
    importance: int
    essential: bool


@dataclass
class Interpretation:
    genre: str
    scene: str
    artistic_intent: str
    focal_point: tuple[float, float]
    horizon_y: float | None
    subjects: list[Subject] = field(default_factory=list)
    label_text: str = ""
    faces: list[Subject] = field(default_factory=list)     # label = the subject the face belongs to

    def to_dict(self) -> dict:
        return asdict(self)


def _box(b) -> tuple[float, float, float, float] | None:
    try:
        x0, y0, x1, y1 = (min(1000.0, max(0.0, float(v))) / 1000.0 for v in b)
    except (TypeError, ValueError):
        return None
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    return (x0, y0, x1, y1) if (x1 - x0) > 0.005 and (y1 - y0) > 0.005 else None


def interpret(vlm: VLM, img: Image.Image, facts: dict) -> Interpretation:
    fact_line = "; ".join(f"{k}: {v}" for k, v in facts.items() if v)
    d = vlm.ask_json(PROMPT.format(facts=fact_line, genres=", ".join(GENRES)), [img],
                     required=("genre", "subjects", "faces", "label_text", "focal_point"), max_side=1280,
                     num_predict=3072)
    subjects = []
    for s in d.get("subjects") or []:
        b = _box(s.get("box_2d"))
        if b:
            subjects.append(Subject(str(s.get("label", "")), b, int(s.get("importance") or 1),
                                    bool(s.get("essential"))))
    if not subjects:
        raise ValueError("interpretation located no subjects")
    if not isinstance(d.get("faces"), list):
        raise ValueError("interpretation gave no faces list (an empty list is a valid answer)")
    faces = []
    for f in d["faces"]:
        b = _box(f.get("box_2d")) if isinstance(f, dict) else None
        # a "face" over a third of the picture is a group the model boxed whole (a choir of angels),
        # not a head: it would gate the crop on the entire frame, so it is not taken as a face
        if b and (b[2] - b[0]) * (b[3] - b[1]) <= MAX_FACE_AREA:
            faces.append(Subject(str(f.get("of", "")), b, 5, bool(f.get("essential"))))
    fp = d.get("focal_point") or [500, 500]
    hz = d.get("horizon_y")
    genre = str(d.get("genre", "")).lower().strip()
    return Interpretation(
        genre=genre if genre in GENRES else "genre-scene" if not genre else genre,
        scene=str(d.get("scene", "")), artistic_intent=str(d.get("artistic_intent", "")),
        focal_point=(float(fp[0]) / 1000, float(fp[1]) / 1000),
        horizon_y=(float(hz) / 1000 if isinstance(hz, (int, float)) else None),
        subjects=subjects, label_text=str(d.get("label_text", "")).strip(), faces=faces)
