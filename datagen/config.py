"""
Shared configuration for the meme dataset pipeline.

The label of a class is its index in CLASSES (0..9), exactly like CIFAR-10.
Change this list (and re-run the pipeline) to build a different meme dataset.

Each class has:
  name        - folder / label name (snake_case)
  display     - human readable name
  description - what counts as this class (used when curating images by hand)
  imgflip     - how scrape_imgflip.py finds candidate images:
      official : imgflip "official" template slugs (big templates with thousands of memes)
      pages    : how many pages of user-made memes to read per official template
      queries  : imgflip /memesearch queries used to discover user-uploaded templates
      include  : regex a discovered template title must match
      exclude  : regex that rejects a discovered template title
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # project root
DATA = ROOT / "data"

IMAGE_SIZE = 32          # LeNet-5 input resolution (32x32)
CHANNELS = 3             # RGB

CLASSES = [
    dict(
        name="six_seven",
        display="6-7 (six seven)",
        description="Someone doing the '6-7' hands (palms up, alternating up/down), the 67 kid, Skrilla / LaMelo 6-7 edits.",
        imgflip=dict(
            official=[],
            queries=["6 7", "67", "6 7 kid", "67 kid", "six seven", "6 7 hands", "doot doot 6 7",
                     "6 7 meme", "67 meme", "67 hands", "6 7 hand", "skrilla", "lamelo 6 7", "6-7",
                     "the 67 kid", "67 guy", "6 7 dance", "6 7 trend"],
            include=r"(?<!\d)6\s*[-_ ]?\s*7(?!\d)|six\s*seven|skrilla|doot",
            exclude=r"order|skibidi|tung|sahur|lego|tralal|1-2-3|fortune|cloud|number|mango|mustard",
        ),
    ),
    dict(
        name="tung_tung_tung_sahur",
        display="Tung Tung Tung Sahur",
        description="The Italian-brainrot wooden log character holding a baseball bat.",
        imgflip=dict(
            official=[],
            queries=["tung tung tung sahur", "tung tung", "sahur"],
            include=r"tung|sahur",
            exclude=r"6\s*7|67|tralal|skibidi|vs|and ",
        ),
    ),
    dict(
        name="tralalero_tralala",
        display="Tralalero Tralala",
        description="The Italian-brainrot blue shark wearing Nike sneakers.",
        imgflip=dict(
            official=[],
            queries=["tralalero tralala", "tralalero", "shark nike shoes", "tralala", "shark with shoes",
                     "nike shark", "shark shoes", "italian brainrot shark", "brainrot shark", "tralalelo",
                     "tra la la", "tralalero meme"],
            include=r"tralal|shark|nike",
            exclude=r"6\s*7|67|tung|sahur|skibidi|vs",
        ),
    ),
    dict(
        name="skibidi_toilet",
        display="Skibidi Toilet",
        description="A singing head sticking out of a toilet (Skibidi Toilet series).",
        imgflip=dict(
            official=[],
            queries=["skibidi toilet", "skibidi"],
            include=r"skibid|toilet",
            exclude=r"6\s*7|67|camera|speaker|tv ?man|titan|dop|vs",
        ),
    ),
    dict(
        name="chill_guy",
        display="Chill Guy",
        description="'My new character' — the relaxed cartoon dog in a grey sweater, jeans, hands in pockets.",
        imgflip=dict(
            official=[],
            queries=["chill guy", "just a chill guy", "my new character"],
            include=r"chill|new charac",
            exclude=r"scared|hajime|reaction|vs",
        ),
    ),
    dict(
        name="doge",
        display="Doge",
        description="Kabosu the Shiba Inu 'doge' (the side-eye photo and direct edits of it).",
        imgflip=dict(
            official=["Doge"],
            pages=3,
            queries=["doge"],
            include=r"doge",
            exclude=r"cheems|buff|swole|vs|walter|barney|kiss|ripped|bloody|cloud",
        ),
    ),
    dict(
        name="trollface",
        display="Trollface",
        description="The black-and-white rage-comic Trollface grin.",
        imgflip=dict(
            official=[],
            queries=["troll face", "trollface", "trollge"],
            include=r"troll",
            exclude=r"trump|derp|mask|oil|dancing|pill|santa|christmas",
        ),
    ),
    dict(
        name="surprised_pikachu",
        display="Surprised Pikachu",
        description="Pikachu with its mouth open in shock.",
        imgflip=dict(
            official=["Surprised-Pikachu"],
            pages=4,
            queries=["surprised pikachu"],
            include=r"pikachu",
            exclude=r"vs|detective|ash",
        ),
    ),
    dict(
        name="drake_hotline_bling",
        display="Drake Hotline Bling",
        description="Drakeposting: Drake in an orange jacket rejecting (top) / approving (bottom).",
        imgflip=dict(
            official=["Drake-Hotline-Bling", "Drake-Blank"],
            pages=4,
            queries=[],
            include=r"drake",
            exclude=r"vs",
        ),
    ),
    dict(
        name="distracted_boyfriend",
        display="Distracted Boyfriend",
        description="The stock photo of a man looking back at another woman while his girlfriend glares.",
        imgflip=dict(
            official=["Distracted-Boyfriend"],
            pages=5,
            queries=[],
            include=r"distracted",
            exclude=r"vs",
        ),
    ),
]

CLASS_NAMES = [c["name"] for c in CLASSES]
