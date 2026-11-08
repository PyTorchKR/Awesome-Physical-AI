#!/usr/bin/env python3
"""Credibility signals for weekly discovery candidates.

A candidate is credible when its arXiv metadata reports acceptance at a top
venue or its author block names a well-known lab. Edit the lists below to
change which venues and institutions count.
"""

from __future__ import annotations

import html
import re
from urllib.parse import urlsplit


# Accepted papers usually state the venue in the arXiv comment or journal_ref.
TOP_VENUES = {
    "CoRL": r"CoRL|Conference on Robot Learning",
    "RSS": r"RSS|Robotics: Science and Systems",
    "ICRA": r"ICRA",
    "IROS": r"IROS",
    "CVPR": r"CVPR",
    "ICCV": r"ICCV",
    "ECCV": r"ECCV",
    "NeurIPS": r"NeurIPS|NIPS",
    "ICLR": r"ICLR",
    "ICML": r"ICML",
    "RA-L": r"RA-?L|Robotics and Automation Letters",
    "T-RO": r"T-?RO|Transactions on Robotics",
    "IJRR": r"IJRR|International Journal of Robotics Research",
    "Science Robotics": r"Science Robotics",
}
NOT_ACCEPTED_RE = re.compile(
    r"submitted|under review|in submission|in review|preprint version|workshop",
    re.IGNORECASE,
)

# Canonical name -> aliases matched case-sensitively on word boundaries.
# Canonical names follow the org values already used in data/models.yaml.
TRUSTED_INSTITUTIONS = {
    # Companies and industry labs
    "Google DeepMind": (r"Google DeepMind", r"DeepMind"),
    "Google": (r"Google Research", r"Google(?! DeepMind)"),
    "NVIDIA": (r"NVIDIA", r"Nvidia"),
    "Meta": (r"Meta AI", r"Meta Reality Labs", r"FAIR", r"Meta"),
    "OpenAI": (r"OpenAI",),
    "Microsoft": (r"Microsoft Research", r"Microsoft"),
    "Amazon": (r"Amazon",),
    "Apple": (r"Apple",),
    "Toyota Research Institute": (r"Toyota Research Institute",),
    "RAI Institute": (r"RAI Institute", r"The AI Institute", r"Boston Dynamics"),
    "Physical Intelligence": (r"Physical Intelligence",),
    "Hugging Face": (r"Hugging Face", r"HuggingFace"),
    "Samsung": (r"Samsung",),
    "LG AI Research": (r"LG AI Research",),
    "NAVER": (r"NAVER", r"Naver"),
    "Sony": (r"Sony",),
    "ByteDance": (r"ByteDance",),
    "Alibaba": (r"Alibaba", r"DAMO Academy"),
    "Tencent": (r"Tencent",),
    "Huawei": (r"Huawei",),
    "Shanghai AI Lab": (r"Shanghai AI Lab(?:oratory)?", r"Shanghai Artificial Intelligence Laboratory"),
    "BAAI": (r"BAAI", r"Beijing Academy of Artificial Intelligence"),
    # Universities and institutes
    "Stanford": (r"Stanford",),
    "MIT": (r"MIT", r"Massachusetts Institute of Technology"),
    "CMU": (r"CMU", r"Carnegie Mellon"),
    "UC Berkeley": (r"UC Berkeley", r"University of California,? Berkeley", r"Berkeley AI Research"),
    "UC San Diego": (r"UC San Diego", r"UCSD", r"University of California,? San Diego"),
    "UCLA": (r"UCLA", r"University of California,? Los Angeles"),
    "Harvard": (r"Harvard",),
    "Princeton": (r"Princeton",),
    "Caltech": (r"Caltech", r"California Institute of Technology"),
    "Columbia University": (r"Columbia University",),
    "Cornell": (r"Cornell",),
    "NYU": (r"NYU", r"New York University"),
    "University of Pennsylvania": (r"University of Pennsylvania", r"UPenn"),
    "University of Washington": (r"University of Washington",),
    "UT Austin": (r"UT Austin", r"University of Texas at Austin"),
    "Georgia Tech": (r"Georgia Tech", r"Georgia Institute of Technology"),
    "University of Michigan": (r"University of Michigan",),
    "UIUC": (r"UIUC", r"University of Illinois"),
    "University of Toronto": (r"University of Toronto",),
    "Mila": (r"Mila - Quebec", r"Mila, Quebec"),
    "ETH Zurich": (r"ETH Z(?:u|ü)rich",),
    "EPFL": (r"EPFL",),
    "University of Oxford": (r"University of Oxford",),
    "University of Cambridge": (r"University of Cambridge",),
    "Imperial College London": (r"Imperial College",),
    "TU Munich": (r"Technical University of Munich", r"TU Munich", r"TUM"),
    "Max Planck Institute": (r"Max Planck",),
    "KAIST": (r"KAIST",),
    "Seoul National University": (r"Seoul National University",),
    "POSTECH": (r"POSTECH",),
    "University of Tokyo": (r"University of Tokyo",),
    "Tsinghua University": (r"Tsinghua",),
    "Peking University": (r"Peking University",),
    "SJTU": (r"Shanghai Jiao Tong", r"SJTU"),
    "Zhejiang University": (r"Zhejiang University",),
    "The University of Hong Kong": (r"The University of Hong Kong", r"HKU"),
    "HKUST": (r"HKUST", r"Hong Kong University of Science and Technology"),
    "CUHK": (r"CUHK", r"Chinese University of Hong Kong"),
    "NUS": (r"National University of Singapore", r"NUS"),
    "NTU Singapore": (r"Nanyang Technological University",),
}

# Official GitHub or Hugging Face namespaces, lowercased.
TRUSTED_NAMESPACES = {
    "google-deepmind": "Google DeepMind",
    "deepmind": "Google DeepMind",
    "google-research": "Google",
    "google": "Google",
    "nvlabs": "NVIDIA",
    "nvidia": "NVIDIA",
    "nvidia-cosmos": "NVIDIA",
    "facebookresearch": "Meta",
    "facebook": "Meta",
    "openai": "OpenAI",
    "microsoft": "Microsoft",
    "amazon-science": "Amazon",
    "apple": "Apple",
    "toyotaresearchinstitute": "Toyota Research Institute",
    "tri-ml": "Toyota Research Institute",
    "physical-intelligence": "Physical Intelligence",
    "huggingface": "Hugging Face",
    "lerobot": "Hugging Face",
    "bytedance": "ByteDance",
    "bytedance-seed": "ByteDance",
    "opengvlab": "Shanghai AI Lab",
    "internrobotics": "Shanghai AI Lab",
    "stanford-iliad": "Stanford",
    "real-stanford": "Stanford",
    "mit-spark": "MIT",
    "berkeley-hipie": "UC Berkeley",
    "thu-ml": "Tsinghua University",
}

INSTITUTION_PATTERNS = {
    name: re.compile(r"(?<![\w-])(?:" + "|".join(aliases) + r")(?![\w-])")
    for name, aliases in TRUSTED_INSTITUTIONS.items()
}
VENUE_PATTERNS = {
    name: re.compile(r"(?<![\w-])(?:" + pattern + r")(?![\w-])")
    for name, pattern in TOP_VENUES.items()
}
AUTHORS_BLOCK_RE = re.compile(r'class="ltx_authors">(.*?)(?:</div>\s*</div>|$)', re.S)


def detect_venue(comment: str, journal_ref: str) -> str:
    """Return the accepted top venue named in arXiv metadata, or ""."""
    for text in (journal_ref, comment):
        if not text or NOT_ACCEPTED_RE.search(text):
            continue
        for name, pattern in VENUE_PATTERNS.items():
            if pattern.search(text):
                return name
    return ""


def extract_author_block(page_html: str) -> str:
    """Return the plain-text author and affiliation block of an arXiv HTML page."""
    # Cut at the abstract so phrases like "physical intelligence" are not read as affiliations.
    page_html = (page_html or "").split('class="ltx_abstract"', 1)[0]
    match = AUTHORS_BLOCK_RE.search(page_html)
    if not match:
        return ""
    text = html.unescape(re.sub(r"<[^>]+>", " ", match.group(1)))
    return " ".join(text.split())


def match_institutions(author_text: str, urls: list[str]) -> list[str]:
    """Return trusted institutions named in the author block or owning an official link."""
    found: list[str] = []
    for url in urls:
        parts = urlsplit(url)
        if parts.hostname not in {"github.com", "huggingface.co"}:
            continue
        segments = [s for s in parts.path.split("/") if s and s not in {"datasets", "spaces"}]
        name = TRUSTED_NAMESPACES.get(segments[0].lower()) if segments else None
        if name and name not in found:
            found.append(name)
    # Keep the author-block order so the lead author's affiliation comes first.
    positions = {}
    for name, pattern in INSTITUTION_PATTERNS.items():
        match = pattern.search(author_text or "")
        if match and name not in found:
            positions[name] = match.start()
    return found + sorted(positions, key=positions.get)


def credibility_score(venue: str, institutions: list[str]) -> int:
    return int(bool(venue)) + int(bool(institutions))
