"""Shared naming conventions and regular expressions for KASKI IDs.

Reference: [PROJECT_]TYPE_NAME[_ARTIST]_VERSION
Shot:      [PROJECT_]shNUMBER_STAGE[_ARTIST]_VERSION
Sequence:  [PROJECT_]seqNUMBER[-SHOT]_STAGE[_ARTIST]_VERSION

Project and artist are optional; blank or NONE omits them. Modify accepts KEEP
for unchanged text fields. Numeric versions use v<number>; vN is unspecified.
Extract returns the first ID and discards filename suffixes and extensions.
Sequence shot numbers use the width of their sequence number.
"""

import re

CATEGORY = "KASKI/ID-Tools"


######### STAGE TYPES ###########

# References
REFERENCE_TYPES = [
    "location",
    "prop",
    "character",
    "material",
]

# Shots
SHOT_STAGES = [
    "firstFrame",
    "lastFrame",
    "referenceFrame",
    "ffToCleanup",
    "lfToCleanup",
    "rfToCleanup",
    "Depth",
    "Normal",
    "cgi",
    "Scribble",
    "plate",
    "plateToCleanup",
    "notEnhanced",
    "enhanced",
]

# Sequences
SEQUENCE_STAGES = [
    "plate",
    "depth",
    "rendered",
    "scribble",
    "toCleanup",
    "notEnhanced",
    "enhanced",
]





######### REGEX LOGIC ###########

# COMMON
# Allowed identifier components (letters, numbers and hyphens).
ID_COMPONENT_PATTERN = r"[A-Za-z0-9-]+"

# ID Patterns
ID_SUFFIX_PATTERN = (
    r"(?:_(?P<artist_code>[A-Za-z0-9-]+))?_"
    r"(?P<version>v(?:[0-9]+|N))"
)

# Try an explicit version first. A ComfyUI counter like _00001_.png is a
# filename suffix and can follow a versionless ID without an artist code.
EXTRACT_ID_SUFFIX_PATTERN = (
    rf"(?:{ID_SUFFIX_PATTERN}"
    r"|(?:_(?![0-9]{5,}_\.)[A-Za-z0-9-]+)?(?![A-Za-z0-9-]))"
)




# REFERENCES
REFERENCE_TYPE_PATTERN = "|".join(map(re.escape, REFERENCE_TYPES))

REFERENCE_ID_BASE_PATTERN = (
    r"(?<![A-Za-z0-9-])"
    rf"(?:(?P<project_name>(?!(?:{REFERENCE_TYPE_PATTERN})_)[A-Za-z0-9-]+)_)?"
    rf"(?P<reference_type>{REFERENCE_TYPE_PATTERN})_"
    rf"(?P<reference_name>[A-Za-z0-9-]+)"
)

REGEX_REFERENCE_ID = re.compile(REFERENCE_ID_BASE_PATTERN + ID_SUFFIX_PATTERN)
REGEX_EXTRACT_REFERENCE_ID = re.compile(
    REFERENCE_ID_BASE_PATTERN + EXTRACT_ID_SUFFIX_PATTERN
)



# SHOT
SHOT_STAGE_PATTERN = "|".join(map(re.escape, SHOT_STAGES))

SHOT_ID_BASE_PATTERN = (
    rf"(?:(?P<project_name>[A-Za-z0-9-]+)_)?"
    rf"(?P<shot>sh[0-9]+)_"
    rf"(?P<pipeline_step>{SHOT_STAGE_PATTERN})"
)

REGEX_SHOT_ID = re.compile(SHOT_ID_BASE_PATTERN + ID_SUFFIX_PATTERN)
REGEX_EXTRACT_SHOT_ID = re.compile(SHOT_ID_BASE_PATTERN + EXTRACT_ID_SUFFIX_PATTERN)





# SEQUENCE
SEQUENCE_NUMBER_PATTERN = r"[0-9]+(?:-[0-9]+)?"

SEQUENCE_STAGE_PATTERN = "|".join(map(re.escape, SEQUENCE_STAGES))

SEQUENCE_ID_BASE_PATTERN = (
    rf"(?:(?P<project_name>[A-Za-z0-9-]+)_)?"
    rf"(?P<sequence>seq{SEQUENCE_NUMBER_PATTERN})_"
    rf"(?P<pipeline_step>{SEQUENCE_STAGE_PATTERN})"
)

REGEX_SEQUENCE_ID = re.compile(SEQUENCE_ID_BASE_PATTERN + ID_SUFFIX_PATTERN)
REGEX_EXTRACT_SEQUENCE_ID = re.compile(
    SEQUENCE_ID_BASE_PATTERN + EXTRACT_ID_SUFFIX_PATTERN
)
