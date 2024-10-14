import logging
import sys

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter("%(asctime)s - %(module)s - %(levelname)s - %(message)s")

# create console handler and set level to info
ch = logging.StreamHandler(stream=sys.stderr)
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)
