import logging

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter("%(asctime)s - %(module)s - %(levelname)s - %(message)s")
