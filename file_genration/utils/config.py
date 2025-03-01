import colorlog # type: ignore
from colorlog import ColoredFormatter # type: ignore
import logging

# Configure logging
formatter = ColoredFormatter(
    "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)

handler = colorlog.StreamHandler()
handler.setFormatter(formatter)
logger = colorlog.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 65536,
    "response_mime_type": "text/plain",
}
