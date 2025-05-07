from dotenv import load_dotenv
import logging

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # logging.FileHandler("chipmunk.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


__all__ = [
    'logger'
]
