from stream_agents.core.processors.base_processor import Processor
from PIL import Image
from roboflow import Roboflow


class RoboflowProcessor(Processor):
    """
    Roboflow processor.
    """
    def __init__(self, api_key: str = None):
        super().__init__()
        self.roboflow = Roboflow(api_key=api_key)

    def process_image(self, image: Image.Image) -> Image.Image:
        return image