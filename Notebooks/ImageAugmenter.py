import base64
import io
from PIL import Image
import numpy as np

class ImageAugmenter:
    def __init__(self, image_path=None, image_bytes=None):
        if image_path:
            self.image = Image.open(image_path)
        elif image_bytes:
            self.image = Image.open(io.BytesIO(image_bytes))
        else:
            raise ValueError("Either image_path or image_bytes must be provided")
        self.working_image = self.image.copy()
        
    def rotate(self, angle):
        self.working_image = self.working_image.rotate(angle, expand=True)
        return self
    
    def flip_horizontal(self):
        self.working_image = self.working_image.transpose(Image.FLIP_LEFT_RIGHT)
        return self
        
    def adjust_brightness(self, factor):
        self.working_image = Image.fromarray((np.array(self.working_image) * factor).astype(np.uint8))
        return self
    
    def reset(self):
        self.working_image = self.image.copy()
        return self
    
    def encode(self):
        buffer = io.BytesIO()
        self.working_image.save(buffer, format=self.image.format or 'PNG')
        encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
        self.reset()  # Reset for next chain of operations
        return encoded