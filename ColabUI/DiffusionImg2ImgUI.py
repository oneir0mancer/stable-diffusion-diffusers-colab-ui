import torch
from IPython.display import display
from ipywidgets import FloatSlider, Dropdown, HBox, Text, Button, FileUpload
from PIL import Image
import ipywidgets as widgets
import io
from .BaseUI import BaseUI

class DiffusionImg2ImgUI(BaseUI):
    _current_file :str
    _current_image :Image

    def __init__(self):
        super().__init__()
        self.__generator = torch.Generator(device="cuda")
        self.strength_field = FloatSlider(value=0.75, min=0, max=1, step=0.05, description="Strength: ")
        self.path_field = Text(description="Url:", placeholder="Image path...")
        self.upload_button = Button(description='Choose')
        self.image_preview = widgets.Image()

        def uploader_eventhandler(b):
            self._current_file = self.path_field.value
            temp = widgets.Image().from_file(self._current_file)
            self.image_preview.value = temp.value
            self._current_image = Image.open(self._current_file)
            self.set_size(self._current_image)
        self.upload_button.on_click(uploader_eventhandler)

    def render(self):
        super().render()
        display(self.strength_field, HBox([self.path_field, self.upload_button]), self.image_preview)

    def set_size(self, image, scaling_factor=1):
        (self.width_field.value, self.height_field.value) = tuple(scaling_factor * x for x in image.size)

    def generate(self, pipe, init_image=None, generator=None):
        """Generate images given DiffusionPipeline, and settings set in UI."""
        if self.seed_field.value >= 0: 
            seed = self.seed_field.value
        else:
            seed = self.__generator.seed()

        if init_image is None: init_image = self._current_image

        g = torch.cuda.manual_seed(seed)
        self._metadata = self.get_metadata_string() + f"Seed: {seed} "

        init_image = init_image.convert('RGB')
        init_image = init_image.resize((self.width_field.value, self.height_field.value), resample=Image.BILINEAR)

        results = pipe(image=init_image,
                       prompt=self.positive_prompt.value, 
                       negative_prompt=self.negative_prompt.value, 
                       num_inference_steps=self.steps_field.value,
                       num_images_per_prompt = self.batch_field.value,
                       guidance_scale=self.cfg_field.value, 
                       strength=self.strength_field.value,
                       generator=g)
        return results

    def get_dict_to_cache(self):
        cache = super().get_dict_to_cache()
        cache["image_path"] = self.path_field.value
        return cache

    def load_cache(self, cache):
        super().load_cache(cache)
        self.path_field.value = cache["image_path"]
