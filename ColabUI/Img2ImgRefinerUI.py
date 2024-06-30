import torch
from IPython.display import display
from ipywidgets import FloatSlider, Dropdown, HBox, VBox, Text, Button, Label
from PIL import Image
import ipywidgets as widgets
import io
import os
from .BaseUI import BaseUI
from ..utils.image_utils import load_image_metadata

class Img2ImgRefinerUI:
    def __init__(self, ui: BaseUI):
        self.__generator = torch.Generator(device="cuda")
        self.__base_ui = ui
        self.__current_file = ""
        self.__current_img = None
        self.__preview_visible = True
        
        self.strength_field = FloatSlider(value=0.35, min=0, max=1, step=0.05, description="Strength: ")
        self.upscale_field = FloatSlider(value=1, min=1, max=4, step=0.25, description="Upscale factor: ")
        self.path_field = Text(description="Url:", placeholder="Image path...")
        self.load_button = Button(description="Choose")
        self.load_prompts_button = Button(description="Load prompts")
        self.image_preview = widgets.Image(width=300, height=400)
        self.hide_button = Button(description="Hide")
        self.size_label = Label(value="Size")
        
        def load_handler(b):
            if not os.path.isfile(self.path_field.value): return
            self.__current_file = self.path_field.value
            self.image_preview.value = open(self.__current_file, "rb").read()
            self.__current_img = Image.open(self.__current_file)
            self.size_label.value = f"{self.__current_img.size} → {self.get_upscaled_size(self.__current_img.size)}"
        self.load_button.on_click(load_handler)
        
        def load_prompts_handler(b):
            if not os.path.isfile(self.path_field.value): return
            prompt, negative = load_image_metadata(self.path_field.value)
            self.__base_ui.positive_prompt.value = prompt
            self.__base_ui.negative_prompt.value = negative
        self.load_prompts_button.on_click(load_prompts_handler)

        def hide_handler(b):
            self.__preview_visible = not self.__preview_visible
            self.image_preview.layout.width = "300px" if self.__preview_visible else "0px"
            self.hide_button.description = "Hide" if self.__preview_visible else "Show preview"
        self.hide_button.on_click(hide_handler)

        def on_slider_change(change):
            if self.__current_img is None: return
            self.size_label.value = f"{self.__current_img.size} → {self.get_upscaled_size(self.__current_img.size)}"
        self.upscale_field.observe(on_slider_change, 'value')
        
    def generate(self, pipe, init_image=None, generator=None):
        """Generate images given Img2Img Pipeline, and settings set in Base UI and Refiner UI."""
        if self.__base_ui.seed_field.value >= 0: 
            seed = self.__base_ui.seed_field.value
        else:
            seed = self.__generator.seed()
            
        if init_image is None:
            if not os.path.isfile(self.__current_file): return
            init_image = Image.open(self.__current_file)
        
        init_image = init_image.convert('RGB')
        size = self.get_upscaled_size(init_image.size)
        init_image = init_image.resize(size, resample=Image.LANCZOS)
        
        g = torch.cuda.manual_seed(seed)
        self._metadata = self.__base_ui.get_metadata_string() + f"\nImg2Img Seed: {seed}, Noise Strength: {self.strength_field.value}, Upscale: {self.upscale_field.value} "
        
        results = pipe(image=init_image,
                       prompt=self.__base_ui.positive_prompt.value, 
                       negative_prompt=self.__base_ui.negative_prompt.value, 
                       num_inference_steps=self.__base_ui.steps_field.value,
                       num_images_per_prompt = self.__base_ui.batch_field.value,
                       guidance_scale=self.__base_ui.cfg_field.value, 
                       guidance_rescale=self.__base_ui.cfg_rescale,
                       strength=self.strength_field.value,
                       width=size[0], height=size[1],
                       generator=g)
        return results

    @property
    def metadata(self):
        return self._metadata 

    def get_upscaled_size(self, size):
        return (int(self.upscale_field.value * size[0]), int(self.upscale_field.value * size[1]))

    @property
    def render_element(self): 
        return VBox([self.strength_field, self.upscale_field, 
                     HBox([self.path_field, self.load_button, self.load_prompts_button]), 
                     self.size_label, self.image_preview, self.hide_button])

    def render(self):
        display(self.render_element)

    def _ipython_display_(self):
        self.render()
