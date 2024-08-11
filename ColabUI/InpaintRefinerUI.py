import io
import os
import torch
import numpy as np
from ipywidgets import Text, FloatSlider, HBox, VBox, Button, Accordion
from PIL import Image
from .CanvasHolder import CanvasHolder

#TODO can blur: https://huggingface.co/docs/diffusers/using-diffusers/inpaint

class InpaintRefinerUI:
    def __init__(self, ui):
        self.__generator = torch.Generator(device="cuda")
        self.__base_ui = ui
        self.__current_file = ""
        self.scale_factor = 0.5

        self.canvas_holder = CanvasHolder(200, 200)
        self.strength_field = FloatSlider(value=0.35, min=0, max=1, step=0.05, description="Strength: ")
        self.path_field = Text(description="Url:", placeholder="Image path...")
        self.load_button = Button(description="Choose")
        self.load_prompts_button = Button(description="Load prompts")

        def load_handler(b):
            if not os.path.isfile(self.path_field.value): return
            self.__current_file = self.path_field.value
            self.canvas_holder.add_background(self.path_field.value, scale_factor = self.scale_factor)
        self.load_button.on_click(load_handler)

        #TODO accoedion doesn't work for some reason
        self.accordion = Accordion(children=[self.canvas_holder.render_element])
        self.accordion.set_title(0, "Canvas")

    def generate(self, pipe, init_image=None, mask=None, generator=None):
        """Generate images given Inpaint Pipeline, and settings set in Base UI and Refiner UI."""
        if generator is None: generator = self.__generator
        if self.__base_ui.seed_field.value >= 0: 
            seed = self.__base_ui.seed_field.value
        else:
            seed = generator.seed()

        if init_image is None:
            init_image = Image.open(self.__current_file)
        size = init_image.size
        # TODO check if mask size matches, or always scale to img?
        if mask is None:
            mask = Image.fromarray(self.canvas_holder.canvas.get_image_data()[:,:,-1])
            if self.scale_factor != 1:
                mask = mask.resize((int(mask.size[0] / self.scale_factor), int(mask.size[1] / self.scale_factor)))

        g = torch.cuda.manual_seed(seed)
        self._metadata = self.__base_ui.get_metadata_string() + f"\nImg2Img Seed: {seed}, Noise Strength: {self.strength_field.value} "
        
        results = pipe(image=init_image,
                       mask_image=mask,
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

    @property
    def render_element(self): 
        return VBox([self.strength_field, 
                    HBox([self.path_field, self.load_button, self.load_prompts_button]), 
                    self.canvas_holder.render_element])

    def render(self):
        display(self.render_element)

    def _ipython_display_(self):
        self.render()
