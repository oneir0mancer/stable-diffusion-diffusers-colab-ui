from IPython.display import display
from ipywidgets import Textarea, IntText, IntSlider, FloatSlider, HBox, Layout, Label
import ipywidgets as widgets
import io
import torch

class DiffusionPipelineUI:
    def __init__(self):
        self.positive_prompt = Textarea(placeholder='Positive prompt...', layout=Layout(width="50%"))
        self.negative_prompt = Textarea(placeholder='Negative prompt...', layout=Layout(width="50%"))
        self.width_field = IntText(value=512, layout=Layout(width='200px'), description="width: ")
        self.height_field = IntText(value=768, layout=Layout(width='200px'), description="height: ")
        self.steps_field = IntSlider(value=20, min=1, max=100, description="Steps: ")
        self.cfg_field = FloatSlider(value=7, min=1, max=20, step=0.5, description="CFG: ")
        self.seed_field = IntText(value=-1, description="seed: ")
        self.batch_field = IntText(value=1, layout=Layout(width='150px'), description="Batch size ")
        self.count_field = IntText(value=1, layout=Layout(width='150px'), description="Batch count ")
        
    def render(self):
        """Render UI widgets."""
        l1 = Label("Positive prompt:", layout=Layout(width="50%"))
        l2 = Label("Negative prompt:", layout=Layout(width="50%"))
        prompts = HBox([self.positive_prompt, self.negative_prompt])
        size_box = HBox([self.width_field, self.height_field])
        batch_box = HBox([self.batch_field])
        display(HBox([l1, l2], prompts, size_box, self.steps_field, self.cfg_field, self.seed_field, batch_box)
        
    def generate(self, pipe, generator = None):
        """Generate images given DiffusionPipeline, torch.GEnerator, and settings set in UI."""
        if generator is None or self.seed_field.value >= 0: 
            generator = torch.manual_seed(seed_field.value)

        results = pipe([self.positive_prompt.value]*self.batch_field.value, 
                       negative_prompt=[self.negative_prompt.value]*self.batch_field.value, 
                       num_inference_steps=self.steps_field.value, 
                       guidance_scale=self.cfg_field.value, 
                       generator=generator, 
                       height=self.height_field.value, width=self.width_field.value)
        return results
        
    def get_size(self):
        """Get current (width, height) values"""
        return (self.width_field.value, self.height_field.value)
        
    def display_image_previews(self, images):
        width, height = self.get_size()
        wgts = []
        for image in images:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            wgt = widgets.Image(
                value=img_byte_arr.getvalue(),
                width=width/2,
                height=height/2,
            )
            wgts.append(wgt)
        display(HBox(wgts))
