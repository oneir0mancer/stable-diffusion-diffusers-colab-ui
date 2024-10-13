from IPython.display import display
from ipywidgets import Textarea, IntText, IntSlider, FloatSlider, HBox, Layout, Label, Button
from PIL.PngImagePlugin import PngInfo
import ipywidgets as widgets
import io

class BaseUI:
    _metadata :str

    def __init__(self):
        self.positive_prompt = Textarea(placeholder='Positive prompt...', layout=Layout(width="50%"))
        self.negative_prompt = Textarea(placeholder='Negative prompt...', layout=Layout(width="50%"))
        self.width_field = IntText(value=512, layout=Layout(width='200px'), description="width: ")
        self.height_field = IntText(value=768, layout=Layout(width='200px'), description="height: ")
        self.steps_field = IntSlider(value=25, min=1, max=100, description="Steps: ")
        self.cfg_field = FloatSlider(value=7, min=1, max=20, step=0.5, description="CFG: ")
        self.seed_field = IntText(value=-1, description="seed: ")
        self.batch_field = IntText(value=1, layout=Layout(width='150px'), description="Batch size ")
        self.count_field = IntText(value=1, layout=Layout(width='150px'), description="Sample count ")
        self.clip_skip = None
        self.cfg_rescale = 0.0
        self.prompt_preprocessors = []
        
    def render(self):
        """Render UI widgets."""
        l1 = Label("Positive prompt:", layout=Layout(width="50%"))
        l2 = Label("Negative prompt:", layout=Layout(width="50%"))
        prompts = HBox([self.positive_prompt, self.negative_prompt])
        
        def swap_dims(b):
            self.width_field.value, self.height_field.value = self.height_field.value, self.width_field.value
        
        swap_btn = Button(description="⇆", layout=Layout(width='40px'))
        swap_btn.on_click(swap_dims)

        size_box = HBox([self.width_field, self.height_field, swap_btn])
        batch_box = HBox([self.batch_field, self.count_field])

        display(HBox([l1, l2]), prompts, size_box, self.steps_field, self.cfg_field, self.seed_field, batch_box)
        
    def get_size(self):
        """Get current (width, height) values"""
        return (self.width_field.value, self.height_field.value)
        
    def save_image_with_metadata(self, image, path, additional_data = ""):
        meta = PngInfo()
        meta.add_text("Data", self._metadata + additional_data)
        image.save(path, pnginfo=meta)

    def get_positive_prompt(self):
        return self.preprocess_prompt(self.positive_prompt.value)

    def get_negative_prompt(self):
        return self.preprocess_prompt(self.negative_prompt.value)

    def preprocess_prompt(self, prompt : str):
        for preprocessor in self.prompt_preprocessors:
            prompt = preprocessor.process(prompt)
        return prompt

    @property
    def metadata(self):
        return self._metadata

    @property
    def sample_count(self):
        return self.count_field.value
    
    def get_metadata_string(self):
        return f"\nPrompt: {self.positive_prompt.value}\nNegative: {self.negative_prompt.value}\nCGF: {self.cfg_field.value} Steps {self.steps_field.value} "
        
    def get_dict_to_cache(self):
        return {
            "prompt" : self.positive_prompt.value,
            "n_prompt" : self.negative_prompt.value,
            "CGF" : self.cfg_field.value,
            "steps" : self.steps_field.value,
            "h" : self.height_field.value,
            "w" : self.width_field.value,
            "batch" : self.batch_field.value,
        }

    def load_cache(self, cache):
        self.positive_prompt.value = cache["prompt"]
        self.negative_prompt.value = cache["n_prompt"]
        self.cfg_field.value = cache["CGF"]
        self.steps_field.value = cache["steps"]
        self.height_field.value = cache["h"]
        self.width_field.value = cache["w"]
        self.batch_field.value = cache["batch"]

    def _ipython_display_(self):
        self.render()
