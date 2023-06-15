import torch
from .BaseUI import BaseUI

class DiffusionPipelineUI(BaseUI):
    def __init__(self):
        super().__init__()
        self.__generator = torch.Generator(device="cuda")

    def generate(self, pipe, generator = None):
        """Generate images given DiffusionPipeline, and settings set in UI."""
        if self.seed_field.value >= 0: 
            seed = self.seed_field.value
        else:
            seed = self.__generator.seed()

        g = torch.cuda.manual_seed(seed)
        self._metadata = self._get_metadata_string() + f"Seed: {seed} "

        results = pipe(prompt=self.positive_prompt.value, 
                       negative_prompt=self.negative_prompt.value, 
                       num_inference_steps=self.steps_field.value,
                       num_images_per_prompt = self.batch_field.value,
                       guidance_scale=self.cfg_field.value, 
                       generator=g, 
                       height=self.height_field.value, width=self.width_field.value)
        return results
