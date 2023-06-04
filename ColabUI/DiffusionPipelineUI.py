import torch
from .BaseUI import BaseUI

class DiffusionPipelineUI(BaseUI): 
    def generate(self, pipe, generator = None):
        """Generate images given DiffusionPipeline, torch.GEnerator, and settings set in UI."""
        if self.seed_field.value >= 0: 
            generator = torch.manual_seed(self.seed_field.value)
        else:
            generator = torch.Generator("cuda")

        results = pipe([self.positive_prompt.value]*self.batch_field.value, 
                       negative_prompt=[self.negative_prompt.value]*self.batch_field.value, 
                       num_inference_steps=self.steps_field.value, 
                       guidance_scale=self.cfg_field.value, 
                       generator=generator, 
                       height=self.height_field.value, width=self.width_field.value)
        return results
