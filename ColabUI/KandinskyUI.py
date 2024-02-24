import torch
from .BaseUI import BaseUI

class KandinskyUI(BaseUI): 
    def generate(self, pipe, sampler="p_sampler"):
        if self.seed_field.value > 0:
            torch.manual_seed(self.seed_field.value)

        self._metadata = self.get_metadata_string() + f"Seed: {self.seed_field.value} "

        images = model.generate_text2img(self.positive_prompt.value,
                                 negative_prior_prompt = self.negative_prompt.value,
                                 num_steps=self.steps_field.value,
                                 batch_size=self.batch_field.value, 
                                 guidance_scale=self.cfg_field.value,
                                 h=self.height_field.value, w=self.width_field.value,
                                 sampler=sampler)
        return images
