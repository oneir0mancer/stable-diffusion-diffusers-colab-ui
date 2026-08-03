import torch
from sd_embed.embedding_funcs import get_weighted_text_embeddings_sd15
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
        self._metadata = self.get_metadata_string() + f"Seed: {seed} "
        
        (
            prompt_embeds, prompt_neg_embeds, 
        ) = get_weighted_text_embeddings_sd15(
            pipe, 
            prompt = self.get_positive_prompt(), 
            neg_prompt = self.get_negative_prompt()
        )

        results = pipe(prompt_embeds = prompt_embeds, 
                       negative_prompt_embeds = prompt_neg_embeds, 
                       num_inference_steps=self.steps_field.value,
                       num_images_per_prompt = self.batch_field.value,
                       guidance_scale=self.cfg_field.value, 
                       guidance_rescale=self.cfg_rescale,
                       generator=g, clip_skip=self.clip_skip,
                       height=self.height_field.value, width=self.width_field.value)
        
        del prompt_embeds, prompt_neg_embeds
        return results
