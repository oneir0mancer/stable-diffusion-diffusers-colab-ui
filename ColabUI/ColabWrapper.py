import os
import torch
from diffusers import EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, UniPCMultistepScheduler
from diffusers import AutoencoderKL
from .ArtistIndex import ArtistIndex
from .HugginfaceModelIndex import HugginfaceModelIndex
from .LoraDownloader import LoraDownloader
from .LoraApplyer import LoraApplyer

class ColabWrapper:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.output_index = 0
        self.cache = None
        self.lora_cache = None

    def render_model_index(self, filepath: str):
        self.model_index = HugginfaceModelIndex(filepath)
        self.model_index.render()

    def load_model(self, pipeline_interface):
        model_id, from_ckpt = self.model_index.get_model_id()
        loader_func = pipeline_interface.from_single_file if from_ckpt else pipeline_interface.from_pretrained

        #TODO we can try catch here, and load file itself if diffusers doesn't want to load it for us
        self.pipe = loader_func(model_id,
            custom_pipeline="lpw_stable_diffusion",
            torch_dtype=torch.float16).to("cuda")
        self.pipe.safety_checker = None
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_xformers_memory_efficient_attention()

    def choose_sampler(self, sampler_name: str):
        config = self.pipe.scheduler.config
        match sampler_name:
            case "Euler A": sampler = EulerAncestralDiscreteScheduler.from_config(config)
            case "DPM++": sampler = DPMSolverMultistepScheduler.from_config(config)
            case "DPM++ Karras":
                sampler = DPMSolverMultistepScheduler.from_config(config)
                sampler.use_karras_sigmas = True
            case "UniPC":
                sampler = UniPCMultistepScheduler.from_config(config)
            case _: print("Unknown sampler")
        self.pipe.scheduler = sampler
        print(f"Sampler '{sampler_name}' chosen")

    def load_vae(self, id_or_path: str, subfolder: str):
        vae = AutoencoderKL.from_pretrained(id_or_path, subfolder=subfolder, torch_dtype=torch.float16).to("cuda")
        self.pipe.vae = vae

    def load_textual_inversion(self, path: str, filename: str):
        self.pipe.load_textual_inversion(path, weight_name=filename)

    def load_textual_inversions(self, root_folder: str):
        for path, subdirs, files in os.walk(root_folder):
            for name in files:
                try:
                    if os.path.splitext(name)[1] != ".pt": continue
                    self.pipe.load_textual_inversion(path, weight_name=name)
                    print(path, name)
                except: pass

    def render_lora_loader(self, output_dir="Lora"):
        self.lora_downloader = LoraDownloader(self.pipe, output_dir, self.lora_cache)
        self.lora_downloader.render()

    def render_lora_ui(self):
        self.lora_ui = LoraApplyer(self.pipe, self.lora_downloader)
        self.lora_ui.render()

    def render_generation_ui(self, ui, artist_index: str = None):
        self.ui = ui
        if (self.cache is not None): self.ui.load_cache(self.cache)
        self.ui.render()
        
        if artist_index is None: artist_index = "/content/StableDiffusionUi/artist_index.json"
        self.flair = ArtistIndex(ui, artist_index)
        self.flair.render()

    def generate(self, save_images: bool, display_previewes: bool):
        self.cache = self.ui.get_dict_to_cache()
        results = self.ui.generate(self.pipe)
        if save_images:
            for i, image in enumerate(results.images):
                path = os.path.join(self.output_dir, f"{self.output_index:05}.png")
                self.ui.save_image_with_metadata(image, path, f"Batch: {i}\n")
                print(path)
                self.output_index += 1

        if display_previewes:
            self.ui.display_image_previews(results.images)
