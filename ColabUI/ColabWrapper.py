import os
import torch
from diffusers import EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, UniPCMultistepScheduler
from diffusers import AutoencoderKL
from .FlairView import FlairView
from .HugginfaceModelIndex import HugginfaceModelIndex
from .LoraDownloader import LoraDownloader
from .LoraApplyer import LoraApplyer
from .SettingsTabs import SettingsTabs
from .Img2ImgRefinerUI import Img2ImgRefinerUI
from ..utils.preview import get_image_previews, try_create_dir
from ..utils.image_utils import save_image_with_metadata

class ColabWrapper:
    def __init__(self, output_dir: str = "outputs/txt2img", img2img_dir: str = "outputs/img2img",
                 favourite_dir: str = "outputs/favourite"):
        self.output_dir = output_dir
        self.img2img_dir = img2img_dir
        self.favourite_dir = favourite_dir
        self.output_index = 0
        self.cache = None
        self.lora_cache = dict()
        self.pipe = None
        self.custom_pipeline = None
        
        try_create_dir(output_dir)
        try_create_dir(favourite_dir)

    def render_model_index(self, filepath: str):
        self.model_index = HugginfaceModelIndex(filepath)
        self.model_index.render()

    def load_model(self, pipeline_interface, custom_pipeline:str = "lpw_stable_diffusion"):
        model_id, from_ckpt = self.model_index.get_model_id()
        loader_func = pipeline_interface.from_single_file if from_ckpt else pipeline_interface.from_pretrained

        self.custom_pipeline = custom_pipeline

        #TODO we can try catch here, and load file itself if diffusers doesn't want to load it for us
        self.pipe = loader_func(model_id,
            custom_pipeline=self.custom_pipeline,
            torch_dtype=torch.float16).to("cuda")
        self.pipe.safety_checker = None
        self.pipe.enable_xformers_memory_efficient_attention()

    def load_vae(self, id_or_path: str, subfolder: str):
        vae = AutoencoderKL.from_pretrained(id_or_path, subfolder=subfolder, torch_dtype=torch.float16).to("cuda")
        self.pipe.vae = vae

    def load_textual_inversion(self, path: str, filename: str):
        self.pipe.load_textual_inversion(path, weight_name=filename)

    def load_textual_inversions(self, root_folder: str):
        for path, subdirs, files in os.walk(root_folder):
            for name in files:
                try:
                    ext = os.path.splitext(name)[1]
                    if ext != ".pt" and ext != ".safetensors": continue
                    self.pipe.load_textual_inversion(path, weight_name=name)
                    print(path, name)
                except: pass

    def render_lora_loader(self, output_dir="Lora"):
        self.lora_downloader = LoraDownloader(self.pipe, output_dir=output_dir, cache=self.lora_cache)
        self.lora_downloader.render()

    def render_lora_ui(self):
        self.lora_ui = LoraApplyer(self.pipe, self.lora_downloader)
        self.lora_ui.render()

    def render_generation_ui(self, ui):
        self.ui = ui
        if (self.cache is not None): self.ui.load_cache(self.cache)
        
        self.settings = SettingsTabs(self)
        
        self.flair = FlairView(ui)
        
        self.settings.render()
        self.ui.render()
        self.flair.render()

    def restore_cached_ui(self):
        if (self.cache is not None): self.ui.load_cache(self.cache)

    def generate(self):
        self.cache = self.ui.get_dict_to_cache()
        paths = []
        
        for sample in range(self.ui.sample_count):
            results = self.ui.generate(self.pipe)        
            for i, image in enumerate(results.images):
                path = os.path.join(self.output_dir, f"{self.output_index:05}.png")
                save_image_with_metadata(image, path, self.ui, f"Batch: {i}\n")
                self.output_index += 1
                paths.append(path)
        
        if self.settings.display_previews:
            display(get_image_previews(paths, 512, 512, favourite_dir=self.favourite_dir))
        
        return paths

    #StableDiffusionXLImg2ImgPipeline
    def render_refiner_ui(self, pipeline_interface):
        components = self.pipe.components
        self.img2img_pipe = pipeline_interface(**components)
        
        self.refiner_ui = Img2ImgRefinerUI(self.ui)
        self.refiner_ui.render()

    def refiner_generate(self, output_dir: str = None):
        results = self.refiner_ui.generate(self.img2img_pipe)
        paths = []
        
        if output_dir is None: output_dir = self.img2img_dir
        for i, image in enumerate(results.images):
            path = os.path.join(output_dir, f"{self.output_index:05}.png")
            save_image_with_metadata(image, path, self.refiner_ui, f"Batch: {i}\n")
            self.output_index += 1
            paths.append(path)
        
        if self.settings.display_previews:
            display(get_image_previews(paths, 512, 512, favourite_dir=self.favourite_dir))
        return paths

    def render_inpaint_ui(self, pipeline_interface, ui):
        components = self.pipe.components
        self.inpaint_pipe = pipeline_interface(**components)
        
        self.inpaint_ui = ui
        self.inpaint_ui.render()

    def inpaint_generate(self, output_dir: str = None):
        results = self.inpaint_ui.generate(self.inpaint_pipe)
        paths = []
        
        if output_dir is None: output_dir = self.img2img_dir
        for i, image in enumerate(results.images):
            path = os.path.join(output_dir, f"{self.output_index:05}.png")
            save_image_with_metadata(image, path, self.inpaint_ui, f"Batch: {i}\n")
            self.output_index += 1
            paths.append(path)
        
        if self.settings.display_previews:
            display(get_image_previews(paths, 512, 512, favourite_dir=self.favourite_dir))
        return paths
