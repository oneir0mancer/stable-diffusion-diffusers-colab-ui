import os
import torch
from ipywidgets import Text, Layout, Button, HBox, VBox, Output
from diffusers import AutoencoderKL
from ..utils.downloader import download_ckpt
from ..utils.empty_output import EmptyOutput
from ..utils.markdown import SpoilerLabel

class VaeChoice:
    def __init__(self, colab, out:Output = None, default_id:str = "waifu-diffusion/wd-1-5-beta2"):
        self.colab = colab
        if out is None: out = EmptyOutput()
        self.out = out

        self.tooltip_label = SpoilerLabel("Loading single file", "If vae is in single file, just leave 'Subfolder' field empty.")
        self.id_text = Text(description="VAE:", placeholder='Id or path...', layout=Layout(width="35%"))
        self.id_text.description_tooltip = "Huggingface model id, url, or a path to a root folder/file"
        if default_id is not None: self.id_text.value = default_id
        self.subfolder_text = Text(description="Subfolder:", layout=Layout(width="25%"))
        self.subfolder_text.description_tooltip = "Subfolder name, if using diffusers-style model"
        if default_id is not None: self.subfolder_text.value = "vae"

        self.button = Button(description="Load", button_style='success')

        def on_button_clicked(b):
            self.out.clear_output()
            with self.out: 
                self.load_vae(self.colab.pipe)

        self.button.on_click(on_button_clicked)

    def load_vae(self, pipe):
        if self.id_text.value == "": 
            raise ValueError("VAE text field shouldn't be empty")

        url = self.id_text.value
        if url.endswith(".safetensors") or url.startswith("https://civitai.com/api/download/models/"):
            if not os.path.isfile(url): url = download_ckpt(url)    #TODO download dir
            vae = AutoencoderKL.from_single_file(url, torch_dtype=torch.float16)
        else:
            if self.subfolder_text.value == "": 
                raise ValueError("Subfolder text field shouldn't be empty when loading diffusers-style model")
            vae = AutoencoderKL.from_pretrained(url, subfolder=self.subfolder_text.value, torch_dtype=torch.float16)

        pipe.vae = vae.to("cuda")
        print(f"{self.id_text.value} VAE loaded")

    @property
    def render_element(self): 
        return VBox([self.tooltip_label, HBox([self.id_text, self.subfolder_text]), self.button])

    def render(self):
        display(self.render_element)

    def _ipython_display_(self):
        self.render()
