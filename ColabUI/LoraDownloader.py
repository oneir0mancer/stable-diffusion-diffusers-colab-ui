import os
import gdown
from IPython.display import display
from ipywidgets import Dropdown, Button, HBox, Layout, Text
from ..utils.downloader import download_ckpt
from ..utils.event import Event

class LoraDownloader:
    __cache = dict()

    def __init__(self, pipe, output_dir="Lora", cache=None):
        self.output_dir = output_dir
        self.pipe = pipe
        self.pipe.disable_lora()

        if cache is not None: self.load_cache(cache)

        self.url_text = Text(placeholder="Lora url:", layout=Layout(width="40%"))
        def url_changed(change):
            self.adapter_field.value = ""
        self.url_text.observe(url_changed, 'value')
        self.url_text.description_tooltip = "Url or filepath to Lora. Supports google drive links."

        self.adapter_field = Text(placeholder="Adapter name", layout=Layout(width='150px'))
        self.adapter_field.description_tooltip = "Custom name for Lora. Default is filename."

        self.load_button = Button(description="Load", layout=Layout(width='50px'))
        def load(b):
            self.load_lora(self.url_text.value, self.adapter_field.value)
        self.load_button.on_click(load)
        
        self.on_load_event = Event()

    def render(self):
        display(HBox([self.url_text, self.adapter_field, self.load_button]))

    def load_lora(self, url, adapter_name=""):
        if os.path.isfile(url):
            filepath = url
        elif url.startswith("https://drive.google.com"):
            id = gdown.parse_url.parse_url(url)[0]
            if not self.output_dir.endswith("/"): self.output_dir += "/"
            filepath = gdown.download(f"https://drive.google.com/uc?id={id}", self.output_dir)            
        elif url.startswith("https://civitai.com/api/download/models/") or url.endswith(".safetensors"):
            filepath = download_ckpt(url, self.output_dir)
        else:
            print("Error")
            return

        filename = os.path.basename(filepath)
        if adapter_name == "": 
            adapter_name = os.path.splitext(filename)[0]
            self.adapter_field.value = adapter_name
        self.pipe.load_lora_weights(self.output_dir, weight_name=filename, adapter_name=adapter_name)
        self.__cache[adapter_name] = filepath

        self.on_load_event.invoke(adapter_name)

    def load_cache(self, cache):
        for adapter_name, filepath in cache.items():
            try:
                filename = os.path.basename(filepath)
                self.pipe.load_lora_weights(os.path.dirname(filepath), weight_name=filename, adapter_name=adapter_name)
                self.__cache[adapter_name] = filepath
            except: pass

    @property
    def cache(self):
        return self.__cache

    @property
    def render_module(self):
        return HBox([self.url_text, self.adapter_field, self.load_button])
