import os
import gdown
from IPython.display import display
from ipywidgets import Dropdown, Button, HBox, Layout, Text, Output
from ..utils.downloader import download_ckpt
from ..utils.event import Event
from ..utils.empty_output import EmptyOutput

class LoraDownloader:
    __cache = dict()

    def __init__(self, colab, out:Output = None, output_dir:str = "Lora", cache = None):
        self.output_dir = output_dir
        self.colab = colab
        if out is None: out = EmptyOutput()
        self.out = out

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
            self.out.clear_output()
            with self.out:
                self.load_lora(self.url_text.value, self.adapter_field.value)
        self.load_button.on_click(load)
        
        self.repair_button = Button(description="Repair", layout=Layout(width='75px'), button_style='info')
        def on_repair_btn(b):
            self.out.clear_output()
            with self.out: self.repair()
        self.repair_button.on_click(on_repair_btn)
        self.repair_button.tooltip = "Click to refresh list of loaded Loras in case of errors"
        
        self.on_load_event = Event()

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
        self.colab.pipe.load_lora_weights(self.output_dir, weight_name=filename, adapter_name=adapter_name)
        self.__cache[adapter_name] = filepath

        self.on_load_event.invoke(adapter_name)

    def load_cache(self, cache):
        for adapter_name, filepath in cache.items():
            try:
                filename = os.path.basename(filepath)
                self.colab.pipe.load_lora_weights(os.path.dirname(filepath), weight_name=filename, adapter_name=adapter_name)
                self.__cache[adapter_name] = filepath
            except: pass

    def repair(self):
        adapters_dict = self.colab.pipe.get_list_adapters()
        print(adapters_dict)
        adapters = {x for module in adapters_dict.values() for x in module}
        for a in adapters:
            self.__cache[a] = None
        self.on_load_event.invoke("") #no need to call for each adapter, TODO argument

    @property
    def cache(self):
        return self.__cache

    @property
    def render_element(self):
        return HBox([self.url_text, self.adapter_field, self.load_button, self.repair_button])

    def render(self):
        display(self.render_element)

    def _ipython_display_(self):
        self.render()
