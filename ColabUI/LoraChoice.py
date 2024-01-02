import markdown 
from ipywidgets import VBox, HTML, Output
from .LoraDownloader import LoraDownloader
from .LoraApplyer import LoraApplyer

class LoraChoice:
    def __init__(self, colab, out:Output = None, download_dir:str = "Lora", lora_cache=None):
        self.colab = colab

        self.label_download = HTML(markdown.markdown("## Load Lora")) 
        self.lora_downloader = LoraDownloader(colab.pipe, out, output_dir=download_dir, cache=lora_cache)
        self.label_apply = HTML(markdown.markdown("## Apply Lora")) 
        self.lora_ui = LoraApplyer(colab.pipe, self.lora_downloader)

        #TODO handle connection between them here

    @property
    def render_element(self): 
        return VBox([self.label_download, self.lora_downloader.render_element, self.label_apply, self.lora_ui.render_element])

    def render(self):
        display(self.render_element)

    def _ipython_display_(self):
        self.render()