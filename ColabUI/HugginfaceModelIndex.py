from IPython.display import display
from ipywidgets import Dropdown, HTML, HBox, Layout, Text, Checkbox
import json
import requests
import werkzeug
import os
from tqdm.notebook import tqdm
from ..utils.downloader import download_ckpt

class HugginfaceModelIndex:
    def __init__(self, filepath = "model_index.json"):
        with open(filepath) as f:
            self.data = json.load(f)

        self.model_link = HTML()
        self.from_ckpt = Checkbox(value=False, description="A1111 format")

        self.__setup_url_field()
        self.__setup_model_dropdown()

    def render(self):
        """Display ui"""
        self.__set_link_from_dict(self.model_dropdown.value)
        display(self.model_dropdown, HBox([self.url_text, self.from_ckpt]), self.model_link)

    def get_model_id(self):
        """Return model_id/url/local path of the model, and whether it should be loaded with from_ckpt"""
        if self.url_text.value != "":
            return self.__handle_url_value(self.url_text.value)
        return self.data[self.model_dropdown.value]["id"], False

    def __setup_model_dropdown(self):
        self.model_dropdown = Dropdown(
            options=[x for x in self.data],
            description=self.highlight("Model:"),
        )
        def dropdown_eventhandler(change):
            self.__set_link_from_dict(change.new)
        self.model_dropdown.observe(dropdown_eventhandler, names='value')
        self.model_dropdown.description_tooltip = "Choose model from model index"

    def __setup_url_field(self):
        self.url_text = Text(description="Url:", placeholder='Optional url or path...', layout=Layout(width="50%"))
        def url_eventhandler(change):
            self.__set_link_from_url(change.new)
        self.url_text.observe(url_eventhandler, names='value')
        self.url_text.description_tooltip = "Model_id, url, or local path for any other model not in index"

    def __set_link_from_dict(self, key):
        self.from_ckpt.value = False
        self.from_ckpt.disabled = True
        self.url_text.value = ""
        self.model_link.value = f"Model info: <a href=https://huggingface.co/{self.data[key]['id']}>{key}</a>"
        try:
            self.model_link.value += f"<br>Trigger prompt: <code>{self.data[key]['trigger']}</code>"
        except: pass

    def __set_link_from_url(self, new_url):
        if new_url == "":
            self.__set_link_from_dict(self.model_dropdown.value)
            self.url_text.description = "Url:"
            self.model_dropdown.description = self.highlight("Model:")
            self.from_ckpt.value = False
            self.from_ckpt.disabled = True
        else:
            self.model_link.value = f"Model info: <a href={new_url}>link</a>"
            self.url_text.description = self.highlight("Url:")
            self.model_dropdown.description = "Model:"
            if new_url.startswith("https://civitai.com/api/download/models/") or new_url.endswith(".safetensors"):
                self.from_ckpt.value = True
            elif self.is_huggingface_model_id(new_url):
                self.from_ckpt.value = False
            elif os.path.exists(new_url):
                self.from_ckpt.value = os.path.isfile(new_url)
            self.from_ckpt.disabled = False

    def __handle_url_value(self, url):
        if self.is_huggingface_model_id(url):
            return url, False
        elif os.path.exists(url):
            return url, self.from_ckpt.value
        else:
            return download_ckpt(url), self.from_ckpt.value

    @staticmethod
    def highlight(str_to_highlight):
        return f"<font color='green'>{str_to_highlight}</font>"

    @staticmethod
    def is_huggingface_model_id(url):
        return len(url.split('/')) == 2

    def _ipython_display_(self):
        self.render()
