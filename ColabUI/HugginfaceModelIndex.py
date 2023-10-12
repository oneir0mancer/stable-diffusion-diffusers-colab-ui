from IPython.display import display
from ipywidgets import Dropdown, HTML, HBox, Layout, Text, Checkbox
import json
import requests
import werkzeug
import os
from tqdm import tqdm

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
            return self.download_ckpt(self.url_text.value), self.from_ckpt.value
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
            self.from_ckpt.disabled = False
    
    @staticmethod
    def highlight(str_to_highlight): 
        return f"<font color='green'>{str_to_highlight}</font>"

    @staticmethod
    def download_ckpt(ckpt_url):
        """
        download the checkpoint & return file name to be used in diffusers pipeline
        additional actions:
        - try to get file name from header or url
        - download only if file not exist
        - download with progress bar
        """
        HEADERS = {"User-Agent": "github/oneir0mancer/stable-diffusion-diffusers-colab-ui"}
        with requests.get(ckpt_url, headers=HEADERS, stream=True) as resp:

            # get file name
            MISSING_FILENAME = "missing_name"
            if content_disposition := resp.headers.get("Content-Disposition"):
                param, options = werkzeug.http.parse_options_header(content_disposition)
                if param == "attachment":
                    filename = options.get("filename", MISSING_FILENAME)
                else:
                    filename = MISSING_FILENAME
            else:
                filename = os.path.basename(ckpt_url)
                fileext = os.path.splitext(filename)[-1]
                if fileext == "":
                    filename = MISSING_FILENAME

            # download file
            if not os.path.exists(filename):
                TOTAL_SIZE = int(resp.headers.get("Content-Length", 0))
                CHUNK_SIZE = 50 * 1024**2  # 50 MiB or more as colab has high speed internet
                with open(filename, mode="wb") as file, tqdm(total=TOTAL_SIZE, desc="download checkpoint", unit="iB", unit_scale=True) as progress_bar:
                    for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                        progress_bar.update(len(chunk))
                        file.write(chunk)
        return filename