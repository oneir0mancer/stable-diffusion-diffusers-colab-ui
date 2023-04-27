from IPython.display import display
from ipywidgets import Dropdown, HTML, HBox
import json

class HugginfaceModelIndex:
    def __init__(self, filepath = "model_index.json"):
        with open(filepath) as f:
            self.data = json.load(f)
            
        self.model_dropdown = Dropdown(
            options=[x for x in self.data],
            description="Model:",
        )
        self.model_link = HTML()
        
        def dropdown_eventhandler(change):
            self.set_link_from_dict(change.new)
        
        self.model_dropdown.observe(dropdown_eventhandler, names='value')
        
    def set_link_from_dict(self, key):
        self.model_link.value = f"Model info: <a href=https://huggingface.co/{self.data[key]['id']}>{key}</a>"
        try:
            self.model_link.value += f"<br>Trigger prompt: <code>{self.data[key]['trigger']}</code>"
        except: pass
        
    def render(self):
        self.set_link_from_dict(self.model_dropdown.value)
        display(HBox([self.model_dropdown]), self.model_link)
        
    def get_model_id(self):
        return self.data[self.model_dropdown.value]['id']
