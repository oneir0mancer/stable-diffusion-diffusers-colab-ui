import os
from os.path import isfile, join
from ipywidgets import Dropdown, Button, Image, VBox, HBox

class ImageViewer:
    def __init__(self, path: str = "outputs/favourite", width: int = 512, height: int = 512):
        self.path = path
        self.files = sorted([f for f in os.listdir(path) if isfile(join(path, f))])
        self.dropdown = Dropdown(
            options=[x for x in self.files],
            description="Image:",
        )
        self.preview = Image(
            width=width,
            height=height
        )
        if len(self.files) > 0: self.preview.value = open(join(self.path, self.files[0]), "rb").read()

        def dropdown_eventhandler(change):
            self.preview.value=open(join(self.path, change.new), "rb").read()
        self.dropdown.observe(dropdown_eventhandler, names='value')
        
        self.btn_refresh = Button(description="Refresh")
        def btn_handler(b):
            self.refresh()
        self.btn_refresh.on_click(btn_handler)

    def refresh(self):
        value = self.dropdown.value
        self.files = sorted([f for f in os.listdir(self.path) if isfile(join(self.path, f))])
        self.dropdown.options = [x for x in self.files]
        self.dropdown.value = value

    @property
    def render_element(self): 
        return VBox([HBox([self.dropdown, self.btn_refresh]), self.preview])

    def render(self):
        display(self.render_element)

    def _ipython_display_(self):
        self.render()
