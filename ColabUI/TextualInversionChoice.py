import os
from ipywidgets import Text, Button, Layout, VBox, Output
from ..utils.empty_output import EmptyOutput
from ..utils.markdown import SpoilerLabel

class TextualInversionChoice:
    def __init__(self, colab, out:Output = None, default_path:str = "/content/embeddings/"):
        self.colab = colab
        if out is None: out = EmptyOutput()
        self.out = out

        self.tooltip_label = SpoilerLabel("Tooltip", "Paste a path to Textual Inversion .pt file, or path to a folder containig them.")
        self.path = Text(description="TI:", placeholder='Path to file or folder...', layout=Layout(width="50%"))
        self.path.description_tooltip = "Path to a Textual Inversion file or root folder"
        if default_path is not None: self.path.value = default_path

        self.load_button = Button(description="Load", button_style='success')
        def on_load_clicked(b):
            self.out.clear_output()
            with self.out: 
                self.load(self.colab.pipe)
        self.load_button.on_click(on_load_clicked)

    def load(self, pipe):
        if self.path.value == "": 
            raise ValueError("Text field shouldn't be empty")
        
        if os.path.isfile(self.path.value):
            dir, filename = os.path.split(self.path.value)
            self.__load_textual_inversion(self.colab.pipe, dir, filename)
        elif os.path.isdir(self.path.value):
            self.__load_textual_inversions_from_folder(self.colab.pipe, self.path.value)

    #TODO load from file: https://huggingface.co/docs/diffusers/api/loaders/textual_inversion
    def __load_textual_inversion(self, pipe, path: str, filename: str):
        self.colab.pipe.load_textual_inversion(path, weight_name=filename)
        print(f"<{os.path.splitext(filename)[0]}>")

    def __load_textual_inversions_from_folder(self, pipe, root_folder: str):
        n = 0
        for path, subdirs, files in os.walk(root_folder):
            for name in files:
                try:
                    if os.path.splitext(name)[1] != ".pt": continue
                    self.colab.pipe.load_textual_inversion(path, weight_name=name)
                    print(f"{path}:\t<{os.path.splitext(name)[0]}>")
                    n += 1
                except ValueError: pass
        print(f"{n} items added")

    @property
    def render_element(self): 
        return VBox([self.tooltip_label, self.path, self.load_button])

    def render(self):
        display(self.render_element)

    def _ipython_display_(self):
        self.render()
