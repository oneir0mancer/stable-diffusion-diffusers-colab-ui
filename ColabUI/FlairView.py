import os
import typing
from IPython.display import display, Javascript
from ipywidgets import Output, Text, Layout, Button, HBox, VBox, Dropdown, Accordion, Checkbox
from ..utils.empty_output import EmptyOutput
from ..utils.downloader import download_ckpt
from ..utils.ui import SwitchButton

class FlairItem:
    __data : list[str]

    def __init__(self, ui, out:Output = None, index : int = -1):
        self.ui = ui
        if out is None: out = EmptyOutput()
        self.out = out
        self.clipboard_output = Output()
        self.index = index

        self.url_text = Text(placeholder="Path or url to load", layout=Layout(width="40%"))
        if index >= 0: self.url_text.description = f"{index + 1}:"
        self.url_text.description_tooltip = "Path to a file with a list of items, or a url"

        self.load_button = Button(description="Load", layout=Layout(width='50px'))
        def load_btn_click(b):
            self.out.clear_output()
            with self.out: 
                self.load_data(self.url_text.value)
                if self.__data is not None and len(self.__data) > 0:
                    self.switch_to_dropdown()
        self.load_button.on_click(load_btn_click)

        self.load_view = HBox([self.url_text, self.load_button])

        self.flair_dropdown = Dropdown(layout=Layout(width="40%"))
        if index >= 0: self.flair_dropdown.description = f"{index + 1}:"
        self.add_button = Button(description="Add", layout=Layout(width='80px'))
        def add_btn_click(b):
            self.ui.positive_prompt.value += f", by {self.flair_dropdown.value}"
        self.add_button.on_click(add_btn_click)

        self.copy_button = Button(description="Copy", tooltip="Copy to clipboard", layout=Layout(width='80px'))
        def copy_event_handler(b):
            with self.clipboard_output:
                text = self.flair_dropdown.value
                display(Javascript(f"navigator.clipboard.writeText('{text}');")) 
        self.copy_button.on_click(copy_event_handler)

        self.preprocess_btn = SwitchButton("Preprocess", "Preprocess", Layout(width='100px'), 
                                           self.preprocessor_on, self.preprocessor_off)
        self.preprocess_btn.button.tooltip = f"Automatically subsitute {{{index + 1}}} for an element of this list in prompt"
        self.subsitute_index = 0
        self.subsitute_to_random = Checkbox(value=False, description="Random", indent=False)

        self.dropdown_view = HBox([self.flair_dropdown, self.add_button, self.copy_button, 
                                   self.preprocess_btn.render_element, self.subsitute_to_random])
        self.dropdown_view.layout.visibility = "hidden"
        self.dropdown_view.layout.height = "0px"

    def switch_to_dropdown(self):
        self.load_view.layout.visibility = "hidden"
        self.dropdown_view.layout.visibility = "visible"
        self.dropdown_view.layout.height = self.load_view.layout.height
        self.load_view.layout.height = "0px"
        self.flair_dropdown.options = self.__data
        self.flair_dropdown.description_tooltip = self.url_text.value

    @property
    def render_element(self): 
        return VBox([self.load_view, self.dropdown_view, self.clipboard_output])
    
    def render(self):
        display(self.render_element)

    def _ipython_display_(self):
        self.render()

    def preprocessor_on(self):
        self.ui.prompt_preprocessors.append(self)
    
    def preprocessor_off(self):
        self.ui.prompt_preprocessors.remove(self)

    def process(self, prompt:str):
        #TODO guard index
        item = self.__data[self.subsitute_index]
        self.step_subsitute_index()
        return prompt.replace(f"{{{self.index + 1}}}", item)

    def set_subsitute_index(self, index:int = 0):
        self.subsitute_index = index
    
    def step_subsitute_index(self):
        if self.subsitute_to_random.value:
            self.subsitute_index = random.randint(0, len(self.__data))
        else:
            self.subsitute_index += 1
            if self.subsitute_index >= len(self.__data): self.subsitute_index = 0

    def load_data(self, path:str):
        if os.path.isfile(path):
            self.__data = self.read_file(path)
        else:
            f = download_ckpt(path)
            self.__data = self.read_file(f)

    @staticmethod
    def read_file(path:str):
        with open(path) as f: data = f.readlines()
        data = [x.strip('\n') for x in data]
        return data

class FlairView:
    __items : list[FlairItem]

    def __init__(self, ui):
        self.ui = ui
        self.vbox = VBox([])

        self.output = Output(layout={'border': '1px solid black'})
        self.clear_output_button = Button(description="Clear Log")
        def on_clear_clicked(b):
            self.output.clear_output()
        self.clear_output_button.on_click(on_clear_clicked)
        right_align_layout = Layout(display='flex', flex_flow='column', align_items='flex-end', width='99%')
        
        self.add_button = Button(description="Add", layout=Layout(width='80px'), button_style='info')
        def add_btn_click(b):
            f = FlairItem(self.ui, self.output, len(self.vbox.children))
            self.vbox.children=tuple(list(self.vbox.children) + [f.render_element]) 
        self.add_button.on_click(add_btn_click)

        self.foldout = Accordion(children=[VBox([self.vbox, self.add_button, self.output, 
                                                 HBox(children=[self.clear_output_button],layout=right_align_layout)])])
        self.foldout.set_title(0, "Flair tags")
        self.foldout.selected_index = None

    @property
    def render_element(self): 
        return self.foldout

    def render(self):
        display(self.render_element)

    def _ipython_display_(self):
        self.render()
