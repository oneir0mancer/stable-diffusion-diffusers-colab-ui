from IPython.display import display, Javascript
from ipywidgets import Dropdown, HTML, HBox, Layout, Text, Button, Accordion, Output
import json

class ArtistIndex:
    __clipboard: str

    def __init__(self, ui, index_path = "artist_index.json"):
        with open(index_path) as f:
            self.data = json.load(f)

        self.ui = ui
        self.example_link = HTML()

        self.__setup_artist_dropdown()
        self.__setup_buttons()
        self.__setup_foldout()

    def render(self):
        """Display ui"""
        self.__set_link_from_item(self.artist_dropdown.value)
        display(self.foldout, self.output)

    def __setup_artist_dropdown(self):
        self.artist_dropdown = Dropdown(
            options=[x for x in self.data],
            description="Artist:",
        )
        self.artist_dropdown.description_tooltip = "Choose artist flair tag to add"
        def dropdown_eventhandler(change):
            self.__set_link_from_item(change.new)
        self.artist_dropdown.observe(dropdown_eventhandler, names='value')

    def __setup_buttons(self):
        self.add_button = Button(description="Add", layout=Layout(width='80px'))
        def on_add_clicked(b):
            self.ui.positive_prompt.value += f", by {self.artist_dropdown.value}"
        self.add_button.on_click(on_add_clicked)

        self.copy_button = Button(description='Copy', tooltip="Copy to clipboard", layout=Layout(width='80px'))
        self.output = Output()
        def copy_event_handler(b):
            with self.output:
                text = self.__clipboard
                display(Javascript(f"navigator.clipboard.writeText('{text}');")) 
        self.copy_button.on_click(copy_event_handler)

    def __setup_foldout(self):
        hbox = HBox([self.artist_dropdown, self.add_button, self.copy_button, self.example_link])
        self.foldout = Accordion(children=[hbox])
        self.foldout.set_title(0, "Flair tags")
        self.foldout.selected_index = None

    def __set_link_from_item(self, item):
        self.__clipboard = item
        key = item.lower().replace(" ", "-")
        self.example_link.value = f"Example: <a href=https://midlibrary.io/styles/{key}>{item}</a>"

    def _ipython_display_(self):
        self.render()
