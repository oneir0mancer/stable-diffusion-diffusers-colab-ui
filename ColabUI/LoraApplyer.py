from ipywidgets import Dropdown, Button, HBox, VBox, Layout, Text, FloatSlider
from IPython.display import display
from .LoraDownloader import LoraDownloader

class LoraApplyer:
    def __init__(self, pipe, loader: LoraDownloader):
        self.pipe = pipe
        self.__cache = loader.cache
        self.__applier_loras = dict()
        self.vbox = VBox()
        self.is_fused = False

        self.__setup_dropdown()
        #TODO move this to caller
        loader.on_load_event.clear_callbacks()
        loader.on_load_event.add_callback(self.__update_dropdown)

        self.add_button = Button(description="Add", layout=Layout(width='50px'))
        def add_lora(b):
            self.__add_lora(self.dropdown.value, 1)
        self.add_button.on_click(add_lora)

    def fuse_lora(self):
        if self.is_fused: return
        self.is_fused = True
        self.__apply_adapters()
        self.pipe.fuse_lora()
        #TODO disable button interactivity

    def unfuse_lora(self):
        if not self.is_fused: return
        self.is_fused = False
        self.pipe.unfuse_lora()

    def __setup_dropdown(self):
        self.dropdown = Dropdown(
            options=[x for x in self.__cache],
            description="Lora:",
        )
        self.dropdown.description_tooltip = "Choose lora to load"

    def __update_dropdown(self, new_adapter):
        self.dropdown.options = [x for x in self.__cache]

    def __add_lora(self, adapter: str, scale: float):
        if adapter in self.__applier_loras: return
        self.__applier_loras[adapter] = scale
        self.__rerender_items()
        self.__apply_adapters()

    def __rerender_items(self):
        w = []
        for adapter, scale in self.__applier_loras.items():
            slider = self.__add_lora_scale_slider(adapter, scale)
            remove_btn = self.__add_lora_remove_button(adapter)
            w.append(HBox([slider, remove_btn]))

        self.vbox.children = w

    def __add_lora_scale_slider(self, adapter, scale):
        slider = FloatSlider(min=0, max=1.5, value=scale, step=0.05, description=adapter)
        def value_changed(change):
            self.__applier_loras[adapter] = change.new
            self.__apply_adapters()
        slider.observe(value_changed, 'value')
        return slider

    def __add_lora_remove_button(self, adapter):
        remove_btn = Button(description="X", layout=Layout(width='40px'))
        def on_button(b):
            del self.__applier_loras[adapter]
            self.__rerender_items()
            self.__apply_adapters()
        remove_btn.on_click(on_button)
        return remove_btn

    def __apply_adapters(self):
        if self.is_fused: return
        self.pipe.enable_lora()     #TODO use separate button
        self.pipe.set_adapters([x for x in self.__applier_loras.keys()], 
                          adapter_weights=[x for x in self.__applier_loras.values()])

    @property
    def render_element(self): 
        return VBox([HBox([self.dropdown, self.add_button]), self.vbox])

    def render(self):
        display(self.render_element)

    def _ipython_display_(self):
        self.render()
