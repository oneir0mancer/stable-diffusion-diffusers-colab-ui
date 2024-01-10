from IPython.display import display
from ipywidgets import Dropdown, Button, HBox, VBox, Layout, Text, FloatSlider, Output
from ..utils.empty_output import EmptyOutput

class LoraApplyer:
    def __init__(self, colab, out:Output = None, cache = None):
        self.colab = colab
        if out is None: out = EmptyOutput()
        self.out = out
        self.__cache = cache
        self.__applier_loras = dict()
        self.vbox = VBox()
        self.is_fused = False

        self.__setup_dropdown()

        self.add_button = Button(description="Add", layout=Layout(width='50px'))
        def add_lora(b):
            self.out.clear_output()
            with out: self.__add_lora(self.dropdown.value, 1)
        self.add_button.on_click(add_lora)
        
        self.fuse_button = Button(description="Fuse", layout=Layout(width='75px'), button_style='success')
        def on_fuse_btn(b):
            self.out.clear_output()
            with out: self.fuse_lora()
        self.fuse_button.on_click(on_fuse_btn)
        self.fuse_button.tooltip = "Merge loras into the model weights to speed-up inference"
        
        self.unfuse_button = Button(description="Unfuse", layout=Layout(width='75px'), button_style='danger')
        def on_unfuse_btn(b):
            self.out.clear_output()
            with out: self.unfuse_lora()
        self.unfuse_button.on_click(on_unfuse_btn)
        self.unfuse_button.tooltip = "Return model weights to the original state"
        
        self.__set_ui_fuse_state(False)

    def fuse_lora(self):
        if self.is_fused: return
        self.is_fused = True
        self.__set_ui_fuse_state(True)
        self.__apply_adapters()
        self.colab.pipe.fuse_lora()

    def unfuse_lora(self):
        if not self.is_fused: return
        self.is_fused = False
        self.__set_ui_fuse_state(False)
        self.colab.pipe.unfuse_lora()

    def __setup_dropdown(self):
        self.dropdown = Dropdown(
            options=[x for x in self.__cache],
            description="Lora:",
        )
        self.dropdown.description_tooltip = "Choose lora to load"

    def update_dropdown(self, new_adapter):
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
        self.colab.pipe.enable_lora()     #TODO use separate button
        self.colab.pipe.set_adapters([x for x in self.__applier_loras.keys()], 
                          adapter_weights=[x for x in self.__applier_loras.values()])

    def __set_ui_fuse_state(self, value):
        self.add_button.disabled = value
        self.fuse_button.disabled = value
        self.unfuse_button.disabled = not value
        self.fuse_button.layout.width = "50px" if value else "75px"
        self.unfuse_button.layout.width = "50px" if not value else "75px"
        #self.fuse_button.layout.visibility = "hidden" if value else "visible"
        #self.unfuse_button.layout.visibility = "visible" if value else "hidden"

    @property
    def render_element(self): 
        return VBox([HBox([self.dropdown, self.add_button, self.fuse_button, self.unfuse_button]), self.vbox])

    def render(self):
        display(self.render_element)

    def _ipython_display_(self):
        self.render()
