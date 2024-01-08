from ipywidgets import Button, Tab, VBox, Output
from .SamplerChoice import SamplerChoice   #TODO
from .VaeChoice import VaeChoice
from .TextualInversionChoice import TextualInversionChoice
from .LoraChoice import LoraChoice

class SettingsTabs:
    #TODO config file, default import file
    def __init__(self, colab):
        self.colab = colab
        self.output = Output(layout={'border': '1px solid black'})
        self.clear_output_button = Button(description="Clear Log")
        def on_clear_clicked(b):
            self.output.clear_output()
        self.clear_output_button.on_click(on_clear_clicked)

        self.sampler_choice = SamplerChoice(self.colab, self.output)
        self.vae_choice = VaeChoice(self.colab, self.output)
        self.ti_choice = TextualInversionChoice(self.colab, self.output)
        self.lora_choice = LoraChoice(self.colab, self.output)

        self.tabs = Tab()
        self.tabs.children = [self.sampler_choice.render_element, 
                        self.vae_choice.render_element,
                        self.ti_choice.render_element, 
                        self.lora_choice.render_element]

        for i,x in enumerate(["Sampler", "VAE", "Textual Inversion", "Lora"]): 
            self.tabs.set_title(i, x)
    
    @property
    def render_element(self): 
        return VBox([self.tabs, self.clear_output_button, self.output])

    def render(self):
        display(self.render_element)

    def _ipython_display_(self):
        self.render()
