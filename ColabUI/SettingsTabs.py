from ipywidgets import Button, Tab, VBox, Output, Accordion, IntSlider, Text
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
        self.lora_choice = LoraChoice(self.colab, self.output, lora_cache=colab.lora_cache)
        self.general_settings = self.__general_settings()

        self.tabs = Tab()
        self.tabs.children = [self.sampler_choice.render_element, 
                        self.vae_choice.render_element,
                        self.ti_choice.render_element, 
                        self.lora_choice.render_element,
                        self.general_settings]

        for i,x in enumerate(["Sampler", "VAE", "Textual Inversion", "Lora", "Other"]): 
            self.tabs.set_title(i, x)
            
        self.accordion = Accordion(children=[VBox([self.tabs, self.clear_output_button, self.output])])
        self.accordion.set_title(0, "Settings")
        self.accordion.selected_index = None
    
    def __general_settings(self):
        clip_slider = IntSlider(value=self.colab.ui.clip_skip, min=0, max=4, description="Clip Skip")
        def clip_value_changed(change):
            self.output.clear_output()
            with self.output:
                self.colab.ui.clip_skip = change.new
                print(f"Clip Skip set to {change.new}")
        clip_slider.observe(clip_value_changed, "value")

        favourite_dir = Text(placeholder="Path to folder", description="Favourites:")
        favourite_dir.value = self.colab.favourite_dir
        def favourite_value_changed(change):
            self.output.clear_output()
            with self.output:
                if (os.path.isdir(change.new)):
                    self.colab.favourite_dir = change.new
                    t.description = "Favourites:"
                    print(f"Favourites foulder changed to {change.new}")
                else:
                    t.description = highlight("Favourites:", color="red")
        favourite_dir.observe(favourite_value_changed, "value")

        return VBox([clip_slider, favourite_dir])
    
    @staticmethod
    def highlight(str_to_highlight: str, color: str = "red") -> str:
        return f"<font color='{color}'>{str_to_highlight}</font>"
    
    @property
    def render_element(self): 
        return self.accordion

    def render(self):
        display(self.render_element)

    def _ipython_display_(self):
        self.render()
