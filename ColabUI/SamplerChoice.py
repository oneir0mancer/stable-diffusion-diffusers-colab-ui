from ipywidgets import Dropdown, Output
from diffusers import EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, UniPCMultistepScheduler
from ..utils.empty_output import EmptyOutput

class SamplerChoice:
    def __init__(self, colab, out:Output = None):
        self.colab = colab

        if out is None: out = EmptyOutput()
        self.out = out

        self.dropdown = Dropdown(
            options=["Euler A", "DPM++", "DPM++ Karras", "UniPC"],
            description='Sampler:',
        )
        
        def dropdown_eventhandler(change):
            self.out.clear_output()
            with self.out:
                self.choose_sampler(self.colab.pipe, change.new)

        self.dropdown.observe(dropdown_eventhandler, names='value')

    def choose_sampler(self, pipe, sampler_name: str):
        config = pipe.scheduler.config
        match sampler_name:
            case "Euler A": sampler = EulerAncestralDiscreteScheduler.from_config(config)
            case "DPM++": sampler = DPMSolverMultistepScheduler.from_config(config)
            case "DPM++ Karras":
                sampler = DPMSolverMultistepScheduler.from_config(config)
                sampler.use_karras_sigmas = True
            case "UniPC":
                sampler = UniPCMultistepScheduler.from_config(config)
            case _: raise NameError("Unknown sampler")
        pipe.scheduler = sampler
        print(f"Sampler '{sampler_name}' chosen")

    @property
    def render_element(self): 
        return self.dropdown

    def render(self):
        display(self.render_element)

    def _ipython_display_(self):
        self.render()
