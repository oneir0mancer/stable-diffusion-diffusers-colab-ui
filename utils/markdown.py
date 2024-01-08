import markdown
from ipywidgets import HTML

class SpoilerLabel:
    def __init__(self, warning, text):
        self.text = f"""
<details>
    <summary>{warning}</summary>
    {text}
</details>
"""

    @property
    def render_element(self): 
        return HTML(markdown.markdown(self.text))

    def render(self):
        display(self.render_element)

    def _ipython_display_(self):
        self.render()
