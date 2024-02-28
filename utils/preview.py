from ipywidgets import Image, HBox, VBox, Button, Layout
from typing import List

def get_image_previews(paths: List[str], width: int, height: int):
    wgts = []
    for filepath in paths:
        preview = Image(
            value=open(filepath, "rb").read(),
            width=width,
            height=height,
        )
        btn = Button(description="Favourite", tooltip=filepath, layout=Layout(width="99%"))
        wgt = VBox([preview, btn])
        wgts.append(wgt)
    return HBox(wgts)