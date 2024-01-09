import markdown
from ipywidgets import HTML

def SpoilerLabel(warning:str, spoiler_text:str):
    md = f"""
<details>
    <summary>{warning}</summary>
    {spoiler_text}
</details>
"""
    return HTML(markdown.markdown(md))

def MarkdownLabel(text:str):
    return HTML(markdown.markdown(text))
