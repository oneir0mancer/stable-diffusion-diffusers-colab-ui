from PIL import Image
from PIL.PngImagePlugin import PngInfo

def load_image_metadata(filepath: str):
    image = Image.open(filepath)
    prompt, etc = image.text["Data"].split("\nNegative: ")
    prompt = prompt.replace("\nPrompt: ", "")
    negative_prompt, etc = etc.split("\nCGF: ")
    # TODO parse other metadata into dict perhabs
    return prompt, negative_prompt

def save_image_with_metadata(image, path, metadata_provider, additional_data = ""):
    meta = PngInfo()
    meta.add_text("Data", metadata_provider.metadata + additional_data)
    image.save(path, pnginfo=meta)
