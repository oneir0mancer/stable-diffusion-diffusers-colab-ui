from PIL import Image

def load_image_metadata(filepath: str):
    image = Image.open(filepath)
    prompt, etc = targetImage.text["Data"].split("\nNegative: ")
    prompt = prompt.replace("\nPrompt: ", "")
    negative_prompt, etc = etc.split("\nCGF: ")
    # TODO parse other metadata into dict perhabs
    return prompt, negative_prompt
