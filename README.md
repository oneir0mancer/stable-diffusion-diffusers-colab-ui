# Colab UI for Stable Diffusion

Txt2Img: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oneir0mancer/stable-diffusion-diffusers-colab-ui/blob/main/sd_diffusers_colab_ui.ipynb)

Img2Img: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oneir0mancer/stable-diffusion-diffusers-colab-ui/blob/main/sd_diffusers_img2img_ui.ipynb)

Colab UI for running Stable Diffusion with 🤗 diffusers library.

This repository aims to create a GUI using native Colab and IPython widgets. 
This eliminates the need for gradio WebUI, which [seems to be prohibited](https://github.com/googlecolab/colabtools/issues/3591) now on Google Colab.

![UI example](docs/ui-example.jpg)

### Features:
 - [X] UI based on IPython Widgets
 - [X] Stable diffusion 1.x, 2.x, XL
 - [X] Load models in from Huggingface, and models in Automatic1111 (ckpt/safetensors) format
 - [X] Change VAE and sampler
 - [X] Load textual inversions
 - [x] Load LoRAs
 - [x] Img2Img
 - [ ] Inpainting

### SDXL
SDXL is supported to an extent. You can load huggingface models, but loading models in A1111 format may be impossible because of RAM limitation.
Also it looks like textual inversions and loading LoRAs is not supported for SDXL.

## Kandinsky
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oneir0mancer/stable-diffusion-diffusers-colab-ui/blob/main/sd_kandinsky_colab_ui.ipynb)

Colab UI for Kandinsky txt2img and image mixing.

Original repo: https://github.com/ai-forever/Kandinsky-2
