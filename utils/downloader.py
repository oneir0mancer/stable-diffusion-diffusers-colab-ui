import requests
import werkzeug
import os
from tqdm.notebook import tqdm

def download_ckpt(ckpt_url):
    """
    download the checkpoint & return file name to be used in diffusers pipeline
    additional actions:
    - try to get file name from header or url
    - download only if file not exist
    - download with progress bar
    """
    HEADERS = {"User-Agent": "github/oneir0mancer/stable-diffusion-diffusers-colab-ui"}
    with requests.get(ckpt_url, headers=HEADERS, stream=True) as resp:

        # get file name
        MISSING_FILENAME = "missing_name"
        if content_disposition := resp.headers.get("Content-Disposition"):
            param, options = werkzeug.http.parse_options_header(content_disposition)
            if param == "attachment":
                filename = options.get("filename", MISSING_FILENAME)
            else:
                filename = MISSING_FILENAME
        else:
            filename = os.path.basename(ckpt_url)
            fileext = os.path.splitext(filename)[-1]
            if fileext == "":
                filename = MISSING_FILENAME

        # download file
        if not os.path.exists(filename):
            TOTAL_SIZE = int(resp.headers.get("Content-Length", 0))
            CHUNK_SIZE = 50 * 1024**2  # 50 MiB or more as colab has high speed internet
            with open(filename, mode="wb") as file, tqdm(total=TOTAL_SIZE, desc="download checkpoint", unit="iB", unit_scale=True) as progress_bar:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    progress_bar.update(len(chunk))
                    file.write(chunk)
    return filename
