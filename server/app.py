import argparse
import io
import os
import sys
import time

import uvicorn
from PIL import Image
from fastapi import FastAPI, UploadFile

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from omniparser.omni_parser import OmniParser


def parse_arguments():
    parser = argparse.ArgumentParser(description='omniparser api')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host for the API')
    parser.add_argument('--port', type=int, default=30005, help='Port for the API')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model')
    arguments = parser.parse_args()
    return arguments

args = parse_arguments()
config = vars(args)

app = FastAPI()
omniparser = OmniParser(config)


@app.post("/parse")
async def parse(file: UploadFile):
    print('start parsing...')
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    w, h = image.size

    start = time.time()
    labeled_elements = omniparser.parse(image)
    labeled_elements = [{
        'type': element.type,
        'content': element.content,
        'bbox': element.bbox,
        'interactivity': element.interactivity,
        'source': element.source
    } for element in labeled_elements]
    latency = time.time() - start
    print('time:', latency)
    return {
        'content': labeled_elements,
        'width': w,
        'height': h,
        'latency': latency
    }

@app.get("/probe")
async def probe():
    return {'status': 'ok'}

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)