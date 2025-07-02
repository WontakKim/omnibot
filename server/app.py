import argparse
import base64
import io
import os
import sys
import time

import uvicorn
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from omniparser.omni_parser import OmniParser


def parse_arguments():
    parser = argparse.ArgumentParser(description='omniparser api')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host for the API')
    parser.add_argument('--port', type=int, default=8000, help='Port for the API')
    arguments = parser.parse_args()
    return arguments

args = parse_arguments()
config = vars(args)

app = FastAPI()
omniparser = OmniParser(config)

class ParseRequest(BaseModel):
    base64_image: str


@app.post("/parse")
async def parse(request: ParseRequest):
    print('start parsing...')
    start = time.time()

    image_bytes = base64.b64decode(request.base64_image)
    image = Image.open(io.BytesIO(image_bytes))

    annotated_image, labeled_elements = omniparser.parse(image)
    labeled_elements = [{
        'type': element.type,
        'content': element.content,
        'bbox': element.bbox.tolist(),
        'interactivity': element.interactivity,
        'source': element.source
    } for element in labeled_elements]
    latency = time.time() - start
    print('time:', latency)
    return {"content": labeled_elements, 'latency': latency}


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)