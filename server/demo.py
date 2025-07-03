import argparse
import os
import sys
import warnings

import gradio as gr

from omniparser.omni_parser import OmniParser
from omniparser.util.image_utils import create_annotated_image

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

warnings.simplefilter('always', SyntaxWarning)

def parse_arguments():
    parser = argparse.ArgumentParser(description='omniparser api')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host for the API')
    parser.add_argument('--port', type=int, default=9000, help='Port for the API')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model')
    arguments = parser.parse_args()
    return arguments

args = parse_arguments()
config = vars(args)

omniparser = OmniParser(config)

def process(image_input):
    labeled_elements = omniparser.parse(image_input)
    annotated_image = create_annotated_image(image_input, labeled_elements)
    annotated_image.convert('RGBA')
    return annotated_image, str(labeled_elements)

with gr.Blocks() as demo:
    state = gr.State({})

    with gr.Row():
        model = gr.Dropdown(
            label="Model",
            choices=["gpt-4o"]
       )

    with gr.Row():
        with gr.Column():
            image_input_component = gr.Image(type="pil", label="Upload image")
            submit_button_component = gr.Button(value='Submit', variant='primary')
        with gr.Column():
            image_output_component = gr.Image(type='pil', label='Image Output')
            text_output_component = gr.Textbox(label='Parsed screen elements', placeholder='Text Output')

    submit_button_component.click(
        fn=process,
        inputs=[
            image_input_component,
        ],
        outputs=[image_output_component, text_output_component]
    )

if __name__ == "__main__":
    demo.launch()