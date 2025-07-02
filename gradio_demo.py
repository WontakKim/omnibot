import base64
import io
import json
import time
import warnings
from typing import Optional

import gradio as gr
from PIL import Image

from util import check_ocr_box, get_yolo_model, get_caption_model_processor, \
    get_som_labeled_bbox, get_annotated_image_frame

warnings.simplefilter('always', SyntaxWarning)

yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')
caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence")

def process(
    image_input,
    box_threshold,
    iou_threshold,
    cell_size,
    icon_detect_image_size
):
    start_time = time.time()

    box_overlay_ratio = image_input.size[0] / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }

    ocr_texts, ocr_bboxes = check_ocr_box(
        image_source=image_input,
        output_bbox_format='xyxy',
        easyocr_args={'paragraph': False, 'text_threshold': 0.9}
    )
    som_label_bbox = get_som_labeled_bbox(
        image_source=image_input,
        image_size=icon_detect_image_size,
        model=yolo_model,
        box_threshold=box_threshold,
        iou_threshold=iou_threshold,
        spatial_cell_size=cell_size,
        ocr_bboxes=ocr_bboxes,
        ocr_texts=ocr_texts,
        caption_model_processor=caption_model_processor,
    )

    print(f'laps: {time.time() - start_time} secs')

    # draw boxes
    frame = get_annotated_image_frame(
        image_source=image_input,
        bbox_elements=som_label_bbox,
        draw_bbox_config=draw_bbox_config,
    )

    pil_image = Image.fromarray(frame)
    # buffered = io.BytesIO()
    # pil_image.save(buffered, format='PNG')
    # encoded_image = base64.b64encode(buffered.getvalue()).decode('ascii')
    # image = Image.open(io.BytesIO(base64.b64decode(encoded_image)))
    return pil_image, str(json.dumps(som_label_bbox))

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
            # set the threshold for removing the bounding boxes with low confidence, default is 0.05
            box_threshold_component = gr.Slider(label="Box Threshold", minimum=0.01, maximum=1.0, step=0.01, value=0.05)
            # set the threshold for removing the bounding boxes with large overlap, default is 0.1
            iou_threshold_component = gr.Slider(label='IOU Threshold', minimum=0.01, maximum=1.0, step=0.01, value=0.1)
            # set the cell size for removing the bounding boxes with large overlap, default is 0.1
            cell_size_component = gr.Slider(label='Spatial Hash Cell Size', minimum=0.001, maximum=1.0, step=0.005, value=0.04)
            icon_detect_image_size_component = gr.Slider(label='Icon Detect Image Size', minimum=640, maximum=1920, step=32, value=640)
            submit_button_component = gr.Button(value='Submit', variant='primary')
        with gr.Column():
            image_output_component = gr.Image(type='pil', label='Image Output')
            text_output_component = gr.Textbox(label='Parsed screen elements', placeholder='Text Output')

    submit_button_component.click(
        fn=process,
        inputs=[
            image_input_component,
            box_threshold_component,
            iou_threshold_component,
            cell_size_component,
            icon_detect_image_size_component
        ],
        outputs=[image_output_component, text_output_component]
    )

if __name__ == "__main__":
    demo.launch()