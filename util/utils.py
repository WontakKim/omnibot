import time
from typing import Union, List

import cv2
import easyocr
import numpy as np
import supervision as sv
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms.v2 import ToPILImage

from util.box_annotator import BoxAnnotator
from util.spatial_hash import SpatialHash

reader = easyocr.Reader(['ko','en'], gpu=torch.cuda.is_available())

def get_caption_model_processor(model_name, model_name_or_path="Salesforce/blip2-opt-2.7b", device=None):
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, processor = None, None

    if model_name == "blip2":
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        if device == 'cpu':
            model = Blip2ForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=model_name_or_path,
                device_map=None,
                torch_dtype=torch.float32
            )
        else:
            model = Blip2ForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=model_name_or_path,
                device_map=None,
                torch_dtype=torch.float16
            ).to(device)

    elif model_name == "florence2":
        from transformers import AutoProcessor, AutoModelForCausalLM
        processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path="microsoft/Florence-2-base",
            trust_remote_code=True
        )
        if device == 'cpu':
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16,
                trust_remote_code=True
            ).to(device)

    return {'model': model.to(device) if model else None, 'processor': processor}

def get_yolo_model(model_path):
    from ultralytics import YOLO
    # Load the model.
    model = YOLO(model_path)
    return model

@torch.inference_mode()
def gen_caption_text(
    image_source,
    bboxes,
    starting_index,
    caption_model_processor,
    prompt=None,
    batch_size=128
):
    # slice
    bboxes = bboxes[starting_index:]
    if len(bboxes) == 0:
        return []

    # Number of samples per batch, --> 128 roughly takes 4 GB of GPU memory for florence v2 model
    to_pil = ToPILImage()

    cropped_pil_images = []
    for i, coord in enumerate(bboxes):
        # crop and resize to 64x64 image
        try:
            x_min, x_max = int(coord[0] * image_source.shape[1]), int(coord[2] * image_source.shape[1])
            y_min, y_max = int(coord[1] * image_source.shape[0]), int(coord[3] * image_source.shape[0])

            cropped_image = image_source[y_min:y_max, x_min:x_max, :]
            cropped_image = cv2.resize(cropped_image, (64, 64))
            cropped_pil_images.append(to_pil(cropped_image))
        except:
            continue

    model, processor = caption_model_processor["model"], caption_model_processor["processor"]
    if not prompt:
        if 'florence' in model.config.name_or_path:
            prompt = "<CAPTION>"
        else:
            prompt = "The image shows"

    gen_texts = []
    device = model.device

    for i in range(0, len(cropped_pil_images), batch_size):
        batch_images = cropped_pil_images[i:i+batch_size]
        if model.device.type == 'cuda':
            inputs = processor(
                images=batch_images,
                text=[prompt] * len(batch_images),
                return_tensors="pt",
                do_resize=False,
            ).to(device=device, dtype=torch.float16)
        else:
            inputs = processor(
                images=batch_images,
                text=[prompt] * len(batch_images),
                return_tensors="pt",
            ).to(device=device)

        if 'florence' in model.config.name_or_path:
            gen_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=20,
                num_beams=1,
                do_sample=False
            )
        else:
            gen_ids = model.generate(
                **inputs,
                max_length=100,
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True,
                num_return_sequences=1
            )  # temperature=0.01, do_sample=True,

        gen_text = processor.batch_decode(
            gen_ids,
            skip_special_tokens=True
        )
        gen_text = [gen.strip() for gen in gen_text]
        gen_texts.extend(gen_text)

    return gen_texts

def predict_yolo(
    model,
    image,
    image_size,
    scale_image,
    box_threshold,
    iou_threshold=0.7
):
    if scale_image:
        result = model.predict(
            source=image,
            image_size=image_size,
            conf=box_threshold,
            iou=iou_threshold
        )
    else:
        result = model.predict(
            source=image,
            conf=box_threshold,
            iou=iou_threshold
        )
    boxes = result[0].boxes.xyxy
    conf = result[0].boxes.conf
    phrases = [str(i) for i in range(len(boxes))]
    return boxes, conf, phrases

def get_xywh(input):
    x, y, w, h = input[0][0], input[0][1], input[2][0] - input[0][0], input[2][1] - input[0][1]
    x, y, w, h = int(x), int(y), int(w), int(h)
    return x, y, w, h

def get_xyxy(input):
    x, y, xp, yp = input[0][0], input[0][1], input[2][0], input[2][1]
    x, y, xp, yp = int(x), int(y), int(xp), int(yp)
    return x, y, xp, yp

def int_box_area(box, w, h):
    x1, y1, x2, y2 = box
    int_box = [int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)]
    area = (int_box[2] - int_box[0]) * (int_box[3] - int_box[1])
    return area

def remove_overlap(
    iou_threshold,
    icon_bbox_elements,
    ocr_bbox_elements=None,
    cell_size=0.04
):
    """
    icon_bboxes format: [{ 'type': 'icon', 'bbox':[x,y], 'interactivity': True, 'content': str }, ...]
    ocr_bboxes format: [{ 'type': 'text', 'bbox':[x,y], 'interactivity': False, 'content': str }, ...]
    """
    assert isinstance(icon_bbox_elements, List)
    assert ocr_bbox_elements is None or isinstance(ocr_bbox_elements, List)

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def intersection_area(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    def iou(box1, box2):
        intersection = intersection_area(box1, box2)
        union = box_area(box1) + box_area(box2) - intersection + 1e-6
        if box_area(box1) > 0 and box_area(box2) > 0:
            ratio1 = intersection / box_area(box1)
            ratio2 = intersection / box_area(box2)
        else:
            ratio1, ratio2 = 0, 0
        return max(intersection / union, ratio1, ratio2)

    def is_inside(box1, box2):
        # return box1[0] >= box2[0] and box1[1] >= box2[1] and box1[2] <= box2[2] and box1[3] <= box2[3]
        intersection = intersection_area(box1, box2)
        ratio = intersection / box_area(box1)
        return ratio > 0.80

    final_bbox_elements = list(ocr_bbox_elements or [])
    removed_ocr_indices = set()

    icon_spatial_hash = SpatialHash(cell_size)
    for i, element in enumerate(icon_bbox_elements):
        icon_spatial_hash.insert(i, element["bbox"])

    ocr_spatial_hash = SpatialHash(cell_size)
    for i, element in enumerate(ocr_bbox_elements or []):
        ocr_spatial_hash.insert(i, element["bbox"])

    for i, icon_element in enumerate(icon_bbox_elements):
        icon_bbox = icon_element["bbox"]
        is_valid_bbox = True

        potential_icon_overlaps = icon_spatial_hash.query_candidates(icon_bbox)

        # check overlap bounding box
        for candidate in potential_icon_overlaps:
            if i == candidate or candidate >= len(icon_bbox_elements):
                continue

            candidate_icon_element = icon_bbox_elements[candidate]
            candidate_icon_bbox = candidate_icon_element["bbox"]

            if iou(icon_bbox, candidate_icon_bbox) > iou_threshold and box_area(icon_bbox) > box_area(candidate_icon_bbox):
                is_valid_bbox = False
                break

        if is_valid_bbox:
            if ocr_bbox_elements:
                box_added = False
                ocr_labels = []

                potential_ocr_overlaps = ocr_spatial_hash.query_candidates(icon_bbox)

                for candidate in potential_ocr_overlaps:
                    if candidate >= len(ocr_bbox_elements):
                        continue

                    candidate_ocr_element = ocr_bbox_elements[candidate]
                    candidate_ocr_bbox = candidate_ocr_element["bbox"]

                    if is_inside(candidate_ocr_bbox, icon_bbox): # ocr inside icon
                        try:
                            ocr_label = candidate_ocr_element["content"]
                            ocr_labels.append(ocr_label)
                            removed_ocr_indices.add(candidate)
                        except:
                            continue
                    elif is_inside(icon_bbox, candidate_ocr_bbox): # icon inside ocr
                        box_added = True
                        break

                if not box_added:
                    ocr_content = ' '.join(ocr_labels) if ocr_labels else None
                    final_bbox_elements.append({
                        'type': 'icon',
                        'bbox': icon_bbox,
                        'interactivity': True,
                        'content': ocr_content,
                        'source': 'box_yolo_content_ocr' if ocr_content else 'box_yolo_content_yolo'
                    })
            else:
                final_bbox_elements.append(icon_element)

    for i in sorted(removed_ocr_indices, reverse=True):
        if i >= len(final_bbox_elements):
            continue
        del final_bbox_elements[i]

    return final_bbox_elements

def get_som_labeled_bbox(
    image_source: Union[str, Image.Image],
    image_size=None,
    scale_image=False,
    model=None,
    box_threshold=0.1,
    ocr_bboxes=None, # xyxy
    ocr_texts=None,
    iou_threshold=0.9,
    spatial_cell_size=0.05,
    use_local_semantics=True,
    caption_model_processor=None,
    prompt=None,
    batch_size=128
):
    if isinstance(image_source, str):
        image_source = Image.open(image_source)

    image_source = image_source.convert("RGB")
    w, h = image_source.size
    if not image_size:
        image_size = (h, w)

    icon_bboxes, logits, phrases = predict_yolo(
        model=model,
        image=image_source,
        image_size=image_size,
        scale_image=scale_image,
        box_threshold=box_threshold,
        iou_threshold=0.1
    ) # icon bbox contains xyxy
    # normalize
    icon_bboxes = icon_bboxes / torch.Tensor([w, h, w, h]).to(icon_bboxes.device)
    image_source = np.asarray(image_source)

    # annotate the image with labels
    if ocr_bboxes:
        ocr_bboxes = torch.Tensor(ocr_bboxes) / torch.Tensor([w, h, w, h])
        ocr_bboxes = ocr_bboxes.tolist()
    else:
        print("no ocr bbox")
        ocr_bboxes = []
    if ocr_texts is None:
        ocr_texts = []

    ocr_bbox_elements = [{
        "type": "text",
        "bbox": bbox,
        "interactivity": False,
        "content": text,
        "source": "box_ocr_content_ocr"
    } for bbox, text in zip(ocr_bboxes, ocr_texts) if int_box_area(bbox, w, h) > 0]
    icon_bbox_elements = [{
        "type": "icon",
        "bbox": bbox,
        "interactivity": True,
        "content": None
    } for bbox in icon_bboxes.tolist() if int_box_area(bbox, w, h) > 0]

    start_time = time.time()
    bbox_elements = remove_overlap(
        icon_bbox_elements=icon_bbox_elements,
        ocr_bbox_elements=ocr_bbox_elements,
        iou_threshold=iou_threshold,
        cell_size=spatial_cell_size
    )
    print(f"remove overlap: {time.time() - start_time} secs")

    # sort the sorted_bbox_elements so that the one with 'content': None is at the end, and get the index of the first 'content': None
    sorted_bbox_elements = sorted(bbox_elements, key=lambda item: item["content"] is None)
    # get the index of the first 'content': None
    sorted_bboxes = torch.Tensor([element["bbox"] for element in sorted_bbox_elements])
    starting_index = next((i for i, box in enumerate(sorted_bbox_elements) if box["content"] is None), len(sorted_bbox_elements))
    print(f"len(bbox): {len(sorted_bboxes)}, starting_index: {starting_index}")

    # get parsed icon local semantics
    if use_local_semantics:
        caption_model = caption_model_processor["model"]

        start_time = time.time()
        if 'phi3_v' in caption_model.config.model_type:
            print("phi3_v")
            gen_caption_texts = []
        else:
            gen_caption_texts = gen_caption_text(
                image_source=image_source,
                bboxes=sorted_bboxes,
                starting_index=starting_index,
                caption_model_processor=caption_model_processor,
                prompt=prompt,
                batch_size=batch_size
            )
        print(f"gen_caption_texts: {time.time()-start_time} takes")

        # fill the gen caption text into elements
        for i in range(starting_index, len(sorted_bbox_elements)):
            element = sorted_bbox_elements[i]
            if element["content"] is None:
                element["content"] = gen_caption_texts[i - starting_index]

    return sorted_bbox_elements

def get_annotated_image_frame(
    image_source: Union[str, Image.Image],
    bbox_elements,
    draw_bbox_config=None,
):
    if isinstance(image_source, str):
        image_source = Image.open(image_source)

    image_source = image_source.convert("RGB")
    image_source = np.asarray(image_source)
    h, w, _ = image_source.shape


    bboxes = np.array([element["bbox"] for element in bbox_elements])
    bboxes = bboxes * np.array([w, h, w, h])
    detections = sv.Detections(xyxy=bboxes)

    labels = [str(i) for i in range(len(bbox_elements))]

    box_annotator = BoxAnnotator(**draw_bbox_config)
    annotated_frame = image_source.copy()
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels,
        image_size=(w, h)
    )
    return annotated_frame

def create_image(image_source, coords, w, h, output):
    opencv_image = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
    for item in coords:
        coord = item["bbox"]
        x = int(coord[0] * w)
        y = int(coord[1] * h)
        x2 = int(coord[2] * w)
        y2 = int(coord[3] * h)
        cv2.rectangle(
            opencv_image,
            (min(x, x2), min(y, y2)),
            (max(x, x2), max(y, y2)),
            (0, 255, 0),
            2
        )
        cv2.rectangle(opencv_image, (min(x, x2), min(y, y2)), (max(x, x2), max(y, y2)), (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))
    plt.savefig(output)

def check_ocr_box(
    image_source: Union[str, Image.Image],
    output_bbox_format="xywh",
    easyocr_args=None
):
    if isinstance(image_source, str):
        image_source = Image.open(image_source)

    if image_source.mode == "RGBA":
        # Convert RGBA to RGB to avoid alpha channel issues
        image_source = image_source.convert("RGB")

    image_np = np.array(image_source)

    if easyocr_args is None:
        easyocr_args = {}

    result = reader.readtext(image_np, **easyocr_args)
    coords, texts = zip(*[(item[0], item[1]) for item in result]) if result else ([], [])

    format_func = get_xyxy if output_bbox_format == "xyxy" else get_xywh
    bboxes = [format_func(coord) for coord in coords]

    return texts, bboxes