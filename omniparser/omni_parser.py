from typing import Union, List, Dict

import torch
from PIL import Image

from omniparser.adapter.object_detection.yolo_adapter import YoloAdapter
from omniparser.adapter.ocr.easy_ocr_adapter import EasyOCRAdapter
from omniparser.adapter.vision_language.florence_adapter import FlorenceAdapter
from omniparser.data.som import SOM
from omniparser.util.box_utils import remove_overlap, box_inclusion
from omniparser.util.image_utils import create_annotated_image, get_cropped_image


class OmniParser:
    def __init__(self, config: Dict):
        self.config = config
        self.device = 'cuda' if config['device'] == 'cuda' and torch.cuda.is_available() else 'cpu'

        self.ocr_adapter = EasyOCRAdapter(['en', 'ko'], self.device)
        self.object_detection_adapter = YoloAdapter('../weights/icon_detect/model.pt', self.device)
        self.vision_language_adapter = FlorenceAdapter('../weights/icon_caption_florence', self.device)

    def parse(self, image: Union[str, Image.Image]) -> List[SOM]:
        if isinstance(image, str):
            image = Image.open(image)
        return [element.safe() for element in self.get_som_labeled_element(image)]

    def get_som_labeled_element(self, image: Image.Image, use_local_semantics: bool=True) -> List[SOM]:
        # remove alpha channel
        image = image.convert('RGB')
        w, h = image.size

        # extract icon and text
        ocr_result = self.ocr_adapter.extract_text(image)
        obj_result = self.object_detection_adapter.predict(image, iou_threshold=0.1)

        # normalize
        whwh = torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device)
        ocr_bboxes = ocr_result.bboxes / whwh if len(ocr_result.bboxes) > 0 else []
        ocr_texts = ocr_result.texts if len(ocr_result.texts) > 0 else []
        obj_bboxes = obj_result.bboxes / whwh if len(obj_result.bboxes) > 0 else []

        # custom nms
        obj_keeps = remove_overlap(obj_bboxes)
        obj_bboxes = obj_bboxes[obj_keeps]

        # create labeled elements
        labeled_elements = self._create_labeled_element(obj_bboxes, ocr_bboxes, ocr_texts)

        if use_local_semantics:
            labeled_elements = sorted(labeled_elements, key=lambda x: x.content is None)
            starting_index = next((i for i, element in enumerate(labeled_elements) if element.content is None), len(labeled_elements))

            # extract no content element
            no_labeled_elements = labeled_elements[starting_index:]
            if len(no_labeled_elements) > 0:
                bboxes = torch.stack([element.bbox for element in no_labeled_elements])
                # create caption text
                gen_texts = self._gen_caption_text(image, bboxes)
                for i, element in enumerate(no_labeled_elements):
                    element.content = gen_texts[i]

        return labeled_elements

    def _gen_caption_text(self, image: Image.Image, bboxes) -> List[str]:
        cropped_images = get_cropped_image(image, bboxes, (64, 64))
        return self.vision_language_adapter.gen_text(cropped_images)

    @staticmethod
    def _create_labeled_element(obj_bboxes, ocr_bboxes, ocr_texts) -> List[SOM]:
        assert len(ocr_bboxes) == len(ocr_texts)
        inclusions = box_inclusion(ocr_bboxes, obj_bboxes)
        ocr_match_indices, obj_match_indices = torch.where(inclusions)
        use_ocr_indices = torch.unique(ocr_match_indices)

        labeled_elements = []

        for i, bbox in enumerate(obj_bboxes):
            ocr_mask = obj_match_indices == i
            if torch.any(ocr_mask):
                filtered = ocr_match_indices[ocr_mask]
                texts = [ocr_texts[j] for j in filtered]
                content = ' '.join(texts)
            else:
                content = None

            som = SOM(
                type='icon',
                bbox=bbox,
                interactivity=True,
                content=content,
                source='box_yolo_content_ocr' if content else 'box_yolo_content_yolo'
            )
            labeled_elements.append(som)

        for i, bbox in enumerate(ocr_bboxes):
            if i in use_ocr_indices:
                continue

            som = SOM(
                type='text',
                bbox=bbox,
                interactivity=False,
                content=ocr_texts[i],
                source='box_ocr_content_ocr'
            )
            labeled_elements.append(som)

        return labeled_elements

if __name__ == '__main__':
    image = Image.open('/users/wontak/desktop/test3.png')

    omniparser = OmniParser({'device': 'cpu'})
    labeled_elements = omniparser.parse(image)

    annotated_image = create_annotated_image(image, labeled_elements)
    annotated_image.show()
    # buffered = io.BytesIO()
    # annotated_image.save(buffered, format='JPEG')
    # encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')