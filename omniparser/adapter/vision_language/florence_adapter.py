from typing import List

import torch
from PIL import Image

from omniparser.adapter.vision_language.base import VisionLanguageAdapter


class FlorenceAdapter(VisionLanguageAdapter):
    def __init__(
        self,
        model_path: str,
        batch_size: int=128
    ):
        from transformers import AutoProcessor, AutoModelForCausalLM
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float32 if self.device == 'cpu' else torch.float16
        self.processor = AutoProcessor.from_pretrained(
            'microsoft/Florence-2-base',
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            trust_remote_code=True
        ).to(self.device)

    def gen_text(self, images: List[Image.Image], prompt: str=''):
        if not prompt:
            prompt = '<CAPTION>'

        gen_texts = []
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i+self.batch_size]
            results = self._batch_gen_text(batch_images, prompt)
            gen_texts.extend(results)
        return gen_texts

    @torch.inference_mode()
    def _batch_gen_text(self, images: List[Image.Image], prompt: str):
        inputs = self.processor(
            images=images,
            text=[prompt] * len(images),
            return_tensors='pt',
            do_resize=(self.device == 'cpu')
        )
        inputs = inputs.to(device=self.device, dtype=self.dtype)
        gen_ids = self.model.generate(
            input_ids=inputs['input_ids'],
            pixel_values=inputs['pixel_values'],
            max_new_tokens=20,
            num_beams=1,
            do_sample=False
        )
        return self.processor.batch_decode(gen_ids, skip_special_tokens=True)