# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from vidgear.gears import CamGear

from torch import nn

from Utils import gather


class YouTube:
    """
    Live-streaming environment
    """
    def __init__(self, url, train=True, steps=1000):
        self.video = CamGear(source=url, stream_mode=True, logging=True).start()

        self.train = train
        self.steps = steps  # Controls evaluation episode length
        self.step = 0

    def step(self, action=None):
        return self.reset()

    def reset(self):
        self.step += 1
        return {'obs': self.video.read(), 'done': not self.train and not self.step % self.steps}


class AutoLabel(nn.Module):
    """
    A transform for auto-labelling object locations based on image and caption
    """
    def __init__(self, caption='little robot dog', device=None):
        super().__init__()

        # SotA object detection foundation model
        self.GroundingDINO = GroundingDINO(caption)

        mps = getattr(torch.backends, 'mps', None)  # M1 MacBook speedup

        # Set device
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() \
            else 'mps' if mps and mps.is_available() \
            else 'cpu'

    def __call__(self, exp):
        if self.device is not None:
            exp.obs = exp.obs.to(self.device)

        boxes, logits, phrases = self.GroundingDINO(exp.obs)

        indices = logits.argmax(-1)
        box = gather(boxes, indices)  # Highest proba bounding-box

        # Extract label
        exp.label = box

        return exp


class GroundingDINO(nn.Module):
    """
    GroundingDINO - one of the coolest vision-language models, combining DINO with language.
    Repo: ShilongLiu/GroundingDINO.
    """
    def __init__(self, caption='little robot dog'):
        super().__init__()

        from huggingface_hub import hf_hub_download

        # pip install groundingdino-py perhaps
        # or
        # git clone https://github.com/IDEA-Research/GroundingDINO.git
        # python -m pip install -e GroundingDINO
        # pip install transformers
        from GroundingDINO.groundingdino.util.inference import predict
        from GroundingDINO.demo.gradio_app import load_model_hf

        self.caption = caption

        repo_id = 'ShilongLiu/GroundingDINO'

        cache_config_file = hf_hub_download(repo_id=repo_id, filename='GroundingDINO_SwinB.cfg.py')

        self.GroundingDINO = load_model_hf(cache_config_file,
                                           repo_id=repo_id,
                                           filename='groundingdino_swinb_cogcoor.pth')

        self.predict = predict

    def forward(self, obs, caption=None):
        boxes, logits, phrases = self.predict(
            model=self.GroundingDINO,
            image=obs,
            caption=caption or self.caption,
            box_threshold=0.3,
            text_threshold=0.25,
            device=obs.device
        )

        return boxes, logits, phrases


if __name__ == '__main__':
    import os

    import torch

    import HEIC2PNG

    from GroundingDINO.groundingdino.util.inference import load_image

    for local_image_path in os.listdir('../Playground/Magic/images/'):
        if local_image_path[-5:] == '.heic':
            HEIC2PNG('../Playground/Magic/images/' + local_image_path).save()
            os.system(f'rm ../Playground/Magic/images/{local_image_path}')
            local_image_path = local_image_path.replace('.heic', '.png')

        image_source, image = load_image('../Playground/Magic/images/' + local_image_path)

        GD = GroundingDINO()
        boxes, logits, _ = GD(image)

        indices = logits.argmax(-1)
        box = gather(boxes, indices)  # Highest proba bounding-box

        # annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        #
        # os.makedirs('./generated2/', exist_ok=True)
        #
        # cv2.imwrite('./generated2/' + local_image_path, annotated_frame)

        break
