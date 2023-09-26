# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch
from torch import nn

from Utils import gather


# Example Usage:
# python Run.py env=YouTube env.url='https://youtube.com/live/...=share'
#                           env.transform=Sequential
#                           env.transform._targets_='["transforms.Resize(32)","World.Environments.YouTube.AutoLabel"]'
#               stream=true

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

        self._exp_ = True  # Flag that toggles passing in full exp/batch-dict instead of just obs

    def forward(self, exp):
        if self.device is not None:
            exp.obs = exp.obs.to(self.device)

        boxes, logits, phrases = self.GroundingDINO(exp.obs)

        if boxes.nelement():
            indices = logits.argmax(-1)
            box = gather(boxes, indices)  # Highest proba bounding-box

            # Extract label  TODO RandomResizeCrop Just the area around the Bittle.
            # Use absolute for position
            exp.label = box
        else:
            exp.label = torch.zeros(1, 6)

        return exp


def predict_batch(
        model,
        images: torch.Tensor,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda"
):
    from groundingdino.util.inference import preprocess_caption
    from groundingdino.util.utils import get_phrases_from_posmap

    caption = preprocess_caption(caption=caption)

    model = model.to(device)
    image = images.to(device)

    print(f"Image shape: {image.shape}") # Image shape: torch.Size([num_batch, 3, 800, 1200])
    with torch.no_grad():
        outputs = model(image, captions=[caption for _ in range(len(images))]) # <------- I use the same caption for all the images for my use-case

    print(f'{outputs["pred_logits"].shape}') # torch.Size([num_batch, 900, 256])
    print(f'{outputs["pred_boxes"].shape}') # torch.Size([num_batch, 900, 4])
    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)

    mask = prediction_logits.max(dim=1)[0] > box_threshold
    logits = prediction_logits[mask]  # logits.shape = (n, 256)
    boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)

    phrases = [
        get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
        for logit
        in logits
    ]

    return boxes, logits.max(dim=1)[0], phrases


class GroundingDINO(nn.Module):
    """
    GroundingDINO - one of the coolest vision-language models, combining DINO with language.
    Repo: ShilongLiu/GroundingDINO.
    """

    def __init__(self, caption='little robot dog'):
        super().__init__()

        try:
            from groundingdino.util.inference import predict

            from groundingdino.models import build_model
            from groundingdino.util.slconfig import SLConfig
            from groundingdino.util.utils import clean_state_dict

            from huggingface_hub import hf_hub_download
        except Exception as e:
            print(e)
            raise RuntimeError('Make sure you have installed GroundingDINO: \n'
                               '$ pip install groundingdino-py\n'
                               'and huggingface_hub. See ShilongLiu/GroundingDINO.')

        def load_model_hf(model_config_path, repo_id, filename, device='cpu'):
            args = SLConfig.fromfile(model_config_path)
            model = build_model(args)
            args.device = device

            cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
            checkpoint = torch.load(cache_file, map_location='cpu')
            log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
            print("Model loaded from {} \n => {}".format(cache_file, log))
            _ = model.eval()
            return model

        self.caption = caption

        repo_id = 'ShilongLiu/GroundingDINO'

        cache_config_file = hf_hub_download(repo_id=repo_id, filename='GroundingDINO_SwinB.cfg.py')

        self.GroundingDINO = load_model_hf(cache_config_file,
                                           repo_id=repo_id,
                                           filename='groundingdino_swinb_cogcoor.pth')

        self.predict = predict

    def forward(self, obs, caption=None):
        # TODO batch iterate instead of assume-squeeze OR
        # TODO https://github.com/IDEA-Research/GroundingDINO/issues/102#issuecomment-1558728065
        #     https://github.com/yuwenmichael/Grounding-DINO-Batch-Inference
        if len(obs.shape) == 4:
            obs = obs.squeeze(0)

        # TODO Somehow override one op to cpu
        boxes, logits, phrases = self.predict(
            model=self.GroundingDINO,
            image=obs.to(torch.float32),
            caption=caption or self.caption,
            box_threshold=0.3,
            text_threshold=0.25,
            device=obs.device
        )

        return boxes, logits, phrases


# TODO Delete
if __name__ == '__main__':
    import os

    import numpy as np
    from PIL import Image

    from heic2png import HEIC2PNG  # pip install HEIC2PNG

    import groundingdino.datasets.transforms as T


    def load_image(image_path):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        src = Image.open(image_path).convert("RGB")
        img = np.asarray(src)
        image_transformed, _ = transform(src, None)
        return img, image_transformed


    path = '/Users/sam/Code/Playground/Magic/images/'

    for local_image_path in os.listdir(path):
        if local_image_path[-5:] == '.heic':
            HEIC2PNG(path + local_image_path).save()
            os.system(f'rm {path}{local_image_path}')
            local_image_path = local_image_path.replace('.heic', '.png')

        image_source, image = load_image(path + local_image_path)

        GD = GroundingDINO()
        boxes, logits, _ = GD(image)

        indices = logits.argmax(-1)
        box = gather(boxes, indices)  # Highest proba bounding-box

        print(box)

        # annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        #
        # os.makedirs('./generated2/', exist_ok=True)
        #
        # cv2.imwrite('./generated2/' + local_image_path, annotated_frame)

        break
