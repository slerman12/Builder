# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from torch import nn


class GroundingDINO(nn.Module):
    """
    GroundingDINO - one of the coolest vision-language models, combining DINO with language.
    Repo: ShilongLiu/GroundingDINO.
    """
    def __init__(self, caption='little robot dog'):
        super().__init__()

        from huggingface_hub import hf_hub_download

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
