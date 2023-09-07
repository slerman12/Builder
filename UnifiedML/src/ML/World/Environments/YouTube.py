# from vidgear.gears import CamGear
#
# stream = CamGear(source='https://youtu.be/dQw4w9WgXcQ', stream_mode = True, logging=True).start()
#
# while True:
#     frame = stream.read()

from vidgear.gears import CamGear
import cv2

stream = CamGear(source='https://youtu.be/dQw4w9WgXcQ', stream_mode = True, logging=True).start()  # YouTube Video URL as input

# infinite loop
while True:

    # read frames
    frame = stream.read()

    # check if frame is None
    if frame is None:
        # if True break the infinite loop
        break

    # do something with frame here

    cv2.imshow("Output Frame", frame)
    # Show output window

    key = cv2.waitKey(1) & 0xFF
    # check for 'q' key-press
    if key == ord("q"):
        #if 'q' key-pressed break out
        break

cv2.destroyAllWindows()
# close output window

# safely close video stream.
stream.stop()


from torch import nn

from huggingface_hub import hf_hub_download

from GroundingDINO.groundingdino.util.inference import predict
from GroundingDINO.demo.gradio_app import load_model_hf


class GroundingDINO(nn.Module):
    def __init__(self, caption='little robot dog'):
        super().__init__()

        self.caption = caption

        repo_id = 'ShilongLiu/GroundingDINO'

        cache_config_file = hf_hub_download(repo_id=repo_id, filename='GroundingDINO_SwinB.cfg.py')

        self.GroundingDINO = load_model_hf(cache_config_file,
                                           repo_id=repo_id,
                                           filename='groundingdino_swinb_cogcoor.pth')

    def forward(self, obs, caption=None):
        boxes, logits, phrases = predict(
            model=self.GroundingDINO,
            image=obs,
            caption=caption or self.caption,
            box_threshold=0.3,
            text_threshold=0.25,
            device=obs.device
        )

        return boxes, logits, phrases


class AutoLabel(nn.Module):
    def __init__(self, caption='little robot dog'):
        super().__init__()

        self.GroundingDINO = GroundingDINO(caption)

    def forward(self, batch):
        # TODO Highest proba, 4 bounding-box coords, flatten
        batch.label = self.GroundingDINO(batch.obs)

        return batch

