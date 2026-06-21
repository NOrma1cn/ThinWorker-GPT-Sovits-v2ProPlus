# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/models/t2s_lightning_module.py
# reference: https://github.com/lifeiteng/vall-e
import torch

from thin_tts.models.t2s_model import Text2SemanticDecoder


class Text2SemanticLightningModule(torch.nn.Module):
    def __init__(self, config, output_dir=None, is_train=False):
        super().__init__()
        self.config = config
        self.top_k = 3
        self.model = Text2SemanticDecoder(config=config, top_k=self.top_k)
