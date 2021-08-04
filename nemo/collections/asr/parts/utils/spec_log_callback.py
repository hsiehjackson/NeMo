from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

import wandb

from torchvision import transforms

class SpectrogramLogCallback(Callback):

    def __init__(self, num_display=8):
        super().__init__()
        self.num_display = num_display

    def get_image(self, t):
        t = t.float()
        t -= t.min()
        t /= t.max()

        i = transforms.ToPILImage(t)

        w, h = i.size
        if w < h // 2:
            w = h // 2
        if h < w // 2:
            h = w // 2
        i = i.resize((w, h))
        
        return wandb.Image(i)

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx % 100 != 0:
            return

        signal, signal_len, transcript, transcript_len = batch
        log_probs = outputs['log_probs']
        spectrograms, masked_spectrograms, spec_masks = outputs['extra']


        #trainer.logger.experiment[0].log({
        wandb.log({
            "global_step": trainer.global_step,
            "train_spec": [self.get_image(x) for x in spectrograms[:self.num_display]],
            "train_spec_masked": [self.get_image(x) for x in masked_spectrograms[:self.num_display]],
            "train_masks": [self.get_image(x) for x in spec_masks[:self.num_display]],
            "train_log_probs": [self.get_image(x) for x in log_probs[:self.num_display]],
        })

    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx % 100 != 0:
            return

        signal, signal_len, transcript, transcript_len = batch
        log_probs = outputs['log_probs']
        spectrograms, masked_spectrograms, spec_masks = outputs['extra']


        #trainer.logger.experiment[0].log({
        wandb.log({
            "global_step": trainer.global_step,
            "val_spec": [self.get_image(x) for x in spectrograms[:self.num_display]],
            "val_spec_masked": [self.get_image(x) for x in masked_spectrograms[:self.num_display]],
            "val_masks": [self.get_image(x) for x in spec_masks[:self.num_display]],
            "val_log_probs": [self.get_image(x) for x in log_probs[:self.num_display]],
        })
