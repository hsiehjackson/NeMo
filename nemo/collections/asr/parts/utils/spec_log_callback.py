from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

import wandb

from torchvision import transforms

class SpectrogramLogCallback(Callback):

    def __init__(self, num_display=8):
        super().__init__()
        self.num_display = num_display

    def get_image(self, t, masks=None):
        """t = t - t.min()
        t = t / t.max()

        i = transforms.ToPILImage()(t)

        w, h = i.size
        if w < h // 6:
            w = h // 6
        if h < w // 6:
            h = w // 6
        i = i.resize((w, h))"""

        if masks is None:
            return wandb.Image(t)
        else:
            wandb.Image(t,
                        masks={
                            "mask_data": masks.round().int(),
                            "class_labels": {1: "masked"}
                        }
                        )

    """@rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx % 100 != 0:
            return

        signal, signal_len, transcript, transcript_len = batch
        log_probs = outputs['log_probs']
        spectrograms, masked_spectrograms, spec_masks, spec_recon = outputs['extra']


        #trainer.logger.experiment[0].log({
        wandb.log({
            "global_step": trainer.global_step,
            "train_spec": [self.get_image(x) for x in spectrograms[:self.num_display]],
            "train_spec_recon": [self.get_image(x) for x in spec_recon[:self.num_display]],
            "train_spec_masked": [self.get_image(x) for x in masked_spectrograms[:self.num_display]],
            "train_masks": [self.get_image(x) for x in spec_masks[:self.num_display]],
        })
        if log_probs is not None:
            wandb.log({
                "train_log_probs": [self.get_image(x) for x in log_probs[:self.num_display]],
            })"""

    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx % 100 != 0:
            return

        signal, signal_len, transcript, transcript_len = batch
        log_probs = outputs['log_probs']
        spectrograms, masked_spectrograms, spec_masks, spec_recon = outputs['extra']


        #trainer.logger.experiment[0].log({
        wandb.log({
            "global_step": trainer.global_step,
            "val_spec": [self.get_image(x) for x in spectrograms[:self.num_display]],
            "val_spec_recon": [self.get_image(x, m) for x, m in
                               zip(spec_recon[:self.num_display], spec_masks[:self.num_display])],
            "val_spec_masked": [self.get_image(x) for x in masked_spectrograms[:self.num_display]],
            "val_masks": [self.get_image(x) for x in spec_masks[:self.num_display]],
        })
        if log_probs is not None:
            wandb.log({
                "val_log_probs": [self.get_image(x) for x in log_probs[:self.num_display]],
            })
