from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only

import wandb


class SpectrogramLogCallback(Callback):

    def __init__(self, num_display=8):
        super().__init__()
        self.num_display = num_display

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx % 100 != 0:
            return

        signal, signal_len, transcript, transcript_len = batch
        log_probs = outputs['log_probs']
        spectrograms, masked_spectrograms, spec_masks = outputs['extra']

        """log_probs -= log_probs.min()
        log_probs /= log_probs.max()

        spectrograms -= spectrograms.min()
        spectrograms /= spectrograms.max()

        masked_spectrograms -= masked_spectrograms.min()
        masked_spectrograms /= masked_spectrograms.max()"""

        spec_masks = spec_masks.float()

        #trainer.logger.experiment[0].log({
        wandb.log({
            "global_step": trainer.global_step,
            "train_spec": [wandb.Image(x) for x in spectrograms[:self.num_display]],
            "train_spec_masked": [wandb.Image(x) for x in masked_spectrograms[:self.num_display]],
            "train_masks": [wandb.Image(x) for x in spec_masks[:self.num_display]],
            "train_log_probs": [wandb.Image(x) for x in log_probs[:self.num_display]],
        })

    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx % 100 != 0:
            return

        signal, signal_len, transcript, transcript_len = batch
        log_probs = outputs['log_probs']
        spectrograms, masked_spectrograms, spec_masks = outputs['extra']

        spec_masks = spec_masks.float()

        #trainer.logger.experiment[0].log({
        wandb.log({
            "global_step": trainer.global_step,
            "val_spec": [wandb.Image(x) for x in spectrograms[:self.num_display]],
            "val_spec_masked": [wandb.Image(x) for x in masked_spectrograms[:self.num_display]],
            "val_masks": [wandb.Image(x) for x in spec_masks[:self.num_display]],
            "val_log_probs": [wandb.Image(x) for x in log_probs[:self.num_display]],
        })
