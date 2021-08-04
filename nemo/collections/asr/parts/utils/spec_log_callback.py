from pytorch_lightning.callbacks import Callback
import wandb


class SpectrogramLogCallback(Callback):

    def __init__(self, num_display=8):
        super().__init__()
        self.num_display = num_display
        self.start_logging = False

    def on_sanity_check_end(self, trainer, pl_module):
        self.start_logging = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx % 100 != 0 or not self.start_logging:
            return

        signal, signal_len, transcript, transcript_len = batch
        log_probs = outputs['log_probs']
        spectrograms, masked_spectrograms, spec_masks = outputs['extra']

        trainer.logger.experiment.log({
            "train_spec": [wandb.Image(spectrograms[:self.num_display])],
            "train_spec_masked": [wandb.Image(masked_spectrograms[:self.num_display])],
            "train_masks": [wandb.Image(spec_masks[:self.num_display])],
            "train_log_probs": [wandb.Image(log_probs[:self.num_display])],
        })

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx % 100 != 0 or not self.start_logging:
            return

        signal, signal_len, transcript, transcript_len = batch
        log_probs = outputs['log_probs']
        spectrograms, masked_spectrograms, spec_masks = outputs['extra']

        trainer.logger.experiment.log({
            "val_spec": [wandb.Image(spectrograms[:self.num_display])],
            "val_spec_masked": [wandb.Image(masked_spectrograms[:self.num_display])],
            "val_masks": [wandb.Image(spec_masks[:self.num_display])],
            "val_log_probs": [wandb.Image(log_probs[:self.num_display])],
        })
