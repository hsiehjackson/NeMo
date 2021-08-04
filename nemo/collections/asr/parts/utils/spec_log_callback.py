from pytorch_lightning.callbacks import Callback
import wandb


class SpectrogramLogCallback(Callback):

    def __init__(self, num_display=8):
        super().__init__()
        self.num_display = num_display

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx % 100 != 0:
            return

        signal, signal_len, transcript, transcript_len = batch
        print(outputs[0].shape)
        print(outputs[1].shape)
        print(outputs[2].shape)
        for i in outputs[3]:
            print(i.shape)
        log_probs, encoded_len, greedy_predictions, (spectrograms, masked_spectrograms, spec_masks) = outputs

        trainer.logger.experiment.log({
            "train_spec": [wandb.Image(spectrograms[:self.num_display])],
            "train_spec_masked": [wandb.Image(masked_spectrograms[:self.num_display])],
            "train_masks": [wandb.Image(spec_masks[:self.num_display])],
            "train_log_probs": [wandb.Image(log_probs[:self.num_display])],
        })

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx % 100 != 0:
            return

        signal, signal_len, transcript, transcript_len = batch
        print(outputs[0].shape)
        print(outputs[1].shape)
        print(outputs[2].shape)
        for i in outputs[3]:
            print(i.shape)
        log_probs, encoded_len, greedy_predictions, (spectrograms, masked_spectrograms, spec_masks) = outputs

        trainer.logger.experiment.log({
            "val_spec": [wandb.Image(spectrograms[:self.num_display])],
            "val_spec_masked": [wandb.Image(masked_spectrograms[:self.num_display])],
            "val_masks": [wandb.Image(spec_masks[:self.num_display])],
            "val_log_probs": [wandb.Image(log_probs[:self.num_display])],
        })
