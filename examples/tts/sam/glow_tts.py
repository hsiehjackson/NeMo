from nemo.collections.tts.sam.models import glow_tts
from pytorch_lightning import Trainer
import hydra
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateLogger, ModelCheckpoint
import os

@hydra.main(config_path="conf/glow_tts_config.yaml")
def main(hps):

    print(hps.train)
    print(os.getcwd())

    model = glow_tts.GlowTTSModel(hps)
    tb_logger = pl_loggers.TensorBoardLogger(hps.train.model_dir)
    lr_logger = LearningRateLogger()
    checkpoint_callback = ModelCheckpoint()
    trainer = Trainer.from_argparse_args(hps.train, gpus=-1, logger=tb_logger, callbacks=[lr_logger, checkpoint_callback])
    trainer.fit(model)


if __name__ == '__main__':
    main()
