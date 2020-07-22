import nemo.collections.tts.sam as nemo_tts_sam
from pytorch_lightning import Trainer
import hydra
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateLogger, ModelCheckpoint
import os
import sys


@hydra.main(config_path="conf/glow_tts_config.yaml")
def main(hps):

    print(hps.train)
    print(os.getcwd())
    print(sys.path)

    model = nemo_tts_sam.models.glow_tts.GlowTTSModel(hps)
    tb_logger = pl_loggers.TensorBoardLogger(hps.train.model_dir)
    lr_logger = LearningRateLogger()
    checkpoint_callback = ModelCheckpoint(hps.train.model_dir)
    trainer = Trainer.from_argparse_args(
        hps.train, gpus=-1, distributed_backend='dp', logger=tb_logger, callbacks=[lr_logger, checkpoint_callback]
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
