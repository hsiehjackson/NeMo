import os

from dataclasses import dataclass, is_dataclass

from omegaconf import OmegaConf

from nemo.core.config import hydra_runner

import torch

from nemo.collections.asr.models import ASRModel
from nemo.utils import logging, model_utils

import time

@dataclass
class EvaluationConfig():
    # Required configs
    model_dir: str = ""


@hydra_runner(config_name="EvaluationConfig", schema=EvaluationConfig)
def main(cfg: EvaluationConfig):
    logging.set_verbosity(50)

    torch.set_grad_enabled(False)

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    map_location = torch.device('cuda:0')

    print(cfg.model_dir)
    model_names = os.listdir(cfg.model_dir)
    model_names = sorted(model_names)
    print(model_names)

    mins = [0.5, 1, 5, 10]

    for model_name in model_names:

        for test_minutes in mins:
            model_cfg = ASRModel.restore_from(restore_path=cfg.model_dir + "/" + model_name, return_config=True)
            classpath = model_cfg.target  # original class path
            imported_class = model_utils.import_class_by_path(classpath)  # type: ASRModel
            logging.info(f"Restoring model : {imported_class.__name__}")
            asr_model = imported_class.restore_from(
                restore_path=cfg.model_dir + "/" + model_name, map_location=map_location,
            )  # type: ASRModel
            asr_model = asr_model.eval()

            len = int(16000 * 60 * test_minutes)
            input_signal_long = torch.randn(size=(1, len), device=asr_model.device)
            length_long = torch.tensor([len], device=asr_model.device)

            # switch to local attn
            #asr_model.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=(64, 64))

            avg_time = 0
            rounds = 5

            for i in range(rounds):
                with torch.no_grad():
                    start_time = time.time()
                    asr_model.forward(input_signal=input_signal_long, input_signal_length=length_long)
                    end_time = time.time()
                elapsed_time = end_time - start_time  # calculate the elapsed time
                avg_time += elapsed_time

            avg_time /= rounds

            print(model_name, "achieved avg time of", avg_time, "on", test_minutes, "minutes")

        #print('---maximum minutes for ' + model_name + ' is ' + str(max_mins))


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter