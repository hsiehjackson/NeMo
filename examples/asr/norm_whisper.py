import os
import soundfile as sf
import numpy as np

from dataclasses import dataclass, is_dataclass

from omegaconf import OmegaConf

from nemo.core.config import hydra_runner

from whisper.normalizers import EnglishTextNormalizer
import json

@dataclass
class EvaluationConfig():
    # Required configs
    manifest: str
    out_manifest: str = ""


@hydra_runner(config_name="EvaluationConfig", schema=EvaluationConfig)
def main(cfg: EvaluationConfig):

    if cfg.out_manifest == "":
        cfg.out_manifest = cfg.manifest[:-5] + "_norm.json"

    std = EnglishTextNormalizer()

    with open(cfg.out_manifest, 'w', encoding='utf-8') as f:
        with open(cfg.manifest, 'r') as fr:
            for idx, line in enumerate(fr):
                item = json.loads(line)
                text = item["text"]
                pred = item["pred_text"]

                item["text"] = std(text)
                item["pred_text"] = std(pred)

                f.write(json.dumps(item) + "\n")


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter