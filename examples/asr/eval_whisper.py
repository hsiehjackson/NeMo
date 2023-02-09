import os
import soundfile as sf
import numpy as np

from dataclasses import dataclass, is_dataclass

from omegaconf import OmegaConf

from nemo.core.config import hydra_runner

import json
import whisper
from whisper.normalizers import EnglishTextNormalizer

@dataclass
class EvaluationConfig():
    # Required configs
    manifest: str
    out_manifest: str
    whisper_model: str = "small"


@hydra_runner(config_name="EvaluationConfig", schema=EvaluationConfig)
def main(cfg: EvaluationConfig):

    model = whisper.load_model("small")
    std = EnglishTextNormalizer()

    with open(cfg.out_manifest, 'w', encoding='utf-8') as f:
        with open(cfg.manifest, 'r') as fr:
            for idx, line in enumerate(fr):
                item = json.loads(line)

                pred = model.transcribe(item["audio_filepath"], verbose=True)["text"]
                pred = std(pred)

                item['pred_text'] = pred
                item['text'] = std(item['text'])

                f.write(json.dumps(item) + "\n")


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter