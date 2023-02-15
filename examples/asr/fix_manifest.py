import os
import soundfile as sf
import numpy as np

from dataclasses import dataclass, is_dataclass

from omegaconf import OmegaConf

from nemo.core.config import hydra_runner

from whisper.normalizers import EnglishTextNormalizer
import json

from pydub import AudioSegment


@dataclass
class EvaluationConfig():
    # Required configs
    manifest: str = "/home/samuel/data/manifests/ls_test_other_manifest.json"
    out_manifest: str = ""


@hydra_runner(config_name="EvaluationConfig", schema=EvaluationConfig)
def main(cfg: EvaluationConfig):
    if cfg.out_manifest == "":
        cfg.out_manifest = cfg.manifest[:-5] + "_fix.json"

    durs = []

    with open(cfg.manifest, 'r') as fr:
        for idx, line in enumerate(fr):
            item = json.loads(line)

            path = item["audio_filepath"]

            a = AudioSegment.from_file(path)

            correct_dur = round(a.duration_seconds - 0.1, 2)
            cur_dur = item["duration"]

            if cur_dur > correct_dur + 0.01:
                print("fixed", cur_dur, "to", correct_dur)

            item["duration"] = correct_dur

            durs.append((correct_dur, item))

            # f.write(json.dumps(item) + "\n")

    durs = sorted(durs, key=lambda x: x[0])

    with open(cfg.out_manifest, 'w', encoding='utf-8') as f:
        for dur, it in durs:
            print(dur, it["audio_filepath"])
            f.write(json.dumps(it) + "\n")

if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
