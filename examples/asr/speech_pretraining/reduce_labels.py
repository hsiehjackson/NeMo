import json

@dataclass
class ReduceLabelsConfig:
    in_manifest: str
    out_manifest: str

@hydra_runner(config_name="ReduceLabelsConfig", schema=ReduceLabelsConfig)
def main(cfg: FeatClusteringConfig) -> FeatClusteringConfig:
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    with open(cfg.out_manifest, 'w', encoding='utf-8') as f:
        with open(cfg.in_manifest, 'r') as fr:
            for idx, line in enumerate(fr):
                item = json.loads(line)

                new_list = []
                prev_tok = -1
                for tok in item['token_labels']:
                    if tok != prev_tok:
                        prev_tok = tok
                        new_list.append(tok)

                print(item['token_labels'])
                print(new_list)
                input()

                item['token_labels'] = new_list

                f.write(json.dumps(item) + "\n")
