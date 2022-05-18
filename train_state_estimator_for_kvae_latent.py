import os
import hydra
from attrdict import AttrDict
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
from RunDNNSingleNetwork import RunDNNSingleNetwork

import sys
import pathlib
p = pathlib.Path(__file__).resolve()
sys.path.append(str(p.parent))
repository_path = "/".join(str(p.parent).split("/")[:-1])

# @hydra.main(config_path="conf/config_RAL_revise.yaml")
@hydra.main(config_path="conf/config_RAL_revise_block.yaml")
def get_config(cfg: DictConfig) -> None:
    cfg.log_dir = repository_path + "/logs/" + cfg.kvae_path + "/state_estimator"
    os.makedirs(cfg.log_dir, exist_ok=True)
    run = RunDNNSingleNetwork(cfg)
    run.dataset.kvae_path = cfg.kvae_path

    run.run()

get_config()


