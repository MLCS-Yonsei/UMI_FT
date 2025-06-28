"""
cd umi_act/UMI_Ft

python -m dit.train.py
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import hydra
from omegaconf import OmegaConf
import pathlib
from dit.workspace.base_workspace import BaseWorkspace
from dit.common import misc

base_path = os.path.dirname(os.path.abspath(__file__))

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=os.path.join(base_path, "config"),
    config_name="finetune.yaml"
)
def main(cfg: OmegaConf):
    # cls = hydra.utils.get_class(cfg._target_)
    cls = hydra.utils.get_class(cfg.workspace._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()