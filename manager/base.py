from pathlib import Path
from typing import Union
import yaml


class Config(object):
    """Basic Config Class"""
    def __init__(self, cfg_yaml_path:str, root:str=".", data_path:str="./data"):
        r"""
        Configuration of Settings

        Args:
            root: root path of project, default="."
            data_path: data path that contains data directories
            cfg_yaml_path: argument file path(`str`)

        It will create directory automatically by `cfg_yaml_path`, 
        
        ```
        checkpoints
        └── data_type
            └── eval_type
                ├── exp_arg1
                │   ├── exp1_summary
                │   ├── model_type + attr_type1  <-weights
                │   ├── model_type + attr_type2
                │   └── model_type + attr_type3
                ├── exp_arg2
                └── exp_arg3
        ```
        `cfg_yaml_path` file shuould like below.
        ```yaml
        # confiugre.yaml
        type:
          data_type: mnist
          eval_type: roar
          model_type: resnet18
          attr_type: ["vanillagrad", "gradcam"]
        ...
        ```
        """
        self.prj_path = Path(root)
        self.data_path = Path(data_path)
        with open(cfg_yaml_path, mode="r") as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
        # vars(self).update(conf)
        self.__dict__.update(conf)
        self.check_type_args()
        
    def check_type_args(self):
        r"""
        Check arguments and create experiment path
        """
        type_args = self.conf["type_args"]
        check_types = ["data_type", "eval_type", "model_type", "attr_type"]
        for c_type in check_types:
            if not (c_type in type_args):
                raise KeyError(f"Configure file dosen't have {c_type}, check your argument file")
        
        self.exp_path = self.prj_path / "checkpoints" / type_args["data_type"] / type_args["eval_type"]
        self.check_dir_exist(self.exp_path)

    def check_dir_exist(self, path:Union[str, Path], file:bool=False):
        r"""
        Check directory file is exists, if not exists will create one

        Args:
            path: `str` or `pathlib.Path` type
            file: if True, will create a file, not a directory path
        """
        if not isinstance(path, Path):
            path = Path(path)
        if file:
            if not path.exists():
                path.touch()
                print(f"Given path doesn't exists, created {path}")
        else:
            if not path.exists():
                path.mkdir(parents=True)
                print(f"Given path doesn't exists, created {path}")

    @property
    def conf(self):
        return self.__dict__



class Checkpoints(object):
    """Model Checkpoint Manager"""
    def __init__(self, cfg):
        r"""
        Save details about model weights and summaries
        """

    def save_model(self):
        r"""
        Save model weights
        """

    def save_summary(self):
        r"""
        Save training stats
        """

