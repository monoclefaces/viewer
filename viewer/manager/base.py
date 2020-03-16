from pathlib import Path
from typing import Union
import yaml
import pprint

class Config(object):
    """Basic Config Class"""
    def __init__(self, cfg_yaml_path:str):
        r"""
        Configuration of Settings

        Args:
            cfg_yaml_path: argument file path(`str`)

        It will create directory automatically by `cfg_yaml_path`, 
        
        ```
        checkpoints
        └── eval_type
            ├── exp_arg1
            │   ├── exp1_summary
            │   ├── "sv_name-attr_type1"
            │   ├── "sv_name-attr_type2"
            │   └── "sv_name-attr_type3"
            ├── exp_arg2
            └── exp_arg3
        ```

        `cfg_yaml_path` file shuould like below.
        ```yaml
        # confiugre.yaml
        # path settings
        project_path: "."
        data_path: "./data"
        experiment:
          eval_type: roar
          attr_type: 
            - "vanillagrad"
            - "gradcam"
          sv_name: "{your sv file name}"
        ...
        ```
        """
        self.cfg_yaml_path = Path(cfg_yaml_path)
        with self.cfg_yaml_path.open(mode="r", encoding="utf-8") as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
        self.__dict__.update(conf)
        self.set_default_path(conf)
        
    def set_default_path(self, conf):
        r"""
        Check arguments and create experiment path
        """
        ConfigChecker.check_default_conf(conf)

        self.project_path = Path(self.project_path)
        self.data_path = Path(self.data_path)
        self.exp_path = self.project_path / "checkpoints" / self.eval_type
        ConfigChecker.check_dir_exist(self.exp_path, file=False)

    @property
    def conf(self):
        return self.__dict__


class ConfigChecker(object):
    r"""
    Check configure file, if not exist set to default values.
    """
    # checklist = ["project_path", "data_path", "eval_type", "attr_type", "sv_name"]
    checklist = ["project_path", "data_path"]

    @classmethod
    def check_default_conf(cls, conf:dict):
        r"""
        Default Argument setting
        must contain following five arguments: 
            - project_path
            - data_path
        """
        for check_arg in cls.checklist:
            if conf.get(check_arg) is None:
                try:
                    key_error_text = [f"Configure file dosen't have `{check_arg}`, check your argument file in `{self.cfg_yaml_path}`"]
                    key_error_text += [f"{'='*20}", f"Your yaml file settings:"]
                    raise KeyError("\n".join(key_error_text))
                except KeyError as e:
                    print(e.args[0])
                    pprint.pprint(conf)

    @classmethod
    def check_postprocessor(cls, conf):
        r"""
        PostProcessor Argument setting
        if there is no setting about `postprocessor` in config file, setting to default values
        """
        if conf.get("postprocessor") is None:
            print("")

    @classmethod
    def check_dir_exist(cls, path:Union[str, Path], file:bool=False):
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
                print(f"Given path doesn't exists, created `{path}`")
        else:
            if not path.exists():
                path.mkdir(parents=True)
                print(f"Given path doesn't exists, created `{path}`")


class Checkpoints(object):
    """Checkpoint Manager"""
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

