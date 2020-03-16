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
        # general settings
        project_path: "."
        data_path: "./data"

        experiment: 
            eval_type: "roar"  
            attr_type: 
                - "vanillagrad"
                - "inputgrad"
            sv_name: "custom_model"

        # saliency settings
        postprocessor:
            collaspe_mode: 0

        evaluator:
            channel_reduction: True
        ...
        ```
        """
        self.cfg_yaml_path = Path(cfg_yaml_path)
        with self.cfg_yaml_path.open(mode="r", encoding="utf-8") as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
        # self.__dict__.update(conf)
        self.setting_by_cfg_file(conf)
    
    @property
    def conf(self):
        return self.__dict__

    def setting_by_cfg_file(self, conf):
        r"""
        Check arguments and create experiment path
        """
        cfg = conf.get("default")
        check_dict = ConfigChecker.check_env(cfg, env_name="default", update_default=True)
        self.__dict__.update(check_dict)
        
        self.project_path = Path(self.project_path)
        self.data_path = Path(self.data_path)

        if conf.get("experiment") is not None:
            cfg = conf.get("experiment")
            check_dict = ConfigChecker.check_env(cfg, env_name="experiment", update_default=True)
            self.__dict__.update(check_dict)
            self.exp_path = self.project_path / "checkpoints" / self.eval_type
            ConfigChecker.check_dir_exist(self.exp_path, file=False)
        else:
            print("No Configure for `experiment`")

        if conf.get("postprocessor") is not None:
            cfg = conf.get("postprocessor")
            check_dict = ConfigChecker.check_env(cfg, env_name="postprocessor", update_default=True)
        else:
            check_dict = ConfigChecker.checkdefault_generator("postprocessor")
        self.__dict__.update(check_dict)

class ConfigChecker(object):
    r"""
    Check configure file
    """    
    @classmethod
    def check_env(cls, conf:dict, env_name:str, update_default=False):
        r"""
        ```yaml
        environment:
          a: 1
          b: 2
        ```
        conf = {"a": 1, "b": 2}
        env_name = "environment"
        """
        check = cls.checkdefault_generator(env_name)
        for check_arg in check.keys():
            if conf.get(check_arg) is None:
                cls.error_print(check_arg, conf, env_name)
        if update_default:
            check.update(conf)
        return dict(check)

    @staticmethod
    def checkdefault_generator(env_name:str):
        r"""
        This will generate (check_argument, default value) tuple.
        [default]
        Default Argument setting must contain following arguments: 
            - project_path: None
            - data_path: None

        [experiment]
        Experiment Argument setting must contain following arguments: 
            - eval_type: None
            - attr_type: None
            - sv_name: None

        [postprocessor]
        PostProcessor Argument setting contains following arguments: 
            - collaspe_mode: 0
        """
        if env_name == "default":
            check = [("project_path", None), ("data_path", None)]
        elif env_name == "experiment":
            check = [("eval_type", None), ("attr_type", None), ("sv_name", None)]
        elif env_name == "postprocessor":
            check = [("collaspe_mode", 0)]
        return dict(check)

    @staticmethod
    def error_print(check_arg, conf, env_name:str="default"):
        if env_name == "default":
            env_str = ""
        else:
            env_str = f" under `{env_name}`"
        try:
            key_error_text = [f"Configure file dosen't have `{check_arg}`{env_str}, "]
            key_error_text += [f"{'='*20}", f"Your yaml file settings{env_str}:"]
            raise KeyError("\n".join(key_error_text))
        except KeyError as e:
            print(e.args[0])
            pprint.pprint(conf)

    @staticmethod
    def check_dir_exist(path:Union[str, Path], file:bool=False):
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

