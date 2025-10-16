import importlib
import importlib.util
import os
from dataclasses import dataclass

import yaml
from openai import OpenAI


@dataclass
class Config:
    data_path: str
    output_dir: str
    host: str = "localhost"
    port: int = 8000
    model_name: str = None

    def __post_init__(self):
        self.openai_api_key = "EMPTY"
        self.openai_api_base = f"http://{self.host}:{self.port}/v1"
        
        # If model_name is not provided, get it from the API
        if self.model_name is None:
            client = OpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_api_base
            )
            self.model_name = client.models.list().data[0].id.strip()
            del client
        else:
            # Clean the model name if provided
            self.model_name = self.model_name.strip()


def bypass_import_function(loader, node):
    return node


def import_function(loader, node):
    function_name = loader.construct_scalar(node)
    module_name, function_name = function_name.rsplit(".", 1)
    if "." in module_name:
        module_filename = module_name.replace(".", "/")
    else:
        module_filename = module_name

    module_path = os.path.normpath(
        os.path.join(os.getcwd(), "src", f"{module_filename}.py"))

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        function = getattr(module, function_name)
        return function
    else:
        return node


def load_yaml_config(config_path, full=True):
    constructor_key = "!function"
    if full:
        yaml.add_constructor(constructor_key, import_function)
    else:
        yaml.add_constructor(constructor_key, bypass_import_function)

    with open(config_path, "r") as f:
        yaml_config = yaml.full_load(f)
    return yaml_config
