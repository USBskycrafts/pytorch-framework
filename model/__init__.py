from .symbiosis.net import Symbiosis

model_list = {
    "symbiosis-v3": Symbiosis
}


def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
