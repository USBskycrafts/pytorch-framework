from .template.net import TemplateNet

model_list = {
    "symbiosis": TemplateNet
}


def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
