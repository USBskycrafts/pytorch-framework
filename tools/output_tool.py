import json

from .accuracy_tool import gen_micro_macro_result, general_image_metrics


def null_output_function(data, config, *args, **params):
    return ""


def basic_output_function(data, config, *args, **params):
    which = config.get("output", "output_value").replace(" ", "").split(",")
    temp = gen_micro_macro_result(data)
    result = {}
    for name in which:
        result[name] = temp[name]

    return json.dumps(result, sort_keys=True)



def vision_output_function(data, config, *args, **params):
    # data is the output from accuracy_tool
    which = config.get("output", "output_value").replace(" ", "").split(",")
    result = {}
    for metric, value in data.items():
        if metric in which:
            result[metric] = value
    return json.dumps(result, sort_keys=True)
        