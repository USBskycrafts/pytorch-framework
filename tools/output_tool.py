import json
import numpy as np

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
    result = {}
    for metric, value in data.items():
        result[metric +
               " mean"] = "{:<7}".format(f"{np.mean(value):<2.2f}")[:7]
        # result[metric +
        #        " std"] = "{:<7}".format(f"{np.std(value):<2.2f}")[:7]
    return json.dumps(result, sort_keys=True, ensure_ascii=False)
