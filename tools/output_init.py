from .output_tool import basic_output_function, null_output_function, general_image_metrics

output_function_dic = {
    "Basic": basic_output_function,
    "Null": null_output_function,
    "VISION": general_image_metrics
}


def init_output_function(config, *args, **params):
    name = config.get("output", "output_function")

    if name in output_function_dic:
        return output_function_dic[name]
    else:
        raise NotImplementedError
