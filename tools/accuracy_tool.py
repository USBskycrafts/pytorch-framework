from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import logging
import torch
from ignite.metrics import PSNR, SSIM
logger = logging.Logger(__name__)


def get_prf(res):
    # According to https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
    if res["TP"] == 0:
        if res["FP"] == 0 and res["FN"] == 0:
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        else:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
    else:
        precision = 1.0 * res["TP"] / (res["TP"] + res["FP"])
        recall = 1.0 * res["TP"] / (res["TP"] + res["FN"])
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def gen_micro_macro_result(res):
    precision = []
    recall = []
    f1 = []
    total = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for a in range(0, len(res)):
        total["TP"] += res[a]["TP"]
        total["FP"] += res[a]["FP"]
        total["FN"] += res[a]["FN"]
        total["TN"] += res[a]["TN"]

        p, r, f = get_prf(res[a])
        precision.append(p)
        recall.append(r)
        f1.append(f)

    micro_precision, micro_recall, micro_f1 = get_prf(total)

    macro_precision = 0
    macro_recall = 0
    macro_f1 = 0
    for a in range(0, len(f1)):
        macro_precision += precision[a]
        macro_recall += recall[a]
        macro_f1 += f1[a]

    macro_precision /= len(f1)
    macro_recall /= len(f1)
    macro_f1 /= len(f1)

    return {
        "micro_precision": round(micro_precision, 3),
        "micro_recall": round(micro_recall, 3),
        "micro_f1": round(micro_f1, 3),
        "macro_precision": round(macro_precision, 3),
        "macro_recall": round(macro_recall, 3),
        "macro_f1": round(macro_f1, 3)
    }


def null_accuracy_function(outputs, label, config, result=None):
    return None


def single_label_top1_accuracy(outputs, label, config, result=None):
    if result is None:
        result = []
    id1 = torch.max(outputs, dim=1)[1]
    # id2 = torch.max(label, dim=1)[1]
    id2 = label
    nr_classes = outputs.size(1)
    while len(result) < nr_classes:
        result.append({"TP": 0, "FN": 0, "FP": 0, "TN": 0})
    for a in range(0, len(id1)):
        # if len(result) < a:
        #    result.append({"TP": 0, "FN": 0, "FP": 0, "TN": 0})

        it_is = int(id1[a])
        should_be = int(id2[a])
        if it_is == should_be:
            result[it_is]["TP"] += 1
        else:
            result[it_is]["FP"] += 1
            result[should_be]["FN"] += 1

    return result


def multi_label_accuracy(outputs, label, config, result=None):
    if len(label[0]) != len(outputs[0]):
        raise ValueError('Input dimensions of labels and outputs must match.')

    outputs = outputs.data
    labels = label.data

    if result is None:
        result = []

    total = 0
    nr_classes = outputs.size(1)

    while len(result) < nr_classes:
        result.append({"TP": 0, "FN": 0, "FP": 0, "TN": 0})

    for i in range(nr_classes):
        outputs1 = (outputs[:, i] >= 0.5).long()
        labels1 = (labels[:, i].float() >= 0.5).long()
        total += int((labels1 * outputs1).sum())
        total += int(((1 - labels1) * (1 - outputs1)).sum())

        if result is None:
            continue

        # if len(result) < i:
        #    result.append({"TP": 0, "FN": 0, "FP": 0, "TN": 0})

        result[i]["TP"] += int((labels1 * outputs1).sum())
        result[i]["FN"] += int((labels1 * (1 - outputs1)).sum())
        result[i]["FP"] += int(((1 - labels1) * outputs1).sum())
        result[i]["TN"] += int(((1 - labels1) * (1 - outputs1)).sum())

    return result


def single_label_top2_accuracy(outputs, label, config, result=None):
    raise NotImplementedError
    # still bug here

    if result is None:
        result = []
        # print(label)

    id1 = torch.max(outputs, dim=1)[1]
    # id2 = torch.max(label, dim=1)[1]
    id2 = label
    nr_classes = outputs.size(1)
    while len(result) < nr_classes:
        result.append({"TP": 0, "FN": 0, "FP": 0, "TN": 0})
    for a in range(0, len(id1)):
        # if len(result) < a:
        #    result.append({"TP": 0, "FN": 0, "FP": 0, "TN": 0})

        it_is = int(id1[a])
        should_be = int(id2[a])
        if it_is == should_be:
            result[it_is]["TP"] += 1
        else:
            result[it_is]["FP"] += 1
            result[should_be]["FN"] += 1

    _, prediction = torch.topk(outputs, 2, 1, largest=True)
    prediction1 = prediction[:, 0:1]
    prediction2 = prediction[:, 1:]

    prediction1 = prediction1.view(-1)
    prediction2 = prediction2.view(-1)

    return result


def general_accuracy(metric, result, name):
    if result is None:
        result = {
            name: []
        }
    if result.get(name) is None:
        result[name] = []
    result[name].append(metric)
    return result


def ssim_accuracy(outputs, ground_truth, config):
    outputs = outputs.squeeze().detach().cpu().numpy()
    ground_truth = ground_truth.squeeze().detach().cpu().numpy()
    metric = ssim(outputs, ground_truth, data_range=1, channel_axis=0)
    return metric


def psnr_accuracy(outputs, ground_truth, config):
    outputs = outputs.squeeze().detach().cpu().numpy()
    ground_truth = ground_truth.squeeze().detach().cpu().numpy()
    metric = psnr(outputs, ground_truth, data_range=1)
    return metric


def general_image_metrics(outputs: torch.Tensor, ground_truth: torch.Tensor, config, result=None):
    if result is None:
        result = {
            "PSNR": [],
            "SSIM": []
        }
    if result.get("PSNR") is None:
        result["PSNR"] = []
    if result.get("SSIM") is None:
        result["SSIM"] = []
    psnr = psnr_accuracy(outputs, ground_truth, config)
    ssim = ssim_accuracy(outputs, ground_truth, config)
    result["PSNR"].append(psnr)
    result["SSIM"].append(ssim)
    return result
    # return {
    #     "
    #     # "PSNR": "{:<7.5}".format(sum(psnr_list) / len(psnr_list)),
    #     # "SSIM": "{:<7.5}".format(sum(ssim_list) / len(ssim_list))
    # }
