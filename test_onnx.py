import torch
import onnxruntime
import numpy as np
from utils.datasets import LoadImagesAndLabels
from utils.metrics import ap_per_class
from models.common import non_max_suppression, scale_coords

nc = 1


def load_data(dataset_path, img_size=640):
    dataset = LoadImagesAndLabels(dataset_path, img_size, batch_size=1, augment=False, hyp=None, rect=False,
                                  image_weights=False, cache_images=False)
    return dataset


def preprocess_image(img, img_size=640):
    img = torch.from_numpy(img).to(torch.float32)
    # img = img.permute(2, 0, 1)
    img /= 255.0
    return img.unsqueeze(0)


def infer_onnx_model(onnx_path, img_tensor):
    ort_session = onnxruntime.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: img_tensor.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs

def main():
    dataset_path = "./datasets/Anti-UAV-jiafang/"
    onnx_model_path = "./best.onnx"
    dataset = load_data(dataset_path)

    predictions, targets = [], []
    seen = 0
    for batch in dataset:
        img_tensor, target, img_path, shapes = batch
        img_tensor = preprocess_image(img_tensor.numpy())
        outputs = infer_onnx_model(onnx_model_path, img_tensor)
        outputs = torch.tensor(outputs[0])

        # 应用NMS
        outputs = non_max_suppression(outputs, conf_thres=0.001, iou_thres=0.65, classes=None, agnostic=False)
        outputs = scale_coords(img_tensor.shape[2:], outputs[0], shapes[0]).round()

        predictions.append(outputs)
        targets.append(target)
        seen += 1

    # 计算 mAP
    stats = []
    for i, pred in enumerate(predictions):
        labels = targets[i][:, 0]
        tbox = targets[i][:, 1:5]
        pbox = pred[:, :4]
        scores = pred[:, 4]
        pcls = pred[:, 5]
        stats.append((pbox, scores, pcls, labels))

    # ap, p, r = ap_per_class(*zip(*stats))
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    # print(f"mAP: {ap.mean()}")
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))


if __name__ == "__main__":
    main()
