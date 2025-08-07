# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YOLO11 segmentation inference using OpenCV and ONNX."""

import argparse
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch

import ultralytics.utils.ops as ops
from ultralytics.engine.results import Results
from ultralytics.utils import ASSETS, YAML
from ultralytics.utils.checks import check_yaml


class YOLO11Seg:
    """YOLO11 segmentation model for performing instance segmentation with OpenCV."""

    def __init__(
        self,
        onnx_model: str,
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: Union[int, Tuple[int, int]] = 640,
    ) -> None:
        """Initialise segmentation model with given ONNX weights."""
        self.net = cv2.dnn.readNetFromONNX(onnx_model)
        self.imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        self.classes = YAML.load(check_yaml("coco8.yaml"))["names"]
        self.conf = conf
        self.iou = iou

    def __call__(self, img: np.ndarray) -> List[Results]:
        """Run inference on a single image."""
        prep_img = self.preprocess(img, self.imgsz)
        self.net.setInput(prep_img)
        preds, protos = self.net.forward(["output0", "output1"])
        preds = torch.from_numpy(np.squeeze(preds).T).unsqueeze(0)
        protos = torch.from_numpy(protos)
        return self.postprocess(img, prep_img, [preds, protos])

    @staticmethod
    def letterbox(img: np.ndarray, new_shape: Tuple[int, int]) -> np.ndarray:
        """Resize and pad image while meeting stride-multiple constraints."""
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img

    def preprocess(self, img: np.ndarray, new_shape: Tuple[int, int]) -> np.ndarray:
        """Prepare image for inference."""
        img = self.letterbox(img, new_shape)
        img = img[..., ::-1].transpose([2, 0, 1])[None]
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        return img

    def postprocess(self, img: np.ndarray, prep_img: np.ndarray, outs: List[torch.Tensor]) -> List[Results]:
        """Process model outputs into `Results` objects."""
        preds, protos = outs
        preds = ops.non_max_suppression(preds, self.conf, self.iou, nc=len(self.classes))
        results = []
        for i, pred in enumerate(preds):
            if not len(pred):
                results.append(Results(img, path="", names=self.classes, boxes=pred[:, :6], masks=torch.zeros(0, *img.shape[:2])))
                continue
            pred[:, :4] = ops.scale_boxes(prep_img.shape[2:], pred[:, :4], img.shape)
            masks = self.process_mask(protos[i], pred[:, 6:], pred[:, :4], img.shape[:2])
            results.append(Results(img, path="", names=self.classes, boxes=pred[:, :6], masks=masks))
        return results

    def process_mask(
        self, protos: torch.Tensor, masks_in: torch.Tensor, bboxes: torch.Tensor, shape: Tuple[int, int]
    ) -> torch.Tensor:
        """Generate masks from mask coefficients and prototypes."""
        c, mh, mw = protos.shape
        masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
        masks = ops.scale_masks(masks[None], shape)[0]
        masks = ops.crop_mask(masks, bboxes)
        return masks.gt_(0.0)


def main(onnx_model: str, source: str, conf: float = 0.25, iou: float = 0.7) -> List[Results]:
    """Run YOLO11 segmentation inference using OpenCV."""
    model = YOLO11Seg(onnx_model, conf, iou)
    img = cv2.imread(source)
    results = model(img)
    cv2.imshow("Segmented Image", results[0].plot())
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--source", type=str, default=str(ASSETS / "bus.jpg"), help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    args = parser.parse_args()
    main(args.model, args.source, args.conf, args.iou)
