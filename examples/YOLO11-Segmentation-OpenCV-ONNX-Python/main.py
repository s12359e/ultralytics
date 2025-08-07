# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""YOLO11 segmentation inference using only OpenCV and NumPy."""

import argparse
from typing import Tuple, Union

import cv2
import numpy as np


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
        self.conf = conf
        self.iou = iou

    def __call__(self, img: np.ndarray):
        """Run inference on a single image."""
        prep, ratio, dwdh, _ = self.preprocess(img, self.imgsz)
        self.net.setInput(prep)
        preds, protos = self.net.forward(["output0", "output1"])
        return self.postprocess(img, preds, protos, ratio, dwdh)

    @staticmethod
    def letterbox(img: np.ndarray, new_shape: Tuple[int, int]):
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
        return img, r, (dw, dh), new_unpad

    def preprocess(self, img: np.ndarray, new_shape: Tuple[int, int]):
        """Prepare image for inference."""
        img, r, dwdh, new_unpad = self.letterbox(img, new_shape)
        img = img[..., ::-1].transpose(2, 0, 1)[None]
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        return img, r, dwdh, new_unpad

    def postprocess(self, img: np.ndarray, preds: np.ndarray, protos: np.ndarray, ratio: float, dwdh):
        """Process model outputs returning boxes, masks, class ids and confidences."""
        preds = np.squeeze(preds).T
        protos = np.squeeze(protos)
        nm = protos.shape[0]
        nc = preds.shape[1] - nm - 4
        boxes = preds[:, :4]
        scores = preds[:, 4 : 4 + nc]
        masks_in = preds[:, 4 + nc :]
        class_ids = scores.argmax(1)
        scores = scores[np.arange(scores.shape[0]), class_ids]
        keep = scores > self.conf
        boxes, scores, class_ids, masks_in = boxes[keep], scores[keep], class_ids[keep], masks_in[keep]
        if not boxes.size:
            return [], [], [], []

        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

        boxes_nms = boxes_xyxy.copy()
        boxes_nms[:, 2] -= boxes_nms[:, 0]
        boxes_nms[:, 3] -= boxes_nms[:, 1]
        indices = cv2.dnn.NMSBoxes(boxes_nms.tolist(), scores.tolist(), self.conf, self.iou)
        if len(indices):
            indices = np.array(indices).reshape(-1)
            boxes_xyxy = boxes_xyxy[indices]
            scores = scores[indices]
            class_ids = class_ids[indices]
            masks_in = masks_in[indices]
        else:
            return [], [], [], []

        boxes_xyxy = self.scale_boxes(boxes_xyxy, ratio, dwdh, img.shape)
        masks = self.process_mask(protos, masks_in, boxes_xyxy, img.shape, ratio, dwdh)
        return boxes_xyxy, masks, class_ids, scores

    @staticmethod
    def scale_boxes(boxes, ratio, dwdh, orig_shape):
        """Scale boxes from letterboxed shape back to original image shape."""
        boxes[:, [0, 2]] -= dwdh[0]
        boxes[:, [1, 3]] -= dwdh[1]
        boxes /= ratio
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, orig_shape[1])
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, orig_shape[0])
        return boxes

    def process_mask(self, protos, masks_in, boxes, orig_shape, ratio, dwdh):
        """Generate masks from mask coefficients and prototypes."""
        c, mh, mw = protos.shape
        masks = masks_in @ protos.reshape(c, -1)
        masks = 1 / (1 + np.exp(-masks))
        masks = masks.reshape(-1, mh, mw)
        masks_resized = []
        for mask in masks:
            mask = cv2.resize(mask, self.imgsz[::-1], interpolation=cv2.INTER_LINEAR)
            dh, dw = dwdh[1], dwdh[0]
            h, w = int(orig_shape[0] * ratio), int(orig_shape[1] * ratio)
            mask = mask[int(dh) : int(dh + h), int(dw) : int(dw + w)]
            mask = cv2.resize(mask, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR)
            masks_resized.append(mask)
        masks_resized = np.stack(masks_resized, axis=0)
        masks_final = []
        for mask, box in zip(masks_resized, boxes):
            x1, y1, x2, y2 = box.astype(int)
            m = np.zeros_like(mask, dtype=np.uint8)
            m[y1:y2, x1:x2] = (mask[y1:y2, x1:x2] > 0.5).astype(np.uint8)
            masks_final.append(m)
        return np.stack(masks_final, axis=0)

    def draw(self, img, boxes, masks, class_ids, scores):
        """Draw masks and boxes on image."""
        colors = np.random.RandomState(42).randint(0, 255, (len(set(class_ids)), 3))
        for i, box in enumerate(boxes):
            color = colors[class_ids[i] % len(colors)].tolist()
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            mask = masks[i].astype(bool)
            img[mask] = img[mask] * 0.5 + np.array(color) * 0.5
            label = f"{class_ids[i]}:{scores[i]:.2f}"
            cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return img


def main(onnx_model: str, source: str, conf: float = 0.25, iou: float = 0.7):
    """Run YOLO11 segmentation inference using OpenCV."""
    model = YOLO11Seg(onnx_model, conf, iou)
    img = cv2.imread(source)
    boxes, masks, class_ids, scores = model(img)
    if len(boxes):
        result = model.draw(img.copy(), boxes, masks, class_ids, scores)
        cv2.imshow("Segmented Image", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return boxes, masks, class_ids, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--source", type=str, required=True, help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    args = parser.parse_args()
    main(args.model, args.source, args.conf, args.iou)
