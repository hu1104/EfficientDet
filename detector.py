import cv2
import numpy as np
import torch
from torch.backends import cudnn
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess
from utils.utils import xyxy_to_xywh


class EfficientDet(object):
    obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', '', 'stop sign', 'parking meter', 'bench',
                'bird', 'cat', 'dog', 'horse', 'sheep',
                'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack',
                'umbrella', '', '', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', '',
                'wine glass', 'cup', 'fork', 'knife', 'spoon',
                'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
                'carrot', 'hot dog', 'pizza', 'donut',
                'cake', 'chair', 'couch', 'potted plant', 'bed', '',
                'dining table', '', '', 'toilet', '', 'tv',
                'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', '', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush']

    def __init__(self, weightfile, score_thresh,
                 nms_thresh, is_xywh=True, use_cuda=True, use_float16=False):
        print('Loading weights from %s... Done!' % (weightfile))

        # constants
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.use_cuda = use_cuda
        self.is_xywh = is_xywh

        compound_coef = 0
        force_input_size = None  # set None to use default size

        self.use_float16 = False
        cudnn.fastest = True
        cudnn.benchmark = True

        # tf bilinear interpolation is different from any other's, just make do
        input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        self.input_size = input_sizes[compound_coef] if \
            force_input_size is None else force_input_size

        # load model
        self.model = EfficientDetBackbone(compound_coef=compound_coef,
                                          num_classes=len(self.obj_list))
        # f'weights/efficientdet-d{compound_coef}.pth'
        self.model.load_state_dict(torch.load(weightfile))
        self.model.requires_grad_(False)
        self.model.eval()

        if self.use_cuda:
            self.model = self.model.cuda()
        if self.use_float16:
            self.model = self.model.half()

        # Box
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()

    def __call__(self, imgs):
        # frame preprocessing
        _, framed_imgs, framed_metas = preprocess(imgs,
                                                  max_size=self.input_size)

        if self.use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

        dtype = torch.float32 if not self.use_float16 else torch.float16
        x = x.to(dtype).permute(0, 3, 1, 2)

        # model predict
        with torch.no_grad():
            features, regression, classification, anchors = self.model(x)

            out = postprocess(x,
                              anchors, regression, classification,
                              self.regressBoxes, self.clipBoxes,
                              self.score_thresh, self.nms_thresh)

        # result
        out = invert_affine(framed_metas, out)

        if len(out) == 0:
            return None, None, None

        rois = [o['rois'] for o in out]
        scores = [o['scores'] for o in out]
        class_ids = [o['class_ids'] for o in out]
        if self.is_xywh:
            return xyxy_to_xywh(rois), scores, class_ids
        else:
            return rois, scores, class_ids


def build_detector(cfg, use_cuda):
    return EfficientDet(cfg.EfficientDet.WEIGHT,
                        score_thresh=cfg.EfficientDet.SCORE_THRESH,
                        nms_thresh=cfg.EfficientDet.NMS_THRESH,
                        is_xywh=False,
                        use_cuda=use_cuda)


if __name__ == "__main__":
    compound_coef = 0

    from easydict import EasyDict as edict
    d = edict({
        'EfficientDet': {
            'WEIGHT': "./weights/efficientdet-d0.pth",
            'SCORE_THRESH': 0.2,
            'NMS_THRESH': 0.2
        }
    })
    detector = build_detector(d, True)

    img_path = 'test/img.png'

    res = detector(img_path)

    def display(preds, imgs, imshow=True, imwrite=False):
        rois, scores, class_ids = preds

        for i in range(len(imgs)):
            if len(rois[i]) == 0:
                continue

            for j in range(len(rois[i])):
                (x1, y1, x2, y2) = rois[i][j].astype(np.int)
                cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
                obj = EfficientDet.obj_list[class_ids[i][j]]
                score = float(scores[i][j])

                cv2.putText(imgs[i], '{}, {:.3f}'.format(obj, score),
                            (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 0), 1)

            if imshow:
                cv2.imshow('img', imgs[i])
                cv2.waitKey(0)

            if imwrite:
                cv2.imwrite(f'test/img_inferred_d{compound_coef}_this_repo_{i}.jpg', imgs[i])

    ori_imgs = [cv2.imread(img_path)]
    display(res, imgs=ori_imgs, imshow=True, imwrite=False)
