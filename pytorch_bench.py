# NOT TO BE CHANGED, EXCEPT FOR CONFIGURATION. REQUIRED FOR BENCHMARK
import torch
import torch.nn as nn
import cv2
import numpy
import os
import math
import torchvision
from copy import deepcopy
import yaml

# --- Configuration ---
MODEL_WEIGHTS = "./yolov8n.pt"
IMAGE_PATH = "./person.jpg"
INPUT_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
# --- End Configuration ---


def letterbox(
    im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32
):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = numpy.mod(dw, stride), numpy.mod(dh, stride)
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return im, ratio, (dw, dh)


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else numpy.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    max_det=300,
):
    nc = prediction.shape[1] - 4
    xc = prediction[:, 4:].amax(1) > conf_thres
    bs = prediction.shape[0]
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):
        x = x.transpose(0, -1)
        x = x[xc[xi]]
        if not x.shape[0]:
            continue
        box = xywh2xyxy(x[:, :4])
        if multi_label:
            i, j = (x[:, 4:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 4, None], j[:, None].float()), 1)
        else:
            conf, j = x[:, 4:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
        if not x.shape[0]:
            continue
        c = x[:, 5:6] * (0 if agnostic else 7680)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        i = i[:max_det]
        output[xi] = x[i]
    return output


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (
            (img1_shape[1] - img0_shape[1] * gain) / 2,
            (img1_shape[0] - img0_shape[0] * gain) / 2,
        )
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, 0].clamp_(0, img0_shape[1])
    coords[:, 1].clamp_(0, img0_shape[0])
    coords[:, 2].clamp_(0, img0_shape[1])
    coords[:, 3].clamp_(0, img0_shape[0])
    return coords


def make_divisible(x, divisor):
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act
            if act is True
            else act
            if isinstance(act, nn.Module)
            else nn.Identity()
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 3, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(
            b, 4, a
        )


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


class Detect(nn.Module):
    stride = None

    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)
            )
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1))
            for x in ch
        )
        self.dfl = DFL(self.reg_max)

    def forward(self, x):
        shape = x[0].shape
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.training:
            return x

        self.anchors, self.strides = (
            x.transpose(0, 1) for x in self.make_anchors(x, self.stride, 0.5)
        )
        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        dbox = (
            dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1)
            * self.strides
        )
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y

    def make_anchors(self, feats, strides, grid_cell_offset=0.5):
        anchor_points, stride_tensor = [], []
        for i, stride in enumerate(strides):
            _, _, h, w = feats[i].shape
            sx = (
                torch.arange(w, device=feats[i].device, dtype=torch.float32)
                + grid_cell_offset
            )
            sy = (
                torch.arange(h, device=feats[i].device, dtype=torch.float32)
                + grid_cell_offset
            )
            sy, sx = torch.meshgrid(sy, sx, indexing="ij")
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(
                torch.full(
                    (h * w, 1), stride, device=feats[i].device, dtype=torch.float32
                )
            )
        return torch.cat(anchor_points), torch.cat(stride_tensor)

    def bias_init(self):
        pass


def parse_model(d, ch):
    nc = d["nc"]
    depth, width = d.get("depth_multiple", 1.0), d.get("width_multiple", 1.0)

    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        m_cls = eval(m) if isinstance(m, str) else m
        if m == "nn.Upsample":
            m_cls = nn.Upsample

        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a
            except (NameError, SyntaxError):
                pass

        n = max(round(n * depth), 1) if n > 1 else n

        if m_cls in [Conv, C2f, SPPF, Bottleneck]:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(c2 * width, 8)
            args = [c1, c2, *args[1:]]
            if m_cls is C2f:
                args.insert(2, n)
                n = 1
        elif m_cls is Concat:
            c2 = sum(ch[x] for x in f)
        elif m_cls is Detect:
            args.append([ch[x] for x in f])
        elif m_cls is nn.Upsample:
            c2 = ch[f]
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m_cls(*args) for _ in range(n))) if n > 1 else m_cls(*args)
        t = str(m_cls).split(".")[-1].replace("'>", "")
        m_.i, m_.f, m_.type = i, f, t
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)

    return nn.Sequential(*layers), sorted(save)


class DetectionModel(nn.Module):
    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None):
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else self.load_yaml(cfg)
        if nc and nc != self.yaml["nc"]:
            self.yaml["nc"] = nc
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])
        self.names = {i: f"class{i}" for i in range(self.yaml["nc"])}
        self.inplace = self.yaml.get("inplace", True)

        m = self.model[-1]
        if isinstance(m, Detect):
            s = 256
            m.stride = torch.tensor(
                [s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))]
            )
            self.stride = m.stride
            m.bias_init()
        else:
            self.stride = torch.Tensor([32])

    def forward(self, x):
        y = []
        for m in self.model:
            if m.f != -1:
                x = (
                    y[m.f]
                    if isinstance(m.f, int)
                    else [x if j == -1 else y[j] for j in m.f]
                )
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x

    @staticmethod
    def load_yaml(path):
        with open(path, "r", errors="ignore") as f:
            return yaml.safe_load(f)


def load_yolo_model_from_pt(weights_path, device):
    print(f"Loading PyTorch model from {weights_path}...")
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    model_cfg = ckpt["model"].yaml
    model = DetectionModel(cfg=model_cfg, ch=3, nc=model_cfg["nc"])
    state_dict = ckpt["model"].float().state_dict()
    model.load_state_dict(state_dict, strict=True)

    if hasattr(ckpt["model"], "stride"):
        model.stride = ckpt["model"].stride
        if hasattr(model.model[-1], "stride"):
            model.model[-1].stride = ckpt["model"].stride
    model.to(device)
    model.eval()
    print("PyTorch model loaded successfully.")
    return model


# Main execution block
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(MODEL_WEIGHTS):
        print(f"Error: Model weights not found at {MODEL_WEIGHTS}.")
        exit()
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image not found at {IMAGE_PATH}. Creating a dummy black image.")
        dummy_img = numpy.zeros((480, 640, 3), dtype=numpy.uint8)
        cv2.imwrite(IMAGE_PATH, dummy_img)

    model = load_yolo_model_from_pt(MODEL_WEIGHTS, device)

    original_image = cv2.imread(IMAGE_PATH)
    image_resized, ratio, pad = letterbox(original_image, INPUT_SIZE, auto=False)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image_rgb).to(device)
    image_tensor = image_tensor.float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        print("Running inference...")
        predictions = model(image_tensor)

    detections = non_max_suppression(predictions, CONF_THRESHOLD, IOU_THRESHOLD)[0]
    annotated_image = original_image.copy()

    if detections is not None and len(detections):
        detections[:, :4] = scale_coords(
            image_tensor.shape[2:],
            detections[:, :4],
            original_image.shape,
            ratio_pad=(ratio, pad),
        ).round()
        for *xyxy, conf, cls in reversed(detections):
            class_id = int(cls)
            label = f"{COCO_CLASSES[class_id]} {conf:.2f}"
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(
                annotated_image,
                (x1, y1 - label_height - baseline),
                (x1 + label_width, y1),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                annotated_image,
                label,
                (x1, y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )
    else:
        print("No objects detected.")

    cv2.imshow("YOLOv8 Inference (Pure PyTorch)", annotated_image)
    print("\nPress any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
