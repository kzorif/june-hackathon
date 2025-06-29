import yaml
import math
import os
import cv2
import numpy as np

# Import MAX framework components
from max import engine, driver, nn
from max.dtype import DType
from max.graph import Graph, ops, TensorType, DeviceRef, Weight
from max.graph.ops import InterpolationMode

# Use torchvision for the highly optimized NMS algorithm.
try:
    import torch
    import torchvision

    _torch_available = True
except ImportError:
    _torch_available = False

# --- Configuration ---
MODEL_WEIGHTS = "./yolov8n.pt"
IMAGE_PATH = "./img/cat.jpg"
INPUT_SIZE = 640
CONF_THRESHOLD = 0.10
IOU_THRESHOLD = 0.30

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


# =================================================================================
# UTILITY FUNCTIONS (NumPy based)
# =================================================================================
def make_divisible(x, divisor):
    if isinstance(divisor, (np.ndarray, driver.Tensor)):
        divisor = int(np.max(divisor))
    return math.ceil(x / divisor) * divisor


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


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
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
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
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


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
    coords[:, 0] = np.clip(coords[:, 0], 0, img0_shape[1])
    coords[:, 1] = np.clip(coords[:, 1], 0, img0_shape[0])
    coords[:, 2] = np.clip(coords[:, 2], 0, img0_shape[1])
    coords[:, 3] = np.clip(coords[:, 3], 0, img0_shape[0])
    return coords


# =================================================================================
# MAX MODULES (GRAPH-BUILDING HELPERS)
# =================================================================================


class MaxConv:
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, name_prefix=""):
        # Note: Weights are defined in RSCF (H, W, I, O) layout for MAX's default NHWC conv
        self.weight = Weight(
            name=f"{name_prefix}.conv.weight",
            dtype=DType.float32,
            shape=[k, k, c1 // g, c2],
            device=DeviceRef.CPU(),
        )
        self.bias = Weight(
            name=f"{name_prefix}.conv.bias",
            dtype=DType.float32,
            shape=[c2],
            device=DeviceRef.CPU(),
        )
        self.s = s
        self.p = p if p is not None else autopad(k, p, d)
        self.act = ops.silu if act else lambda x: x

    def __call__(self, x):
        padding = (self.p, self.p, self.p, self.p)
        conv_out = ops.conv2d(x, self.weight, stride=(self.s, self.s), padding=padding)
        reshaped_bias = self.bias.to(x.device).reshape((1, 1, 1, -1))
        biased_out = conv_out + reshaped_bias
        return self.act(biased_out)


class MaxSequential:
    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MaxDetect:
    def __init__(self, nc=80, ch=(), name_prefix=""):
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = max(ch[0], self.nc)

        self.cv2 = []
        self.cv3 = []
        for i, c1 in enumerate(ch):
            self.cv2.append(
                MaxSequential(
                    MaxConv(c1, c2, 3, name_prefix=f"{name_prefix}.cv2.{i}.0"),
                    MaxConv(c2, c2, 3, name_prefix=f"{name_prefix}.cv2.{i}.1"),
                    MaxConv(
                        c2,
                        4 * self.reg_max,
                        1,
                        act=False,
                        name_prefix=f"{name_prefix}.cv2.{i}.2",
                    ),
                )
            )
            self.cv3.append(
                MaxSequential(
                    MaxConv(c1, c3, 3, name_prefix=f"{name_prefix}.cv3.{i}.0"),
                    MaxConv(c3, c3, 3, name_prefix=f"{name_prefix}.cv3.{i}.1"),
                    MaxConv(
                        c3,
                        self.nc,
                        1,
                        act=False,
                        name_prefix=f"{name_prefix}.cv3.{i}.2",
                    ),
                )
            )

    def __call__(self, x_list):
        outputs = []
        for i in range(self.nl):
            x_i_nhwc = x_list[i]

            # Apply convolutions (which expect NHWC input)
            box_head = self.cv2[i](x_i_nhwc)
            cls_head = self.cv3[i](x_i_nhwc)

            # Transpose results to NCHW for concatenation along the channel axis
            box_head_nchw = box_head.permute([0, 3, 1, 2])
            cls_head_nchw = cls_head.permute([0, 3, 1, 2])

            outputs.append(ops.concat([box_head_nchw, cls_head_nchw], axis=1))
        return outputs


class MaxBottleneck:
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, name_prefix=""):
        c_ = int(c2 * e)
        self.cv1 = MaxConv(c1, c_, k=3, s=1, name_prefix=f"{name_prefix}.cv1")
        self.cv2 = MaxConv(c_, c2, k=3, s=1, g=g, name_prefix=f"{name_prefix}.cv2")
        self.add = shortcut and c1 == c2

    def __call__(self, x):
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class MaxC2f:
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, name_prefix=""):
        self.c = int(c2 * e)
        self.cv1 = MaxConv(c1, 2 * self.c, k=1, s=1, name_prefix=f"{name_prefix}.cv1")
        self.cv2 = MaxConv(
            (2 + n) * self.c, c2, k=1, s=1, name_prefix=f"{name_prefix}.cv2"
        )
        self.m = [
            MaxBottleneck(
                self.c, self.c, shortcut, g, e=1.0, name_prefix=f"{name_prefix}.m.{i}"
            )
            for i in range(n)
        ]

    def __call__(self, x):
        y = ops.split(self.cv1(x), [self.c, self.c], axis=3)
        y_out = [y[0], y[1]]
        m_in = y[1]
        for bottleneck in self.m:
            m_in = bottleneck(m_in)
            y_out.append(m_in)
        return self.cv2(ops.concat(y_out, axis=3))


class MaxSPPF:
    def __init__(self, c1, c2, k=5, name_prefix=""):
        c_ = c1 // 2
        self.cv1 = MaxConv(c1, c_, k=1, s=1, name_prefix=f"{name_prefix}.cv1")
        self.cv2 = MaxConv(c_ * 4, c2, k=1, s=1, name_prefix=f"{name_prefix}.cv2")
        self.k = k

    def __call__(self, x):
        x = self.cv1(x)
        pool_args = {
            "kernel_size": (self.k, self.k),
            "padding": self.k // 2,
            "stride": 1,
        }
        x_nchw = x.permute([0, 3, 1, 2])
        y1 = ops.max_pool2d(x_nchw, **pool_args)
        y2 = ops.max_pool2d(y1, **pool_args)
        y3 = ops.max_pool2d(y2, **pool_args)
        concatenated = ops.concat([x_nchw, y1, y2, y3], axis=1)
        return self.cv2(concatenated.permute([0, 2, 3, 1]))


class DetectionModelMAX(nn.Module):
    def __init__(self, cfg, ch=3, nc=None, device=None):
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else self._load_yaml(cfg)
        if nc and nc != self.yaml["nc"]:
            self.yaml["nc"] = nc
        self.device = device or (
            driver.Accelerator() if driver.accelerator_count() > 0 else driver.CPU()
        )
        self.model_graph = None
        self.compiled_model = None
        self.fused_weights = {}
        self.reg_max = 16
        self.no = self.yaml["nc"] + self.reg_max * 4
        self.nl = len(self.yaml["head"][-1][2][0])
        self.stride = np.array([8.0, 16.0, 32.0])

    def build_graph(self, ch=3, img_size=640):
        print("Building MAX Graph from YAML...")
        input_type = TensorType(
            DType.float32,
            [1, img_size, img_size, ch],
            device=DeviceRef.from_device(self.device),
        )

        with Graph("YOLOv8", input_types=[input_type]) as graph:
            layers, c2 = {}, ch
            x = graph.inputs[0]

            for i, (f, n, m, args) in enumerate(
                self.yaml["backbone"] + self.yaml["head"]
            ):
                m_str = m
                n = (
                    max(round(n * self.yaml.get("depth_multiple", 1.0)), 1)
                    if n > 1
                    else n
                )

                module_class_map = {
                    "Conv": MaxConv,
                    "C2f": MaxC2f,
                    "SPPF": MaxSPPF,
                    "Bottleneck": MaxBottleneck,
                }

                if m_str in module_class_map:
                    c1 = c2 if f == -1 else layers[f].shape[3]
                    c2 = make_divisible(
                        args[0] * self.yaml.get("width_multiple", 1.0), 8
                    )
                    module_args = [c1, c2] + list(args[1:])
                    if m_str == "C2f":
                        module_args.insert(2, n)
                    module = module_class_map[m_str](
                        *module_args, name_prefix=f"model.{i}"
                    )
                    x = module(x)
                elif m_str == "nn.Upsample":
                    scale_factor = float(args[1])
                    x_nchw = x.permute([0, 3, 1, 2])
                    n_dim, c_dim, h_dim, w_dim = x_nchw.shape
                    new_h, new_w = (
                        int(int(h_dim) * scale_factor),
                        int(int(w_dim) * scale_factor),
                    )
                    new_shape_list = [n_dim, c_dim, new_h, new_w]
                    resized = ops.resize(
                        x_nchw, new_shape_list, interpolation=InterpolationMode.BICUBIC
                    )
                    x = resized.permute([0, 2, 3, 1])
                    c2 = x.shape[3]
                elif m_str == "Concat":
                    inputs_to_concat = [layers[j] if j != -1 else x for j in f]
                    x = ops.concat(inputs_to_concat, axis=3)
                    c2 = x.shape[3]
                elif m_str == "Detect":
                    ch_detect = [layers[j].shape[3] for j in f]
                    detect_module = MaxDetect(
                        nc=self.yaml["nc"], ch=ch_detect, name_prefix=f"model.{i}"
                    )
                    detect_inputs = [layers[j] for j in f]
                    final_outputs = detect_module(detect_inputs)
                    graph.output(*final_outputs)
                    break
                else:
                    raise NotImplementedError(f"Module {m_str} not implemented.")

                layers[i] = x

        self.model_graph = graph
        print("MAX Graph built successfully.")

    def compile(self, session: engine.InferenceSession):
        if self.model_graph is None:
            self.build_graph()
        self.compiled_model = session.load(
            self.model_graph, weights_registry=self.fused_weights
        )
        print("Model compiled successfully.")

    def __call__(self, x: driver.Tensor):
        if self.compiled_model is None:
            raise RuntimeError("Model is not compiled. Call .compile(session) first.")
        raw_outputs = self.compiled_model.execute(x)
        raw_outputs_np = [t.to_numpy() for t in raw_outputs]
        return self.forward_postprocess(raw_outputs_np)

    def forward_postprocess(self, p):
        anchors, strides = self._make_anchors_numpy(p, self.stride, 0.5)
        x = [xi.reshape(xi.shape[0], self.no, -1) for xi in p]
        x_cat = np.concatenate(x, axis=2)
        box, cls = np.split(x_cat, [self.reg_max * 4], axis=1)

        # Calculate bounding boxes
        dbox = dist2bbox_numpy(dfl_numpy(box), anchors[np.newaxis, ...], xywh=True)
        dbox *= strides[np.newaxis, ...]

        # Calculate class scores and transpose to the correct shape
        cls = 1 / (1 + np.exp(-cls))
        cls = cls.transpose(0, 2, 1)  # Shape becomes (batch, anchors, classes)

        # Concatenate dbox (1, 8400, 4) and cls (1, 8400, 80) along the last axis
        y = np.concatenate((dbox, cls), axis=-1)

        # Return the result in the correct shape (batch, num_detections, 4+classes)
        return y

    @staticmethod
    def _make_anchors_numpy(feats, strides, grid_cell_offset=0.5):
        anchor_points, stride_tensor = [], []
        for i, stride in enumerate(strides):
            _, _, h, w = feats[i].shape
            sy, sx = np.meshgrid(
                np.arange(h, dtype=np.float32) + grid_cell_offset,
                np.arange(w, dtype=np.float32) + grid_cell_offset,
                indexing="ij",
            )
            anchor_points.append(np.stack((sx, sy), -1).reshape(-1, 2))
            stride_tensor.append(np.full((h * w, 1), stride, dtype=np.float32))
        return np.concatenate(anchor_points), np.concatenate(stride_tensor)

    def load_and_fuse_state_dict(self, state_dict):
        if not _torch_available:
            raise ImportError("PyTorch is required for loading and fusing weights.")

        # We build a temporary dictionary and assign it at the end.
        fused_weights_temp = {}

        # We iterate through all weights in the loaded PyTorch state_dict
        for k, v in state_dict.items():
            # Skip all Batch-Norm parameters, as they are fused into the preceding convolution.
            if ".bn." in k:
                continue

            # Skip the DFL layer from the YOLO head, as it's handled in post-processing.
            if k.endswith(".dfl.conv.weight"):
                continue

            # We trigger our fusion logic only on convolution ".weight" keys.
            if k.endswith(".weight"):
                # The original key from the .pt file is `k`.
                # We must determine the `target_key` that the MAX graph expects.
                target_key = k

                # --- Start of Corrected Remapping Logic for Detect Head ---
                # PyTorch keys like 'model.22.cv2.0.2.weight' need to become
                # 'model.22.cv2.0.2.conv.weight' to match MaxConv's naming.
                parts = k.split(".")
                # This pattern identifies all conv layers within the Detect head's branches.
                if len(parts) > 4 and parts[-4].startswith("cv"):
                    # Insert '.conv' before '.weight' to create the correct MAX key.
                    target_key = ".".join(parts[:-1] + ["conv", parts[-1]])
                # --- End of Corrected Remapping Logic ---

                # The target bias key must match the target weight key's structure.
                target_bias_key = target_key.replace(".weight", ".bias")

                # Determine the prefix of the ORIGINAL key `k` to find its BN layer.
                bn_prefix = (
                    ".".join(k.split(".")[:-2])
                    if ".conv." in k
                    else ".".join(k.split(".")[:-1])
                )

                # Check if a corresponding Batch-Norm layer exists for fusion.
                bn_weight_key = f"{bn_prefix}.bn.weight"
                if bn_weight_key in state_dict:
                    # --- FUSE Conv + BN ---
                    conv_w, bn_w, bn_b, bn_rm, bn_rv = (
                        v,
                        state_dict[f"{bn_prefix}.bn.weight"],
                        state_dict[f"{bn_prefix}.bn.bias"],
                        state_dict[f"{bn_prefix}.bn.running_mean"],
                        state_dict[f"{bn_prefix}.bn.running_var"],
                    )
                    eps = 1e-5
                    scale = bn_w / torch.sqrt(bn_rv + eps)
                    fused_w = conv_w * scale.view(-1, 1, 1, 1)
                    fused_b = bn_b - bn_rm * scale

                    fused_weights_temp[target_key] = np.ascontiguousarray(
                        fused_w.numpy().transpose(2, 3, 1, 0)
                    )
                    fused_weights_temp[target_bias_key] = fused_b.numpy()
                else:
                    # --- This is a Conv layer WITHOUT a BN layer ---
                    fused_weights_temp[target_key] = np.ascontiguousarray(
                        v.numpy().transpose(2, 3, 1, 0)
                    )

                    # Check if this conv layer has its own explicit bias.
                    original_bias_key = k.replace(".weight", ".bias")
                    if original_bias_key in state_dict:
                        fused_weights_temp[target_bias_key] = state_dict[
                            original_bias_key
                        ].numpy()
                    else:
                        # If no BN and no explicit bias, the MAX graph still needs one.
                        # Provide a zero-filled bias.
                        num_out_channels = v.shape[0]
                        fused_weights_temp[target_bias_key] = np.zeros(
                            num_out_channels, dtype=np.float32
                        )

        # Now that all weights are processed, assign the completed dictionary.
        self.fused_weights = fused_weights_temp

    @staticmethod
    def _load_yaml(path):
        with open(path, "r", errors="ignore") as f:
            return yaml.safe_load(f)


def load_yolo_model_from_pt(weights_path):
    print(f"Loading weights from {weights_path}...")
    if not _torch_available:
        raise ImportError("Loading .pt files requires PyTorch.")

    ckpt = torch.load(
        weights_path, map_location=torch.device("cpu"), weights_only=False
    )
    model_cfg = ckpt["model"].yaml
    max_model = DetectionModelMAX(cfg=model_cfg)
    state_dict = ckpt["model"].float().state_dict()
    max_model.load_and_fuse_state_dict(state_dict)
    if hasattr(ckpt["model"], "stride"):
        max_model.stride = ckpt["model"].stride.numpy()
    print("Weights loaded and fused into MAX model shell.")
    return max_model


def dfl_numpy(x, reg_max=16):
    b, c, a = x.shape
    x_reshaped = x.reshape(b, 4, reg_max, a)
    e_x = np.exp(x_reshaped - np.max(x_reshaped, axis=2, keepdims=True))
    softmax_x = e_x / e_x.sum(axis=2, keepdims=True)
    conv_weight = np.arange(reg_max, dtype=np.float32).reshape(1, 1, -1, 1)
    return (softmax_x * conv_weight).sum(axis=2)


def dist2bbox_numpy(distance, anchor_points, xywh=True):
    # Split along the 'sides' dimension (axis=1)
    lt, rb = np.split(distance, 2, axis=1)
    # Transpose lt and rb to align with anchor_points' shape: (1, 8400, 2)
    lt = lt.transpose(0, 2, 1)
    rb = rb.transpose(0, 2, 1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        # Concatenate along the last dimension (axis=-1) to get (..., 4)
        return np.concatenate((c_xy, wh), axis=-1)
    return np.concatenate((x1y1, x2y2), axis=-1)


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    max_det=300,
):
    if not _torch_available:
        raise ImportError("`non_max_suppression` requires PyTorch and Torchvision.")
    if isinstance(prediction, np.ndarray):
        prediction = torch.from_numpy(prediction)

    bs = prediction.shape[0]
    nc = prediction.shape[2] - 4
    xc = prediction[..., 4] > conf_thres
    output = [torch.zeros((0, 6))] * bs

    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue

        box = torch.from_numpy(xywh2xyxy(x[:, :4].numpy()))

        if multi_label:
            i, j = (x[:, 4:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 4, None], j[:, None].float()), 1)
        else:
            conf, j = x[:, 4:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes)).any(1)]
        if not x.shape[0]:
            continue

        c = x[:, 5:6] * (0 if agnostic else 7680)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        i = i[:max_det]
        output[xi] = x[i]

    return [o.numpy() for o in output]


def main():
    if not os.path.exists(MODEL_WEIGHTS):
        print(
            f"Error: Model weights not found at {MODEL_WEIGHTS}. Please download yolov8n.pt"
        )
        return
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image not found at {IMAGE_PATH}. Creating a dummy black image.")
        cv2.imwrite(IMAGE_PATH, np.zeros((480, 640, 3), dtype=np.uint8))

    max_model = load_yolo_model_from_pt(MODEL_WEIGHTS)
    session = engine.InferenceSession()
    max_model.compile(session)

    original_image = cv2.imread(IMAGE_PATH)
    annotated_image = original_image.copy()
    image_resized, ratio, pad = letterbox(
        original_image, (INPUT_SIZE, INPUT_SIZE), auto=False
    )

    img_tensor_np = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    img_tensor_np = np.ascontiguousarray(img_tensor_np)
    img_tensor_np = (img_tensor_np.astype(np.float32) / 255.0)[np.newaxis, ...]

    print("Running inference with MAX Engine...")
    max_tensor = driver.Tensor.from_numpy(img_tensor_np).to(max_model.device)
    processed_output = max_model(max_tensor)

    final_detections = non_max_suppression(
        processed_output, CONF_THRESHOLD, IOU_THRESHOLD
    )[0]

    print("\n--- Inference Results ---")
    if len(final_detections):
        scaled_coords = scale_coords(
            (INPUT_SIZE, INPUT_SIZE),
            final_detections[:, :4],
            original_image.shape[:2],
            (ratio, pad),
        )
        print(f"Detected {len(final_detections)} objects:")
        for i, (*xyxy, conf, cls) in enumerate(final_detections):
            class_id = int(cls)
            label = f"{COCO_CLASSES[class_id]} {conf:.2f}"
            print(f"  - {label} at {[int(c) for c in scaled_coords[i]]}")
            x1, y1, x2, y2 = map(int, scaled_coords[i])
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (lw, lh), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(
                annotated_image, (x1, y1 - lh - base), (x1 + lw, y1), (0, 255, 0), -1
            )
            cv2.putText(
                annotated_image,
                label,
                (x1, y1 - base),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )
    else:
        print("No objects detected.")

    cv2.imshow("YOLOv8 Inference with MAX Engine", annotated_image)
    print("\nPress any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
