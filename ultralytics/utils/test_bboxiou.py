
import torch
# import pytest
# from metrics import bbox_iou
import math


def SIoU_loss(box1, box2, theta=4):
    eps = 1e-7
    cx_pred = (box1[:, 0] + box1[:, 2]) / 2
    cy_pred = (box1[:, 1] + box1[:, 3]) / 2
    cx_gt = (box2[:, 0] + box2[:, 2]) / 2
    cy_gt = (box2[:, 1] + box2[:, 3]) / 2

    dist = ((cx_pred - cx_gt)**2 + (cy_pred - cy_gt)**2) ** 0.5
    ch = torch.max(cy_gt, cy_pred) - torch.min(cy_gt, cy_pred)
    x = ch / (dist + eps)

    angle = 1 - 2*torch.sin(torch.arcsin(x)-torch.pi/4)**2
    # distance cost
    xmin = torch.min(box1[:, 0], box2[:, 0])
    xmax = torch.max(box1[:, 2], box2[:, 2])
    ymin = torch.min(box1[:, 1], box2[:, 1])
    ymax = torch.max(box1[:, 3], box2[:, 3])
    cw = xmax - xmin
    ch = ymax - ymin
    px = ((cx_gt - cx_pred) / (cw+eps))**2
    py = ((cy_gt - cy_pred) / (ch+eps))**2
    gama = 2 - angle
    dis = (1 - torch.exp(-1 * gama * px)) + (1 - torch.exp(-1 * gama * py))

    #shape cost
    w_pred = box1[:, 2] - box1[:, 0]
    h_pred = box1[:, 3] - box1[:, 1]
    w_gt = box2[:, 2] - box2[:, 0]
    h_gt = box2[:, 3] - box2[:, 1]
    ww = torch.abs(w_pred - w_gt) / (torch.max(w_pred, w_gt) + eps)
    wh = torch.abs(h_gt - h_pred) / (torch.max(h_gt, h_pred) + eps)
    omega = (1 - torch.exp(-1 * wh)) ** theta + (1 - torch.exp(-1 * ww)) ** theta

    #IoU loss
    lt = torch.max(box1[..., :2], box2[..., :2])  # [B, rows, 2]
    rb = torch.min(box1[..., 2:], box2[..., 2:])  # [B, rows, 2]

    wh = torch.clamp_(rb - lt, min=0)
    overlap = wh[..., 0] * wh[..., 1]
    area1 = (box1[..., 2] - box1[..., 0]) * (
            box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (
            box2[..., 3] - box2[..., 1])
    iou = overlap / (area1 + area2 - overlap)

    SIoU = 1 - iou + (omega + dis) / 2
    return SIoU, iou

def bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    GIoU: bool = False,
    DIoU: bool = False,
    CIoU: bool = False,
    SIoU: bool = False,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Calculate the Intersection over Union (IoU) between bounding boxes.

    This function supports various shapes for `box1` and `box2` as long as the last dimension is 4.
    For instance, you may pass tensors shaped like (4,), (N, 4), (B, N, 4), or (B, N, 1, 4).
    Internally, the code will split the last dimension into (x, y, w, h) if `xywh=True`,
    or (x1, y1, x2, y2) if `xywh=False`.

    Args:
        box1 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
        box2 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format.
        GIoU (bool, optional): If True, calculate Generalized IoU.
        DIoU (bool, optional): If True, calculate Distance IoU.
        CIoU (bool, optional): If True, calculate Complete IoU.
        eps (float, optional): A small value to avoid division by zero.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if SIoU:
        # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
        s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5
        s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5
        sigma = (s_cw ** 2 + s_ch ** 2).sqrt() + eps
        angle_cost = (s_ch.abs() / sigma).sin().arcsin().pow(2).sin() * 2 - 1
        distance_cost = (1 - torch.exp(-angle_cost * (s_cw / ((b2_x1 - b1_x2).maximum(b1_x1 - b2_x2)) + eps)).pow(2)) + (1 - torch.exp(-angle_cost * (s_ch / ((b2_y1 - b1_y2).maximum(b1_y1 - b2_y2)) + eps)).pow(2))
        shape_cost = (1 - torch.exp(-((w1 - w2).abs() / (w1.maximum(w2)) + eps))).pow(4) + ( 1 - torch.exp(-((h1 - h2).abs() / (h1.maximum(h2)) + eps))).pow(4)
        return iou - (distance_cost + shape_cost) * 0.5

    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
            ) / 4  # center dist**2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU

def test_bbox_iou_xywh():
    # 测试 xywh 格式下的 IoU 计算
    # box1: (中心x, 中心y, 宽, 高)
    box1 = torch.tensor([[5, 5, 4, 4]], dtype=torch.float32)  # 一个中心在(5,5)，宽高为4的框
    box2 = torch.tensor([[5, 5, 4, 4]], dtype=torch.float32)  # 完全重合
    iou = bbox_iou(box1, box2, xywh=True)
    # 完全重合，IoU 应为 1
    assert torch.allclose(iou, torch.tensor([[1.0]]), atol=1e-6)

def test_bbox_iou_xyxy():
    # 测试 xyxy 格式下的 IoU 计算
    # box1: (左上x, 左上y, 右下x, 右下y)
    box1 = torch.tensor([[1, 1, 3, 3]], dtype=torch.float32)
    box2 = torch.tensor([[2, 2, 4, 4]], dtype=torch.float32)
    iou = bbox_iou(box1, box2, xywh=False)
    # 两个框有部分重叠，计算交并比
    # 交集面积为 1*1=1, 并集面积为 4+4-1=7
    assert torch.allclose(iou, torch.tensor([[1/7]]), atol=1e-6)

def test_bbox_iou_giou():
    # 测试 GIoU 计算
    box1 = torch.tensor([[1, 1, 3, 3]], dtype=torch.float32)
    box2 = torch.tensor([[2, 2, 4, 4]], dtype=torch.float32)
    giou = bbox_iou(box1, box2, xywh=False, GIoU=True)
    # GIoU 应小于普通 IoU
    iou = bbox_iou(box1, box2, xywh=False)
    assert giou < iou

def test_bbox_iou_diou():
    # 测试 DIoU 计算
    box1 = torch.tensor([[1, 1, 3, 3]], dtype=torch.float32)
    box2 = torch.tensor([[2, 2, 4, 4]], dtype=torch.float32)
    diou = bbox_iou(box1, box2, xywh=False, DIoU=True)
    iou = bbox_iou(box1, box2, xywh=False)
    # DIoU 应小于普通 IoU
    assert diou < iou

def test_bbox_iou_ciou():
    # 测试 CIoU 计算
    box1 = torch.tensor([[1, 1, 3, 3]], dtype=torch.float32)
    box2 = torch.tensor([[2, 2, 4, 4]], dtype=torch.float32)
    ciou = bbox_iou(box1, box2, xywh=False, CIoU=True)
    iou = bbox_iou(box1, box2, xywh=False)
    # CIoU 应小于普通 IoU
    assert ciou < iou

def test_bbox_iou_siou():
    # 测试 SIoU 计算
    box1 = torch.tensor([[1, 1, 3, 3]], dtype=torch.float32)
    box2 = torch.tensor([[2, 2, 4, 4]], dtype=torch.float32)
    siou = bbox_iou(box1, box2, xywh=False, SIoU=True)
    print(f"siou(使用自定义实现):{siou[0,0]}")
    # iou = bbox_iou(box1, box2, xywh=False)
    iou = bbox_iou(box1,box2, xywh=False, CIoU=True)
    print(f"iou:{iou[0,0]}")

    SIoU_test=SIoU_loss(box1, box2)
    print(f"siou(使用SIoU_loss实现):{SIoU_test[0][0]}")
    # SIoU 应小于普通 IoU
    # assert siou < iou

def test_bbox_iou_batch():
    # 测试批量输入
    box1 = torch.tensor([[1, 1, 3, 3], [0, 0, 2, 2]], dtype=torch.float32)
    box2 = torch.tensor([[2, 2, 4, 4], [1, 1, 3, 3]], dtype=torch.float32)
    iou = bbox_iou(box1, box2, xywh=False)
    print(iou.shape)
    # 检查输出形状
    assert iou.shape == (2, 2)

if __name__ == "__main__":
    # 运行测试
    test_bbox_iou_xywh()
    test_bbox_iou_xyxy()
    test_bbox_iou_giou()
    test_bbox_iou_diou()
    test_bbox_iou_ciou()
    test_bbox_iou_siou()
    print("All tests passed!")
