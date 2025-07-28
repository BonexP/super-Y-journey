import unittest
import torch
import pytest
from metrics import bbox_iou


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


    def test_bbox_iou_xywh(self):
        # 测试 xywh 格式下的 IoU 计算
        # box1: (中心x, 中心y, 宽, 高)
        box1 = torch.tensor([[5, 5, 4, 4]], dtype=torch.float32)  # 一个中心在(5,5)，宽高为4的框
        box2 = torch.tensor([[5, 5, 4, 4]], dtype=torch.float32)  # 完全重合
        iou = bbox_iou(box1, box2, xywh=True)
        # 完全重合，IoU 应为 1
        assert torch.allclose(iou, torch.tensor([[1.0]]), atol=1e-6)

    def test_bbox_iou_xyxy(self):
        # 测试 xyxy 格式下的 IoU 计算
        # box1: (左上x, 左上y, 右下x, 右下y)
        box1 = torch.tensor([[1, 1, 3, 3]], dtype=torch.float32)
        box2 = torch.tensor([[2, 2, 4, 4]], dtype=torch.float32)
        iou = bbox_iou(box1, box2, xywh=False)
        # 两个框有部分重叠，计算交并比
        # 交集面积为 1*1=1, 并集面积为 4+4-1=7
        assert torch.allclose(iou, torch.tensor([[1/7]]), atol=1e-6)

    def test_bbox_iou_giou(self):
        # 测试 GIoU 计算
        box1 = torch.tensor([[1, 1, 3, 3]], dtype=torch.float32)
        box2 = torch.tensor([[2, 2, 4, 4]], dtype=torch.float32)
        giou = bbox_iou(box1, box2, xywh=False, GIoU=True)
        # GIoU 应小于普通 IoU
        iou = bbox_iou(box1, box2, xywh=False)
        assert giou < iou

    def test_bbox_iou_diou(self):
        # 测试 DIoU 计算
        box1 = torch.tensor([[1, 1, 3, 3]], dtype=torch.float32)
        box2 = torch.tensor([[2, 2, 4, 4]], dtype=torch.float32)
        diou = bbox_iou(box1, box2, xywh=False, DIoU=True)
        iou = bbox_iou(box1, box2, xywh=False)
        # DIoU 应小于普通 IoU
        assert diou < iou

    def test_bbox_iou_ciou(self):
        # 测试 CIoU 计算
        box1 = torch.tensor([[1, 1, 3, 3]], dtype=torch.float32)
        box2 = torch.tensor([[2, 2, 4, 4]], dtype=torch.float32)
        ciou = bbox_iou(box1, box2, xywh=False, CIoU=True)
        iou = bbox_iou(box1, box2, xywh=False)
        # CIoU 应小于普通 IoU
        assert ciou < iou

    def test_bbox_iou_batch(self):
        # 测试批量输入
        box1 = torch.tensor([[1, 1, 3, 3], [0, 0, 2, 2]], dtype=torch.float32)
        box2 = torch.tensor([[2, 2, 4, 4], [1, 1, 3, 3]], dtype=torch.float32)
        iou = bbox_iou(box1, box2, xywh=False)
        # 检查输出形状
        assert iou.shape == (2, 2)

if __name__ == '__main__':
    unittest.main()
