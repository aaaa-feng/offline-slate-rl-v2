"""
离线 RL Agent 抽象基类。

定义 train / select_action / save / load 接口。
"""

from abc import ABC, abstractmethod
from typing import Dict
import torch


class BaseOfflineAgent(ABC):
    """离线 RL Agent 抽象基类"""

    def __init__(self, config, device: torch.device):
        self.config = config
        self.device = device
        self.total_it = 0

    @abstractmethod
    def train(self, batch) -> Dict[str, float]:
        """单步训练，返回指标字典"""
        ...

    @abstractmethod
    def select_action(self, state: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """选择动作"""
        ...

    def save(self, path: str):
        """保存 checkpoint"""
        torch.save({
            'total_it': self.total_it,
            'state_dict': self.state_dict(),
        }, path)

    def load(self, path: str):
        """加载 checkpoint"""
        ckpt = torch.load(path, map_location=self.device)
        self.load_state_dict(ckpt['state_dict'])
        self.total_it = ckpt.get('total_it', 0)
        return self

    @abstractmethod
    def state_dict(self) -> dict:
        """返回模型权重"""
        ...

    @abstractmethod
    def load_state_dict(self, state_dict: dict):
        """加载模型权重"""
        ...
