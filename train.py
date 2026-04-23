#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练入口脚本
确保在正确的工作目录下运行训练流程
"""

import os

# 确保工作目录是项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))
os.chdir(project_root)

# 导入训练模块并执行
from model.model_v1.tools.train import main

if __name__ == "__main__":
    main()
