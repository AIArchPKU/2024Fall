---
layout: page
title: Project
permalink: /project/
---
# Warm Up

我们的lab将使用verilog编写，请同学们提前配好本地编写verilog的环境，配环境教程请参考[Environment Setup Tutorial](/2024Fall/static_files/Lab/Verilog环境搭建教程.pdf)

作为Warm Up Session, 也为了让大家检查配置的环境是否正确，请大家在本地实现一个简单的decoder译码器模块并完成验证。


# Lab 1. MIPS Out-of-Order Super-scalar CPU

第一次Lab同学们需要实现一个基于MIPS指令集的OoO超标量CPU，具体任务描述可见:[Lab1](/2024Fall/static_files/Lab/Lab1/Lab_1_Manual.pdf),

Lab的开发在学校计算中心提供的[CLAB平台](https://clab.pku.edu.cn)进行，使用教程参考[CLAB使用教程](/2024Fall/static_files/Lab/Lab1/智能硬件体系结构.pdf)，感谢CLAB技术人员的支持！

实验采用Linux环境开发，运行lab所需的terminal相关指令参考[Linux环境运行Lab说明](/2024Fall/static_files/Lab/Lab1/Linux环境运行Lab说明.html)

CPU的不同模块存在项目./verilog目录下，具体功能请参照：[模块功能介绍](/2024Fall/static_files/Lab/Lab1/模块说明.html)

Lab所使用的MIPS指令集Manual简单版可见[MIPS Muanual Short](/2024Fall/static_files/Lab/Lab1/Instruction_Descriptions_Short.pdf)，相对完整版本可见[MIPS Muanual Complete](/2024Fall/static_files/Lab/Lab1/Instruction_Descriptions_Long.pdf),

**不同指令的格式可见**[Instruction Format](/2024Fall/static_files/Lab/Lab1/Instruction_Formats.pdf),具体对应opcode可以在项目目录下的'sys_defs.vh'找到

**实验结束时间为2024.11.25 23:59:59**，请大家注意按时提交代码 

# Lab 2. 基于CUDA/Triton的GPU算子设计

第二次Lab需要同学们使用CUDA或者Triton设计量化和稀疏的GPU算子，具体任务描述可见：[Lab2](/2024Fall/static_files/Lab/Lab2/Lab2_Manual.pdf),

实验平台使用Google的[Colab](https://colab.research.google.com/)，科学上网方法请参考[Wallespku](https://wallesspku.com/blog/)

开发GPU算子的入门教程可以参考[CUDA教程](https://developer.nvidia.com/blog/even-easier-introduction-cuda)和[Triton教程](https://triton-lang.org/main/getting-started/tutorials/index.html)

在Colab平台使用CUDA的基础编译工具链示例已经提供给大家，[cuda编译示例](/2024Fall/static_files/Lab/Lab2/cuda.rar)

算子模板定义请参考[templates](/2024Fall/static_files/Lab/Lab2/templates.zip)

**实验结束时间为2025.1.5 23:59:59**，请大家注意按时提交代码