<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->
<div align="center">
<img src="images/logo.svg" width="60%" alt="DeepSeek AI" />
</div>
<hr>
<div align="center">
<a href="https://www.deepseek.com/" target="_blank">
<img alt="Homepage" src="images/badge.svg" />
</a>
<a href="https://huggingface.co/spaces/deepseek-ai/deepseek-vl2-small" target="_blank">
<img alt="Chat" src="https://img.shields.io/badge/..."/> 
</div>

<p align="center">
<a href="https://github.com/deepseek-ai/DeepSeek-VL2/tree/main?tab=readme-ov-file#4-quick-start"><b>🚀 快速开始</b></a> |
<a href="https://github.com/deepseek-ai/DeepSeek-VL2/tree/main?tab=readme-ov-file#5-license"><b>📜 许可协议</b></a> |
<a href="https://github.com/deepseek-ai/DeepSeek-VL2/tree/main?tab=readme-ov-file#6-citation"><b>📖 引用文献</b></a> <br>
<a href="./DeepSeek_VL2_paper.pdf"><b>📄 技术白皮书</b></a> |
<a href="https://arxiv.org/abs/2412.10302"><b>📄 arXiv 论文</b></a> |
<a href="https://huggingface.co/spaces/deepseek-ai/deepseek-vl2-small"><b>🎮 在线演示</b></a>
</p>

## 1. 模型概述
DeepSeek-VL2 是新一代混合专家（MoE）视觉语言模型系列，在 DeepSeek-VL 基础上有显著提升。该系列在视觉问答、光学字符识别、文档/表格/图表解析及视觉定位等多项任务中展现卓越性能。模型系列包含三个版本：激活参数分别为 10 亿的 DeepSeek-VL2-Tiny、28 亿的 DeepSeek-VL2-Small 和 45 亿的 DeepSeek-VL2。

相较于同参数规模的开源密集型和 MoE 模型，DeepSeek-VL2 展现出具有竞争力的性能表现。[论文链接：DeepSeek-VL2: 面向高级多模态理解的混合专家视觉语言模型]()

**核心研发团队**：  
吴志宇*、陈晓康*、潘子政*、刘星超*、刘文**、戴大迈、高华卓、马一扬、吴成岳、王冰璇、谢振达、吴宇、胡凯、王佳伟、孙耀峰、李宇坤、朴艺诗、管康、刘爱新、谢鑫、游宇翔、董凯、于兴凯、张浩伟、赵亮、王义松、阮冲***  
（*共同贡献，**项目负责人，***通讯作者）

![](./images/vl2_teaser.jpeg)

## 2. 版本发布
🔹 **2025-2-6**：在 Huggingface Space 发布基础版 Gradio 演示 [deepseek-vl2-small](https://huggingface.co/spaces/deepseek-ai/deepseek-vl2-small)  
🔹 **2024-12-25**：新增 Gradio 演示案例、增量式预填充及 VLMEvalKit 支持  
🔹 **2024-12-13**：正式发布 DeepSeek-VL2 系列，包含 <code>DeepSeek-VL2-tiny</code>、<code>DeepSeek-VL2-small</code> 和 <code>DeepSeek-VL2</code>

## 3. 模型下载
我们正式发布 DeepSeek-VL2 系列模型，包括 <code>DeepSeek-VL2-tiny</code>、<code>DeepSeek-VL2-small</code> 和 <code>DeepSeek-VL2</code>，以促进学术与商业领域更广泛的研究应用。模型使用需遵守[许可条款](#8-许可协议)。

### Huggingface 模型库
| 模型名称 | 序列长度 | 下载链接 |
|--------------|-----------------|-----------------------------------------|
| DeepSeek-VL2-tiny | 4096 | [模型下载](https://huggingface.co/deepseek-ai/deepseek-vl2-tiny) |
| DeepSeek-VL2-small | 4096 | [模型下载](https://huggingface.co/deepseek-ai/deepseek-vl2-small) |
| DeepSeek-VL2 | 4096 | [模型下载](https://huggingface.co/deepseek-ai/deepseek-vl2) |

## 4. 快速使用指南

### 4.1 基础推理示例
```python
import torch
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

# 模型加载配置
model_path = "deepseek-ai/deepseek-vl2-tiny"
vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer
vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(torch.bfloat16).cuda().eval()

# 构建对话数据
conversation = [
    {"role": "<|User|>", "content": "<image>", "images": ["images/giraffe.jpeg"]},
    {"role": "<|Assistant|>", "content": ""}
]

# 图像预处理与模型推理
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(conversations=conversation, images=pil_images, force_batchify=True).to(vl_gpt.device)
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

# 生成响应
outputs = vl_gpt.language.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    max_new_tokens=512
)
print(tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False))
```

### 4.2 多图交互推理
**硬件要求**：运行 deepseek-vl2-small 需 80GB GPU 显存
```python
# 多图对话配置
conversation = [
    {
        "role": "<|User|>",
        "content": "This is image_1: <image>\nThis is image_2: <image>\nThis is image_3: <image>\n请描述图像内容",
        "images": ["images/multi_image_1.jpeg", "images/multi_image_2.jpeg", "images/multi_image_3.jpeg"]
    },
    {"role": "<|Assistant|>", "content": ""}
]

# 图像批处理与推理（处理逻辑与基础示例相同）
```

## 5. 增量预填充技术
**显存优化方案**：40GB GPU 运行指南
```python
# 启用增量预填充
with torch.no_grad():
    inputs_embeds, past_key_values = vl_gpt.incremental_prefilling(
        input_ids=prepare_inputs.input_ids,
        images=prepare_inputs.images,
        chunk_size=512  # 显存分块大小
    )
    
    # 增量生成响应
    outputs = vl_gpt.generate(
        inputs_embeds=inputs_embeds,
        past_key_values=past_key_values,
        max_new_tokens=512
    )
```

## 6. 完整部署方案
```shell
# 标准推理模式
CUDA_VISIBLE_DEVICES=0 python inference.py --model_path "deepseek-ai/deepseek-vl2"

# 显存优化模式（40GB GPU）
CUDA_VISIBLE_DEVICES=0 python inference.py --model_path "deepseek-ai/deepseek-vl2-small" --chunk_size 512
```

## 7. 交互式演示部署
```shell
# 安装依赖
pip install -e .[gradio]

# 不同型号部署指令
# Tiny 版（显存 <40GB）
CUDA_VISIBLE_DEVICES=2 python web_demo.py --model_name "deepseek-ai/deepseek-vl2-tiny" --port 37914

# Small 版（A100 40GB需启用预填充）
CUDA_VISIBLE_DEVICES=2 python web_demo.py --model_name "deepseek-ai/deepseek-vl2-small" --port 37914 --chunk_size 512

# 标准版部署
CUDA_VISIBLE_DEVICES=2 python web_demo.py --model_name "deepseek-ai/deepseek-vl2" --port 37914
```

## 8. 许可协议
本代码库遵循 [MIT 许可](./LICENSE-CODE)，模型使用需遵守 [DeepSeek 模型许可](./LICENSE-MODEL)。本系列模型支持商业应用，具体条款详见官方文档。

## 9. 学术引用
```bibtex
@misc{wu2024deepseekvl2,
    title={DeepSeek-VL2: 面向高级多模态理解的混合专家视觉语言模型},
    author={吴志宇 et al.},
    year={2024},
    eprint={2412.10302},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2412.10302}
}
```

## 10. 技术支持
如有技术问题，请通过以下方式联系我们：  
📮 邮箱: [service@deepseek.com](mailto:service@deepseek.com)  
📝 GitHub Issues: [问题提交页面](https://github.com/deepseek-ai/DeepSeek-VL2/issues)