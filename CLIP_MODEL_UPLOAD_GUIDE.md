# CLIP 模型上传指南

## 上传位置

将手动下载的 CLIP 模型文件上传到以下目录：

```
/root/autodl-tmp/TIA2V/tacm/modules/cache/clip-vit-large-patch14/
```

## 必需文件列表

模型目录应包含以下文件：

### Tokenizer 文件
- `tokenizer_config.json`
- `vocab.json`
- `merges.txt`
- `special_tokens_map.json`（可选）

### Model 文件
- `config.json` （模型配置）
- `pytorch_model.bin` 或 `model.safetensors` （模型权重）

### Processor 文件
- `preprocessor_config.json` （图像预处理配置）

## 验证模型完整性

上传后，可以通过以下命令验证：

```bash
cd /root/autodl-tmp/TIA2V
python -c "
from transformers import CLIPProcessor, CLIPModel
import os

model_path = 'tacm/modules/cache/clip-vit-large-patch14'
if os.path.exists(model_path):
    try:
        processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
        model = CLIPModel.from_pretrained(model_path, local_files_only=True)
        print('✓ Model loaded successfully!')
        print(f'  Text encoder output dim: {model.text_model.config.hidden_size}')
        print(f'  Vision encoder output dim: {model.vision_model.config.hidden_size}')
    except Exception as e:
        print(f'✗ Error loading model: {e}')
else:
    print(f'✗ Directory not found: {model_path}')
"
```

## 替代方案：使用 HuggingFace 缓存

如果上传到项目目录不方便，也可以上传到 HuggingFace 的标准缓存位置：

```
~/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/[hash]/
```

然后将该目录的内容复制或链接到正确的 snapshot 目录。

## 注意

- 模型必须是 `clip-vit-large-patch14`（768 维），不能是 `clip-vit-base-patch32`（512 维）
- 必须与训练时使用的模型完全一致，否则生成的视频将没有语义内容
