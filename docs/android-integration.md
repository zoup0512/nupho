# Android 端侧接入文档

## 1. 概述

本项目导出的 ONNX 模型支持多任务推理，单次前向传播输出三个结果：

- **Domain**（意图域）：`camera` / `closure` / `unknown`
- **Camera Action**（镜头控制）：14 类单标签分类
- **Closure Action**（开合控制）：16 类多标签分类（8 目标 × 2 动作）

## 2. 交付物

将以下 3 个文件放入 Android `assets` 目录：

| 文件 | 说明 | 大小参考 |
|------|------|----------|
| `bert.onnx` | ONNX 模型 | ~400MB |
| `vocab.txt` | BERT 分词器词表 | ~100KB |
| `labels.json` | 标签映射 + 模型配置 | ~2KB |

### labels.json 结构

```json
{
  "id2label": {
    "0": "camera.switch_front",
    "1": "camera.switch_back",
    ...
  },
  "id2domain": {
    "0": "camera",
    "1": "closure",
    "2": "unknown"
  },
  "closure_id2label": {
    "0": "closure.open_left_front_door",
    "1": "closure.close_left_front_door",
    ...
  },
  "model_config": {
    "model_name": "hfl/chinese-macbert-base",
    "num_camera_labels": 14,
    "num_closure_labels": 16,
    "num_domains": 3,
    "max_length": 64
  }
}
```

> `id2label` 对应 camera 域的标签映射（即 `camera_id2label`）。

## 3. 依赖配置

### app/build.gradle

```groovy
dependencies {
    implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.18.0'
}
```

## 4. 模型输入输出

### 输入（3 个张量）

| 名称 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `input_ids` | int64 | `[1, seq_len]` | 分词后的 token ID 序列 |
| `attention_mask` | int64 | `[1, seq_len]` | 注意力掩码，实际 token 为 1，padding 为 0 |
| `token_type_ids` | int64 | `[1, seq_len]` | 句子类型 ID，单句全 0 |

`seq_len` 固定为 `model_config.max_length`（默认 64），不足补 0。

### 输出（3 个张量）

| 名称 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `domain_logits` | float32 | `[1, 3]` | 域分类 logits，经 softmax 得概率 |
| `camera_logits` | float32 | `[1, 14]` | 镜头分类 logits，经 softmax 得概率 |
| `closure_logits` | float32 | `[1, 16]` | 开合分类 logits，经 sigmoid 得概率 |

## 5. 推理流程

```
输入文本
  │
  ▼
分词（WordPiece）→ input_ids / attention_mask / token_type_ids
  │
  ▼
ONNX Runtime 推理 → domain_logits / camera_logits / closure_logits
  │
  ▼
解析 domain（argmax + softmax）
  │
  ├─ domain == "camera"  → 解析 camera_logits（argmax + softmax）→ 单标签
  ├─ domain == "closure" → 解析 closure_logits（sigmoid + 阈值 0.5）→ 多标签
  └─ domain == "unknown" → 无动作
```

## 6. 分词实现

Android 端需自行实现 WordPiece 分词（或使用 `vocab.txt` + 简单 Java 实现）：

### 6.1 加载词表

```kotlin
fun loadVocab(context: Context): Map<String, Int> {
    val vocab = mutableMapOf<String, Int>()
    context.assets.open("vocab.txt").bufferedReader().useLines { lines ->
        lines.forEachIndexed { index, line ->
            vocab[line.trim()] = index
        }
    }
    return vocab
}
```

### 6.2 WordPiece 分词

```kotlin
fun tokenize(
    text: String,
    vocab: Map<String, Int>,
    maxLen: Int = 64
): Triple<LongArray, LongArray, LongArray> {
    // 基础分词：按字符切分（中文）
    val tokens = mutableListOf<String>()
    tokens.add("[CLS]")
    for (ch in text) {
        tokens.add(ch.toString())
    }
    tokens.add("[SEP]")

    // 截断
    if (tokens.size > maxLen) {
        tokens.subList(maxLen - 1, tokens.size).clear()
        tokens.add("[SEP]")
    }

    // 转 ID
    val unkId = vocab["[UNK]"] ?: 100
    val padId = vocab["[PAD]"] ?: 0
    val inputIds = LongArray(maxLen)
    val attentionMask = LongArray(maxLen)
    val tokenTypeIds = LongArray(maxLen)

    for (i in 0 until maxLen) {
        if (i < tokens.size) {
            inputIds[i] = (vocab[tokens[i]] ?: unkId).toLong()
            attentionMask[i] = 1
            tokenTypeIds[i] = 0
        } else {
            inputIds[i] = padId.toLong()
            attentionMask[i] = 0
            tokenTypeIds[i] = 0
        }
    }

    return Triple(inputIds, attentionMask, tokenTypeIds)
}
```

> **注意**：上面的分词是最简实现。对于更准确的分词，建议参考 [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) 的 Android 移植版，或使用 `BertTokenizer` Java 实现。

## 7. ONNX 推理

### 7.1 初始化

```kotlin
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession

class NluModel(context: Context) {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val vocab: Map<String, Int>
    private val labels: JSONObject
    private val maxLen: Int

    init {
        // 加载 ONNX 模型
        val modelBytes = context.assets.open("bert.onnx").use { it.readBytes() }
        session = env.createSession(modelBytes)

        // 加载词表
        vocab = loadVocab(context)

        // 加载标签配置
        labels = JSONObject(
            context.assets.open("labels.json").bufferedReader().use { it.readText() }
        )
        maxLen = labels.getJSONObject("model_config").optInt("max_length", 64)
    }

    fun close() {
        session.close()
    }
}
```

### 7.2 执行推理

```kotlin
import org.json.JSONObject
import kotlin.math.exp

data class ActionResult(
    val target: String,
    val action: String,
    val confidence: Float
)

data class NluResult(
    val text: String,
    val domain: String,
    val domainConfidence: Float,
    val label: String? = null,        // camera 域
    val confidence: Float? = null,    // camera 域
    val actions: List<ActionResult>? = null  // closure 域
)

fun predict(context: Context, text: String): NluResult {
    val (inputIds, attentionMask, tokenTypeIds) = tokenize(text, vocab, maxLen)

    val shape = longArrayOf(1, maxLen.toLong())
    val inputs = mapOf(
        "input_ids" to OnnxTensor.createTensor(env, arrayOf(inputIds)),
        "attention_mask" to OnnxTensor.createTensor(env, arrayOf(attentionMask)),
        "token_type_ids" to OnnxTensor.createTensor(env, arrayOf(tokenTypeIds))
    )

    val outputs = session.run(inputs)
    val domainLogits = (outputs.get(0).value as Array<FloatArray>)[0]
    val cameraLogits = (outputs.get(1).value as Array<FloatArray>)[0]
    val closureLogits = (outputs.get(2).value as Array<FloatArray>)[0]

    // Domain: softmax + argmax
    val domainProbs = softmax(domainLogits)
    val domainId = domainProbs.indices.maxByOrNull { domainProbs[it] }!!
    val id2domain = labels.getJSONObject("id2domain")
    val domain = id2domain.getString(domainId.toString())

    return when (domain) {
        "camera" -> {
            val cameraProbs = softmax(cameraLogits)
            val predId = cameraProbs.indices.maxByOrNull { cameraProbs[it] }!!
            val id2label = labels.getJSONObject("id2label")
            NluResult(
                text = text,
                domain = domain,
                domainConfidence = domainProbs[domainId],
                label = id2label.getString(predId.toString()),
                confidence = cameraProbs[predId]
            )
        }
        "closure" -> {
            val id2closure = labels.getJSONObject("closure_id2label")
            val actions = mutableListOf<ActionResult>()
            for (i in closureLogits.indices) {
                val prob = sigmoid(closureLogits[i])
                if (prob > 0.5f) {
                    val labelStr = id2closure.getString(i.toString())
                    // 解析 "closure.open_left_front_door" -> action="open", target="left_front_door"
                    val parts = labelStr.removePrefix("closure.").split("_", limit = 2)
                    actions.add(ActionResult(
                        target = if (parts.size > 1) parts[1] else "",
                        action = parts[0],
                        confidence = prob
                    ))
                }
            }
            NluResult(
                text = text,
                domain = domain,
                domainConfidence = domainProbs[domainId],
                actions = actions
            )
        }
        else -> {
            NluResult(
                text = text,
                domain = domain,
                domainConfidence = domainProbs[domainId]
            )
        }
    }
}

private fun softmax(logits: FloatArray): FloatArray {
    val max = logits.max()
    val exps = logits.map { exp(it - max) }
    val sum = exps.sum()
    return exps.map { it / sum }.toFloatArray()
}

private fun sigmoid(x: Float): Float = 1.0f / (1.0f + exp(-x))
```

## 8. 输出示例

### Camera 域

```json
{
  "text": "看一下车头",
  "domain": "camera",
  "domain_confidence": 0.98,
  "label": "camera.switch_front",
  "confidence": 0.95
}
```

### Closure 域（多标签）

```json
{
  "text": "打开左前门和引擎盖",
  "domain": "closure",
  "domain_confidence": 0.97,
  "actions": [
    {"target": "left_front_door", "action": "open", "confidence": 0.93},
    {"target": "hood", "action": "open", "confidence": 0.91}
  ]
}
```

### Unknown 域

```json
{
  "text": "今天天气怎么样",
  "domain": "unknown",
  "domain_confidence": 0.89
}
```

## 9. 标签说明

### Camera 标签（14 类）

| ID | 标签 | 含义 |
|----|------|------|
| 0 | camera.switch_front | 切换前视角 |
| 1 | camera.switch_back | 切换后视角 |
| 2 | camera.switch_left | 切换左视角 |
| 3 | camera.switch_right | 切换右视角 |
| 4 | camera.switch_top | 切换俯视 |
| 5 | camera.switch_interior | 切换内饰 |
| 6 | camera.switch_default | 恢复默认视角 |
| 7 | camera.zoom_in | 放大 |
| 8 | camera.zoom_out | 缩小 |
| 9 | camera.adjust_left | 左调 |
| 10 | camera.adjust_right | 右调 |
| 11 | camera.adjust_up | 上调 |
| 12 | camera.adjust_down | 下调 |
| 13 | camera.adjust_around | 环绕 |

### Closure 标签（16 类，多标签）

| ID | 标签 | target | action |
|----|------|--------|--------|
| 0 | closure.open_left_front_door | left_front_door | open |
| 1 | closure.close_left_front_door | left_front_door | close |
| 2 | closure.open_right_front_door | right_front_door | open |
| 3 | closure.close_right_front_door | right_front_door | close |
| 4 | closure.open_left_rear_door | left_rear_door | open |
| 5 | closure.close_left_rear_door | left_rear_door | close |
| 6 | closure.open_right_rear_door | right_rear_door | open |
| 7 | closure.close_right_rear_door | right_rear_door | close |
| 8 | closure.open_all_doors | all_doors | open |
| 9 | closure.close_all_doors | all_doors | close |
| 10 | closure.open_hood | hood | open |
| 11 | closure.close_hood | hood | close |
| 12 | closure.open_trunk | trunk | open |
| 13 | closure.close_trunk | trunk | close |
| 14 | closure.open_fuel_cap | fuel_cap | open |
| 15 | closure.close_fuel_cap | fuel_cap | close |

## 10. 性能参考

| 指标 | 参考值 |
|------|--------|
| 模型大小 | ~400MB |
| 输入长度 | 64 tokens |
| CPU 推理延迟 | ~50-100ms |
| 内存占用 | ~500MB |

> 实际性能取决于设备型号和线程配置。建议在目标设备上实测。

## 11. 注意事项

- **阈值**：closure 域的 sigmoid 概率阈值为 **0.5**，可根据实际效果调整
- **输入长度**：固定为 `max_length`（默认 64），超出截断，不足补 padding
- **线程数**：可通过 `OrtSession.SessionOptions().setIntraOpNumThreads(n)` 调整
- **模型加载**：建议在子线程中初始化 `OrtSession`，避免阻塞 UI
- **资源释放**：`OrtSession` 和 `OnnxTensor` 使用完毕后需调用 `close()`
