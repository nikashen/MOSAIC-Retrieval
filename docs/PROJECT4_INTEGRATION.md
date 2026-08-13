# 与项目四 TRACE-Rec 的冷启动接口

项目四的 `HybridItemEncoder` 能接收 `dense_content_features`；MOSAIC 输出的是
L2-normalized 的多模态 `content_vector`。契约在
`integrations/project4_content_vector.py`，字段为：

```text
video_id:       int64
content_vector: float32 [dimension], L2-normalized
modality_mask:  uint8 (1=image, 2=text, 3=both)
metadata_json:  encoder/version/source SHA-256/provenance
```

安全规则：COCO `image_id` 和 KuaiRec `video_id` 不是同一个 ID 空间。导出时必须
提供显式、双射的 `content_id,video_id` 映射；否则脚本失败。仅 `--allow-identity-demo`
可生成隔离契约 smoke，并会在 metadata 标明 `not_project4_catalog`。

真实接入步骤：

1. 对目标视频提取关键帧、标题、OCR/ASR（这些模态尚未在 COCO 版本验证）。
2. 用同版本 encoder 生成向量和 modality mask。
3. 由内容平台提供 immutable `content_id -> video_id` mapping，并校验覆盖率和一对一。
4. 用项目四训练期的 item feature 维度/标准化契约做一次 as-of 重新训练与 cold-item
   evaluation；不能把 MOSAIC 的 COCO 指标写成 TRACE-Rec 的提升。

