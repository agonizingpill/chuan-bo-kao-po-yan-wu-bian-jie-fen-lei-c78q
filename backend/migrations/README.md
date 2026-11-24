# 初始 schema

`src.models.Base.metadata.create_all()` 按外键顺序建表，由 `src.main` 在 API 首次启动时执行。已有开发卷不执行自动字段迁移。项目 README 记录了模型与首次建库入口。

## 训练样本表

`training_samples` 由 `create_all()` 自动创建，`(incident_code, version)` 唯一。若在已有开发卷上升级，需手动补建该表（字段见 `src/models.py` 的 `TrainingSample`）。
