# MOSAIC-Retrieval 0.2.2 本地发布核验

- 源提交：`69fe49e2024983251e00e7285c0af19877dcf64f`（日志 handler 生命周期补丁）
- wheel：`mosaic_retrieval-0.2.2-py3-none-any.whl`
- 大小 / SHA-256：`65,694 bytes` / `07f0bc95945991f1170b16ad561aea6d59e4a87159ba8874f5d34a193c5358da`
- 构建环境：临时干净 venv，`setuptools 83.0.0`、`wheel 0.47.0`；使用官方 PyPI 索引安装构建依赖后执行 `--no-build-isolation` wheel 构建。
- 隔离安装、`mosaic.__version__ == 0.2.2`、27-file wheel 门禁：PASS
- wheel 内 `configure_logging` / `close_logging` 生命周期 smoke：PASS
- 单元测试：`39/39 PASS`
- MSR-VTT Frozen Final audit：`9/9 PASS`
- deployment structure：`21/21 PASS`

0.2.2 只修复 FastAPI app 的 per-instance 日志 handler 关闭，避免 Windows 临时目录清理时的文件锁；没有修改或重跑 0.2.0 Frozen Final，也没有新增线上、Docker 或业务质量声明。Docker daemon 不可用，因此 image build/container E2E 明确未执行。
