import logging


def configure_logging() -> None:
    """配置全局日志格式。

    这个项目用于本地演示，因此日志策略保持轻量：
    1. 默认 INFO 级别，便于观察关键流程。
    2. 统一输出格式，方便跨模块排查问题。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

