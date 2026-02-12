"""日志模块：基于 loguru 提供统一的日志记录功能。"""
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from loguru import logger


def _get_default_log_file(name: str = "llm_lab") -> Path:
    """
    获取默认日志文件路径：项目根目录下的 logs/ 文件夹，文件名包含时间戳。
    
    Args:
        name: logger 名称，用于生成日志文件名
    
    Returns:
        日志文件路径
    """
    # 获取项目根目录（假设 logger.py 在 src/llm_lab/utils/ 下）
    # 从当前文件向上找到项目根目录
    current_file = Path(__file__).resolve()
    # src/llm_lab/utils/logger.py -> 项目根目录
    project_root = current_file.parent.parent.parent.parent
    
    # 创建 logs 目录
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{name}_{timestamp}.log"
    
    return logs_dir / log_filename


def setup_logger(
    name: str = "llm_lab",
    level: str = "INFO",
    log_file: Optional[Path] = None,
    auto_log_file: bool = True,
    rotation: str = "10 MB",
    retention: str = "7 days",
    format_string: Optional[str] = None,
) -> None:
    """
    设置 loguru logger 配置。
    
    Args:
        name: logger 名称（用于日志格式和默认文件名）
        level: 日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）
        log_file: 日志文件路径，如果为 None 且 auto_log_file=True，则自动创建
        auto_log_file: 如果为 True 且 log_file 为 None，则自动在项目根目录 logs/ 下创建日志文件
        rotation: 日志文件轮转大小（如 "10 MB", "1 day"）
        retention: 日志文件保留时间（如 "7 days", "1 month"）
        format_string: 自定义日志格式字符串
    """
    # 移除默认的 handler
    logger.remove()
    
    # 设置日志格式
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # 添加控制台输出
    logger.add(
        sys.stdout,
        format=format_string,
        level=level,
        colorize=True,
    )
    
    # 确定日志文件路径
    if log_file is None and auto_log_file:
        log_file = _get_default_log_file(name)
    
    # 添加文件输出（如果指定）
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_file),
            format=format_string,
            level=level,
            rotation=rotation,
            retention=retention,
            encoding="utf-8",
            enqueue=True,  # 异步写入，提高性能
        )
        logger.info(f"日志文件: {log_file}")


def get_logger(name: Optional[str] = None) -> "logger":
    """
    获取 logger 实例。
    
    Args:
        name: logger 名称（可选，主要用于标识）
    
    Returns:
        loguru logger 实例
    """
    if name:
        return logger.bind(name=name)
    return logger


# 便捷函数（保持向后兼容）
def debug(msg: str, *args, **kwargs):
    """记录 DEBUG 级别日志。"""
    logger.debug(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs):
    """记录 INFO 级别日志。"""
    logger.info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs):
    """记录 WARNING 级别日志。"""
    logger.warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs):
    """记录 ERROR 级别日志。"""
    logger.error(msg, *args, **kwargs)


def critical(msg: str, *args, **kwargs):
    """记录 CRITICAL 级别日志。"""
    logger.critical(msg, *args, **kwargs)
