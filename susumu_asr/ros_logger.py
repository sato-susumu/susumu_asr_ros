"""loguru をROS2ロガーにブリッジするユーティリティ."""
from loguru import logger
import rclpy.logging

_ROS2_LEVEL_MAP = {
    'DEBUG': 'debug',
    'INFO': 'info',
    'SUCCESS': 'info',
    'WARNING': 'warning',
    'ERROR': 'error',
    'CRITICAL': 'fatal',
}


def _ros2_sink(message):
    record = message.record
    name = record['extra'].get('name', record['name'])
    level = _ROS2_LEVEL_MAP.get(record['level'].name, 'info')
    ros_logger = rclpy.logging.get_logger(name)
    getattr(ros_logger, level)(record['message'])


def setup_loguru(log_path=None):
    """Loguru の sink を設定する。デバッグ時は log_path にファイル出力も追加する."""
    logger.remove()
    logger.add(_ros2_sink, format='{message}')
    if log_path:
        logger.add(
            log_path,
            format=(
                '{time:YYYY-MM-DD HH:mm:ss.SSS}'
                ' [{level}] [{extra[name]}]: {message}'
            ),
            encoding='utf-8',
        )


def get_logger(name):
    """Name を bind した loguru ロガーを返す."""
    return logger.bind(name=name)
