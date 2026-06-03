"""エントリポイントベースのプラグインローダー."""
from importlib.metadata import entry_points

from susumu_asr_ros.plugin_base import ASRPluginBase, VADPluginBase, WakewordPluginBase


class PluginLoader:
    """エントリポイントから ASR/VAD/Wakeword プラグインを動的にロードするクラス."""

    _ASR_GROUP = 'susumu_asr_ros.asr_plugins'
    _VAD_GROUP = 'susumu_asr_ros.vad_plugins'
    _WAKEWORD_GROUP = 'susumu_asr_ros.wakeword_plugins'

    @staticmethod
    def _load_eps(group: str) -> dict:
        return {ep.name: ep for ep in entry_points().select(group=group)}

    @classmethod
    def list_asr_plugins(cls) -> list[str]:
        """登録済み ASR プラグイン名一覧."""
        return list(cls._load_eps(cls._ASR_GROUP).keys())

    @classmethod
    def list_vad_plugins(cls) -> list[str]:
        """登録済み VAD プラグイン名一覧."""
        return list(cls._load_eps(cls._VAD_GROUP).keys())

    @classmethod
    def list_wakeword_plugins(cls) -> list[str]:
        """登録済みウェイクワードプラグイン名一覧."""
        return list(cls._load_eps(cls._WAKEWORD_GROUP).keys())

    @classmethod
    def load_asr(cls, name: str) -> type[ASRPluginBase]:
        """名前で ASR プラグインクラスを返す."""
        eps = cls._load_eps(cls._ASR_GROUP)
        if name not in eps:
            raise ValueError(
                f"ASR プラグイン '{name}' が見つかりません。"
                f'利用可能: {list(eps.keys())}'
            )
        return eps[name].load()

    @classmethod
    def load_vad(cls, name: str) -> type[VADPluginBase]:
        """名前で VAD プラグインクラスを返す."""
        eps = cls._load_eps(cls._VAD_GROUP)
        if name not in eps:
            raise ValueError(
                f"VAD プラグイン '{name}' が見つかりません。"
                f'利用可能: {list(eps.keys())}'
            )
        return eps[name].load()

    @classmethod
    def load_wakeword(cls, name: str) -> type[WakewordPluginBase]:
        """名前でウェイクワードプラグインクラスを返す."""
        eps = cls._load_eps(cls._WAKEWORD_GROUP)
        if name not in eps:
            raise ValueError(
                f"ウェイクワードプラグイン '{name}' が見つかりません。"
                f'利用可能: {list(eps.keys())}'
            )
        return eps[name].load()
