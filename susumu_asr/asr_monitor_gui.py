"""ASR モニター GUI — pyqtgraph ベースのリアルタイム波形・イベント表示."""
import os
import threading
import time

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets  # noqa: I100,I201
import pyqtgraph as pg  # noqa: I201
from susumu_asr.constants import SAMPLE_RATE

# 表示する波形の長さ（秒）
_DISPLAY_SEC = 15
_WAVEFORM_SAMPLES = SAMPLE_RATE * _DISPLAY_SEC

# 音量メーターの平滑化係数
_VU_ALPHA = 0.3

# イベント行の色
_COLOR_VAD = '#4a90d9'
_COLOR_VAD_PRE = '#a8c8f0'
_COLOR_WW = '#e67e22'
_COLOR_P = '#8e44ad'
_COLOR_F = '#16a085'
_COLOR_ASR = '#d4ac0d'

# イベントバーの高さ（0〜1 正規化）
_ROW_VAD = (0.0, 0.28)
_ROW_WW = (0.36, 0.64)
_ROW_ASR = (0.72, 1.00)

# パフォーマンス計測（ASR_MONITOR_PERF=0 で出力停止）
_PERF_ENABLED = os.environ.get('ASR_MONITOR_PERF', '1') != '0'
_PERF_REPORT_SEC = 5.0


def _hex_to_qcolor(h: str) -> QtGui.QColor:
    return QtGui.QColor(h)


class _PerfStats:
    """
    音声取得→表示の各段階の遅延を集計し、定期的に標準出力へ要約を出す.

    段階（名前順 = パイプライン順）:
      1_transport   publisher がフレームを取得してから monitor が受信するまで
      2_buffer_wait 受信してから描画タイマーが拾うまで
      3_draw        _update_plots の処理時間
      4_e2e_draw    フレーム取得から描画処理完了まで（①+②+③）
      5_e2e_painted フレーム取得から Qt のペイント処理後まで（実表示に最も近い）
      frame_gap     描画タイマーの実呼び出し間隔（逆数が実FPS）
    """

    def __init__(self):
        # name -> [合計秒, 最大秒, 回数]
        self._acc: dict[str, list] = {}
        self._last_report = time.perf_counter()

    def add(self, name: str, dt: float):
        e = self._acc.get(name)
        if e is None:
            self._acc[name] = [dt, dt, 1]
        else:
            e[0] += dt
            if dt > e[1]:
                e[1] = dt
            e[2] += 1

    def maybe_report(self, extra: str = ''):
        if not _PERF_ENABLED:
            return
        now = time.perf_counter()
        if now - self._last_report < _PERF_REPORT_SEC:
            return
        gap = self._acc.pop('frame_gap', None)
        fps = f'fps {gap[2] / gap[0]:.1f}' if gap and gap[0] > 0 else 'fps -'
        parts = [
            f'{name}: avg {total / cnt * 1e3:.1f}ms max {mx * 1e3:.1f}ms'
            for name, (total, mx, cnt) in sorted(self._acc.items())
        ]
        print(f'[monitor-perf] {fps} {extra}| ' + ' | '.join(parts),
              flush=True)
        self._acc.clear()
        self._last_report = now


class ASRMonitorWidget(QtWidgets.QWidget):
    """波形・音量・イベントバーをまとめたウィジェット."""

    # スレッドから安全にイベントを受け取るシグナル
    event_received = QtCore.pyqtSignal(dict)

    def __init__(self, parent=None):  # noqa: D107
        super().__init__(parent)
        self._lock = threading.Lock()

        # 波形リングバッファ（_write_idx が次の書き込み位置）
        self._samples = np.zeros(_WAVEFORM_SAMPLES, dtype=np.float32)
        self._write_idx = 0
        self._elapsed_sec = 0.0  # 受信済みサンプル数から計算した経過時間

        # 音量メーター
        self._vu_level = 0.0

        # イベントリスト: (type, t_start, t_end_or_None, label, pre_start_or_None)
        self._events: list[tuple[str, float, float | None, str, float | None]] = []

        # ASRテキスト
        self._asr_text = ''
        self._partial_text = ''

        # 遅延計測（音声取得→表示）
        self._perf = _PerfStats()
        self._last_frame_start: float | None = None
        self._latest_capture_t: float | None = None  # 最新フレームの取得時刻
        self._latest_recv_t: float | None = None  # 最新フレームの受信時刻
        self._last_drawn_recv_t: float | None = None  # 直近描画時のフレーム

        self._build_ui()
        self.event_received.connect(self._on_event)

        # 描画タイマー 60fps（描画自体は数msなので余裕がある。
        # タイマー待ちが表示遅延の主要因の一つのため周期を短くする）
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._update_plots)
        self._timer.start(16)

    # ------------------------------------------------------------------
    # UI構築
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)

        # --- 波形グラフ ---
        self._vad_spans: list[pg.LinearRegionItem] = []
        self._wave_plot = pg.PlotWidget(title='Waveform')
        self._wave_plot.setYRange(-1.05, 1.05)
        self._wave_plot.setXRange(0, _DISPLAY_SEC)
        self._wave_plot.setLabel('left', 'Amplitude')
        self._wave_plot.setLabel('bottom', 'Time (s)')
        self._wave_plot.showGrid(x=True, y=False, alpha=0.3)
        self._wave_plot.setBackground('#1e1e1e')
        self._wave_curve = self._wave_plot.plot(
            pen=pg.mkPen('#5bc8f5', width=1)
        )
        # 画面解像度に合わせてピーク保持ダウンサンプリング（描画点数を削減）
        self._wave_curve.setDownsampling(auto=True, method='peak')
        self._wave_curve.setClipToView(True)
        for _ in range(10):
            span = pg.LinearRegionItem(
                values=(0, 0),
                brush=pg.mkBrush(74, 144, 217, 40),
                pen=pg.mkPen(None),
                movable=False,
            )
            span.hide()
            self._wave_plot.addItem(span)
            self._vad_spans.append(span)
        layout.addWidget(self._wave_plot, stretch=4)

        # --- 音量メーター ---
        vu_row = QtWidgets.QHBoxLayout()
        vu_label = QtWidgets.QLabel('Volume')
        vu_label.setFixedWidth(55)
        vu_row.addWidget(vu_label)
        self._vu_bar = QtWidgets.QProgressBar()
        self._vu_bar.setRange(0, 1000)
        self._vu_bar.setTextVisible(False)
        self._vu_bar.setFixedHeight(18)
        self._vu_bar.setStyleSheet(
            'QProgressBar { border: 1px solid #444; border-radius: 3px;'
            ' background: #1e1e1e; }'
            'QProgressBar::chunk { background: qlineargradient('
            'x1:0, y1:0, x2:1, y2:0,'
            'stop:0 #27ae60, stop:0.6 #f39c12, stop:0.85 #e74c3c); }'
        )
        vu_row.addWidget(self._vu_bar)
        layout.addLayout(vu_row)

        # --- イベントバー ---
        self._event_plot = pg.PlotWidget()
        self._event_plot.setYRange(0, 1)
        self._event_plot.setXRange(0, _DISPLAY_SEC)
        self._event_plot.setLabel('bottom', 'Time (s)')
        self._event_plot.setBackground('#1e1e1e')
        self._event_plot.showGrid(x=True, y=False, alpha=0.3)
        self._event_plot.setXLink(self._wave_plot)

        # Y軸ラベル
        ax = self._event_plot.getAxis('left')
        ax.setTicks([[(
            (_ROW_VAD[0] + _ROW_VAD[1]) / 2, 'VAD',
        ), (
            (_ROW_WW[0] + _ROW_WW[1]) / 2, 'WW',
        ), (
            (_ROW_ASR[0] + _ROW_ASR[1]) / 2, 'ASR',
        )]])
        # 区切り線
        for y in (0.32, 0.68):
            line = pg.InfiniteLine(
                pos=y, angle=0,
                pen=pg.mkPen('#555', width=1, style=QtCore.Qt.DashLine),
            )
            self._event_plot.addItem(line)

        layout.addWidget(self._event_plot, stretch=2)

        # --- ASRテキスト ---
        self._asr_label = QtWidgets.QLabel('')
        self._asr_label.setStyleSheet(
            'font-size: 16px; color: #f0f0f0; background: #2a2a2a;'
            'padding: 4px; border-radius: 4px;'
        )
        self._asr_label.setWordWrap(True)
        self._asr_label.setMinimumHeight(40)
        layout.addWidget(self._asr_label)

        self.setStyleSheet('background-color: #121212; color: #e0e0e0;')

    # ------------------------------------------------------------------
    # 外部からの呼び出し
    # ------------------------------------------------------------------

    def push_audio(self, pcm_bytes: bytes, capture_time: float | None = None):
        """
        マイク/WAVフレームを受け取って波形バッファに追加.

        capture_time は publisher 側でフレームを取得した時刻（epoch秒）。
        与えられた場合、音声取得→表示の遅延計測に使う。
        """
        recv_time = time.time()
        samples = np.frombuffer(
            pcm_bytes, dtype=np.int16
        ).astype(np.float32) / 32768.0
        n_total = len(samples)
        if n_total == 0:
            return
        rms = float(np.sqrt(np.mean(samples ** 2)))
        if n_total >= _WAVEFORM_SAMPLES:
            samples = samples[-_WAVEFORM_SAMPLES:]
        n = len(samples)
        with self._lock:
            idx = self._write_idx
            end = idx + n
            if end <= _WAVEFORM_SAMPLES:
                self._samples[idx:end] = samples
            else:
                k = _WAVEFORM_SAMPLES - idx
                self._samples[idx:] = samples[:k]
                self._samples[:n - k] = samples[k:]
            self._write_idx = end % _WAVEFORM_SAMPLES
            self._elapsed_sec += n_total / SAMPLE_RATE
            self._vu_level = (
                _VU_ALPHA * rms + (1 - _VU_ALPHA) * self._vu_level
            )
            if capture_time is not None:
                self._latest_capture_t = capture_time
                self._latest_recv_t = recv_time
        if capture_time is not None:
            self._perf.add('1_transport', recv_time - capture_time)

    def push_event(self, event: dict):
        """ROS2スレッドから呼ぶ — シグナル経由でUIスレッドへ転送."""
        self.event_received.emit(event)

    # ------------------------------------------------------------------
    # イベント処理（UIスレッド）
    # ------------------------------------------------------------------

    @QtCore.pyqtSlot(dict)
    def _on_event(self, ev: dict):
        et = ev.get('event_type', '')
        t = ev.get('start', self._elapsed_sec)

        if et == 'vad_speech_start':
            pre_start = ev.get('pre_start')
            self._events.append(('vad_start', t, None, '', pre_start))
        elif et == 'vad_speech_stop':
            end = ev.get('end', t)
            # 対応する vad_start を閉じる（pre_start はそのまま引き継ぐ）
            for i in range(len(self._events) - 1, -1, -1):
                ev_i = self._events[i]
                if ev_i[0] == 'vad_start' and ev_i[2] is None:
                    self._events[i] = ('vad', ev_i[1], end, '', ev_i[4])
                    break
        elif et == 'ww_detected':
            self._events.append(('ww', t, t, '', None))
        elif et == 'asr_partial_result':
            self._partial_text = ev.get('text', '')
            # 同じ発話の古いpartialは最新のもので置き換える（重なり防止）
            self._events = [
                e for e in self._events
                if not (e[0] == 'partial' and e[1] == t)
            ]
            self._events.append(('partial', t, t, self._partial_text, None))
        elif et == 'asr_final_result':
            end = ev.get('end', t)
            text = ev.get('text', '')
            self._partial_text = ''
            self._asr_text = text
            # finalが出たら同じ発話のpartial表示は消す（ラベル重なり防止）
            self._events = [
                e for e in self._events
                if not (e[0] == 'partial' and e[1] >= t)
            ]
            self._events.append(('final', t, end, text, None))

        # 古いイベントを削除（表示ウィンドウの2倍以上前）
        cutoff = self._elapsed_sec - _DISPLAY_SEC * 2
        self._events = [e for e in self._events if (e[2] or e[1]) >= cutoff]

    # ------------------------------------------------------------------
    # 描画更新（タイマー）
    # ------------------------------------------------------------------

    def _update_plots(self):
        t_start = time.perf_counter()
        wall_start = time.time()
        if self._last_frame_start is not None:
            self._perf.add('frame_gap', t_start - self._last_frame_start)
        self._last_frame_start = t_start

        with self._lock:
            idx = self._write_idx
            samples = np.concatenate(
                (self._samples[idx:], self._samples[:idx])
            )
            elapsed = self._elapsed_sec
            vu = self._vu_level
            capture_t = self._latest_capture_t
            recv_t = self._latest_recv_t

        t0 = elapsed - _DISPLAY_SEC
        x_left = max(0.0, t0)
        x_right = max(float(_DISPLAY_SEC), elapsed)

        # 波形
        # 実データは常に elapsed 基準で右端に揃える。elapsed < 15s のときは
        # x の左側が負になるが setXRange でクリップされるので問題ない
        x = np.linspace(elapsed - _DISPLAY_SEC, elapsed, len(samples))
        self._wave_curve.setData(x, samples)
        self._wave_plot.setXRange(x_left, x_right)

        # 波形上のVADスパンはpre_startを含む全区間を表示
        vad_regions = [
            (e[4] if e[4] is not None else e[1], e[2] if e[2] is not None else elapsed)
            for e in self._events if e[0] in ('vad', 'vad_start')
        ]
        for i, span in enumerate(self._vad_spans):
            if i < len(vad_regions):
                span.setRegion(vad_regions[i])
                span.show()
            else:
                span.hide()

        # 音量メーター（RMSをdB換算して0〜1000にマッピング）
        db = 20 * np.log10(vu + 1e-9)  # -180〜0 dB
        vu_val = int(max(0, min(1000, (db + 60) / 60 * 1000)))
        self._vu_bar.setValue(vu_val)

        # イベントバー再描画
        self._event_plot.clear()
        for y in (0.32, 0.68):
            line = pg.InfiniteLine(
                pos=y, angle=0,
                pen=pg.mkPen('#555', width=1, style=QtCore.Qt.DashLine),
            )
            self._event_plot.addItem(line)
        self._event_plot.setXRange(x_left, x_right)

        for etype, ts, te, label, pre_start in self._events:
            if etype in ('vad', 'vad_start'):
                te_actual = te if te is not None else (elapsed or ts)
                # pre_speech区間を薄い色で描いた後にspeech区間を濃い色で上書き
                if pre_start is not None:
                    self._draw_bar(pre_start, te_actual, _ROW_VAD, _COLOR_VAD_PRE)
                self._draw_bar(ts, te_actual, _ROW_VAD, _COLOR_VAD)
                self._draw_text(ts, _ROW_VAD, f'{ts:.1f}s', anchor=(0, 0))
            elif etype == 'ww':
                self._draw_vline(ts, _ROW_WW, _COLOR_WW, f'WW\n{ts:.1f}s')
            elif etype == 'partial':
                self._draw_vline(ts, _ROW_ASR, _COLOR_P,
                                 f'[P] {label}\n{ts:.1f}s', top=True)
            elif etype == 'final':
                te_actual = te if te is not None else ts
                self._draw_bar(ts, te_actual, _ROW_ASR, _COLOR_ASR, alpha=60)
                lbl_f = f'[F] {label}\n{te_actual:.1f}s'
                self._draw_vline(
                    te_actual, _ROW_ASR, _COLOR_F, lbl_f, top=False
                )

        # ASRテキスト
        display = self._partial_text or self._asr_text
        prefix = '[部分] ' if self._partial_text else '[確定] '
        self._asr_label.setText(f'{prefix}{display}' if display else '')

        # 遅延計測: このフレームで描画した最新サンプルを基準にする。
        # 音声が止まっている間は古いフレームを測り続けないようスキップ
        self._perf.add('3_draw', time.perf_counter() - t_start)
        if capture_t is not None and recv_t != self._last_drawn_recv_t:
            self._last_drawn_recv_t = recv_t
            self._perf.add('2_buffer_wait', wall_start - recv_t)
            self._perf.add('4_e2e_draw', time.time() - capture_t)
            # singleShot(0) はペンディング中のペイントイベント処理後に
            # 呼ばれるため、実際に画面へ反映された時点の近似になる
            QtCore.QTimer.singleShot(
                0,
                lambda t=capture_t: self._perf.add(
                    '5_e2e_painted', time.time() - t
                ),
            )
        self._perf.maybe_report()

    def _draw_bar(self, x0, x1, row, color, alpha=120):
        ymin, ymax = row
        item = QtWidgets.QGraphicsRectItem(
            x0, ymin + 0.01, x1 - x0, ymax - ymin - 0.02
        )
        c = _hex_to_qcolor(color)
        c.setAlpha(alpha)
        item.setBrush(QtGui.QBrush(c))
        item.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        self._event_plot.addItem(item)

    def _draw_vline(self, x, row, color, label='', top=True):
        ymin, ymax = row
        line = pg.InfiniteLine(
            pos=x, angle=90,
            pen=pg.mkPen(color, width=1.5),
            label=label,
            labelOpts={
                # ラベルは position を中心に描かれるため、上端ぎりぎりに
                # 置くと2行テキストの上半分がプロット外にはみ出して切れる
                'position': 0.85 if top else 0.72,
                'color': '#eeeeee',
                'fill': pg.mkBrush(30, 30, 30, 180),
                'movable': False,
            },
        )
        self._event_plot.addItem(line)

    def _draw_text(self, x, row, text, anchor=(0, 1)):
        ymin, ymax = row
        ymid = (ymin + ymax) / 2
        item = pg.TextItem(text, color='#eeeeee', anchor=anchor)
        item.setPos(x, ymid)
        self._event_plot.addItem(item)
