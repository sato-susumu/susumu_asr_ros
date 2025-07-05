import json
import threading
import time
from collections import deque
from datetime import datetime
from typing import Dict, List

import click
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class AsrMonitorStats:
    def __init__(self):
        self.total_events = 0
        self.wakeword_detections = 0
        self.final_results = 0
        self.partial_results = 0
        self.timeouts = 0
        self.start_time = time.time()
        self.last_activity = None
        self.session_count = 0
        self.processing_times = deque(maxlen=100)

    def update(self, event_dict: dict):
        self.total_events += 1
        self.last_activity = datetime.now()

        event_type = event_dict.get('event_type')

        if event_type == 'wakeword_detected':
            self.wakeword_detections += 1
            self.session_count += 1
        elif event_type == 'final_result':
            self.final_results += 1
            start_time = event_dict.get('start')
            end_time = event_dict.get('end')
            if start_time and end_time:
                processing_time = end_time - start_time
                self.processing_times.append(processing_time)
        elif event_type == 'partial_result':
            self.partial_results += 1
        elif event_type == 'timeout':
            self.timeouts += 1

    def get_uptime(self) -> float:
        return time.time() - self.start_time

    def get_avg_processing_time(self) -> float:
        if not self.processing_times:
            return 0.0
        return sum(self.processing_times) / len(self.processing_times)

    def get_success_rate(self) -> float:
        if self.session_count == 0:
            return 0.0
        return (self.final_results / self.session_count) * 100


class AsrMonitorNode(Node):
    def __init__(self):
        super().__init__('asr_monitor')
        self.stats = AsrMonitorStats()
        self.event_history = deque(maxlen=1000)
        self.running = True

        # ROS2 subscriber
        self.subscription = self.create_subscription(
            String,
            'stt_event',
            self.event_callback,
            10
        )

        self.get_logger().info('ASR Monitor started - listening to stt_event topic')

    def event_callback(self, msg):
        try:
            event_dict = json.loads(msg.data)
            self.stats.update(event_dict)

            # Add timestamp to event
            event_dict['timestamp'] = datetime.now().isoformat()
            self.event_history.append(event_dict)

        except json.JSONDecodeError as e:
            self.get_logger().error(f'Failed to parse JSON: {e}')

    def get_recent_events(self, count: int = 10) -> List[Dict]:
        return list(self.event_history)[-count:]

    def print_stats(self):
        uptime = self.stats.get_uptime()
        avg_processing = self.stats.get_avg_processing_time()
        success_rate = self.stats.get_success_rate()

        print(f"\n{'='*60}")
        print(f"ASR Monitor Status - Uptime: {uptime:.1f}s")
        print(f"{'='*60}")
        print(f"Total Events:        {self.stats.total_events}")
        print(f"Wakeword Detections: {self.stats.wakeword_detections}")
        print(f"Final Results:       {self.stats.final_results}")
        print(f"Partial Results:     {self.stats.partial_results}")
        print(f"Timeouts:            {self.stats.timeouts}")
        print(f"Sessions:            {self.stats.session_count}")
        print(f"Success Rate:        {success_rate:.1f}%")
        print(f"Avg Processing Time: {avg_processing:.2f}s")

        if self.stats.last_activity:
            time_since_last = datetime.now() - self.stats.last_activity
            print(f"Last Activity:       {time_since_last.total_seconds():.1f}s ago")

        # Show recent events
        recent_events = self.get_recent_events(5)
        if recent_events:
            print("\nRecent Events (last 5):")
            for event in recent_events:
                timestamp = event.get('timestamp', '')
                event_type = event.get('event_type', 'unknown')
                text = event.get('text', '')

                if timestamp:
                    dt = datetime.fromisoformat(timestamp)
                    time_str = dt.strftime("%H:%M:%S")
                else:
                    time_str = "??:??:??"

                if event_type == 'final_result':
                    print(f"  {time_str} [FINAL] {text}")
                elif event_type == 'partial_result':
                    print(f"  {time_str} [PARTIAL] {text}")
                elif event_type == 'wakeword_detected':
                    print(f"  {time_str} [WAKEWORD] detected")
                elif event_type == 'timeout':
                    print(f"  {time_str} [TIMEOUT] speech timeout")
                else:
                    print(f"  {time_str} [{event_type.upper()}]")

    def print_detailed_events(self, count: int = 20):
        events = self.get_recent_events(count)
        if not events:
            print("No events recorded yet.")
            return

        print(f"\n{'='*80}")
        print(f"Detailed Event History (last {min(count, len(events))} events)")
        print(f"{'='*80}")

        for event in events:
            timestamp = event.get('timestamp', '')
            event_type = event.get('event_type', 'unknown')
            text = event.get('text', '')
            start_time = event.get('start')
            end_time = event.get('end')

            if timestamp:
                dt = datetime.fromisoformat(timestamp)
                time_str = dt.strftime("%H:%M:%S.%f")[:-3]
            else:
                time_str = "??:??:??.???"

            print(f"{time_str} | {event_type:15} | ", end="")

            if event_type == 'final_result':
                duration = end_time - start_time if start_time and end_time else 0
                print(f"[{duration:.2f}s] {text}")
            elif event_type == 'partial_result':
                print(f"{text}")
            elif event_type == 'wakeword_detected':
                score = event.get('score', 0)
                print(f"score: {score:.2f}")
            elif event_type == 'timeout':
                reason = event.get('reason', 'unknown')
                print(f"reason: {reason}")
            else:
                print(f"data: {event}")


def monitor_display_loop(monitor_node: AsrMonitorNode, update_interval: float, show_details: bool):
    """Display loop for continuous monitoring"""
    try:
        while rclpy.ok():
            # Clear screen (works on most terminals)
            print("\033[2J\033[H", end="")

            if show_details:
                monitor_node.print_detailed_events(15)
            else:
                monitor_node.print_stats()

            print("\nPress Ctrl+C to exit...")

            time.sleep(update_interval)
    except KeyboardInterrupt:
        pass


@click.command()
@click.option('--update-interval', '-u', default=2.0,
              help='Update interval in seconds (default: 2.0)')
@click.option('--show-details', '-d', is_flag=True,
              help='Show detailed event history instead of stats')
@click.option('--once', '-o', is_flag=True,
              help='Show stats once and exit (no continuous monitoring)')
@click.option('--events', '-e', type=int, help='Show last N events and exit')
def main(update_interval: float, show_details: bool, once: bool, events: int):
    """ASR Status Monitor - Monitor speech recognition system status and events"""

    rclpy.init()
    monitor_node = AsrMonitorNode()

    try:
        if events:
            # Show specific number of events and exit
            print("Waiting for events...")
            time.sleep(1)  # Give some time to collect events
            rclpy.spin_once(monitor_node, timeout_sec=1.0)
            monitor_node.print_detailed_events(events)
        elif once:
            # Show stats once and exit
            print("Collecting data...")
            # Spin for a short time to collect some events
            end_time = time.time() + 2.0
            while time.time() < end_time and rclpy.ok():
                rclpy.spin_once(monitor_node, timeout_sec=0.1)
            monitor_node.print_stats()
        else:
            # Continuous monitoring
            print("Starting continuous monitoring...")
            print("Collecting initial data...")

            # Start ROS2 spinning in a separate thread
            spin_thread = threading.Thread(
                target=lambda: rclpy.spin(monitor_node),
                daemon=True
            )
            spin_thread.start()

            # Wait a bit to collect some initial data
            time.sleep(1)

            # Start display loop
            monitor_display_loop(monitor_node, update_interval, show_details)

    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
    finally:
        monitor_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
