#!/usr/bin/env python3
"""
Alert Notification Utility

Sends alert notifications to orin-nano via SSH by writing to a text file.
Used by old-nano YOLO service to alert orin-nano of person detection.
Writes alerts to /tmp/person_alerts.txt on orin-nano.
"""
import subprocess
import sys
from typing import Optional
from pathlib import Path


def send_ssh_notification(
    hostname: str = "orin-nano",
    title: str = "ALERT: PERSON",
    message: str = "Person detected by old-nano",
    user: Optional[str] = None,
    timeout: int = 5,
    alert_file: str = "~/Desktop/person_alerts.txt"
) -> bool:
    """
    Send alert notification to orin-nano via SSH by writing to a text file.
    
    Args:
        hostname: Hostname or IP of orin-nano
        title: Alert title
        message: Alert message
        user: SSH username (optional)
        timeout: SSH connection timeout in seconds
        alert_file: Path to alert file on orin-nano (default: ~/Desktop/person_alerts.txt)
        
    Returns:
        True if alert written successfully, False otherwise
    """
    from datetime import datetime
    
    # Create alert message with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alert_message = f"[{timestamp}] {title}: {message}\n"
    
    # Escape for shell command
    def escape_for_shell(text):
        # Escape single quotes by replacing ' with '\''
        return text.replace("'", "'\\''")
    
    # Expand ~ to $HOME for bash expansion
    alert_file_expanded = alert_file.replace('~', '$HOME')
    alert_file_escaped = escape_for_shell(alert_file_expanded)
    alert_message_escaped = escape_for_shell(alert_message)
    
    # Build SSH command to append to alert file
    # Use bash -c to properly expand $HOME and use single quotes to safely handle the message
    append_cmd = f"bash -c \"echo '{alert_message_escaped}' >> {alert_file_escaped}\""
    
    # Build SSH command
    ssh_target = f"{user}@{hostname}" if user else hostname
    ssh_cmd = [
        "ssh",
        "-o", f"ConnectTimeout={timeout}",
        "-o", "StrictHostKeyChecking=no",  # Accept new hosts automatically
        ssh_target,
        append_cmd
    ]
    
    try:
        result = subprocess.run(
            ssh_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=timeout + 2
        )
        
        if result.returncode == 0:
            return True
        else:
            print(f"[ERROR] SSH alert write failed: {result.stderr}", file=sys.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"[ERROR] SSH alert write timed out connecting to {hostname}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print("[ERROR] SSH command not found. Install OpenSSH client.", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[ERROR] SSH alert write failed: {e}", file=sys.stderr)
        return False


def send_person_alert(
    hostname: str = "orin-nano",
    user: Optional[str] = None,
    confidence: Optional[float] = None,
    alert_file: str = "~/Desktop/person_alerts.txt"
) -> bool:
    """
    Send person detection alert to orin-nano by writing to alert file.
    
    Args:
        hostname: Hostname or IP of orin-nano
        user: SSH username (optional)
        confidence: Detection confidence score (optional, for message)
        alert_file: Path to alert file on orin-nano (default: ~/Desktop/person_alerts.txt)
        
    Returns:
        True if alert written successfully, False otherwise
    """
    message = "Person detected by old-nano"
    if confidence is not None:
        message += f" (confidence: {confidence:.1%})"
    
    return send_ssh_notification(
        hostname=hostname,
        title="ALERT: PERSON",
        message=message,
        user=user,
        alert_file=alert_file
    )


if __name__ == "__main__":
    # Test script
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test alert notification via SSH (writes to text file)"
    )
    parser.add_argument(
        "--host",
        default="orin-nano",
        help="Hostname or IP of orin-nano (default: orin-nano)"
    )
    parser.add_argument(
        "--user",
        help="SSH username (optional)"
    )
    parser.add_argument(
        "--title",
        default="TEST: Alert",
        help="Alert title"
    )
    parser.add_argument(
        "--message",
        default="This is a test alert",
        help="Alert message"
    )
    parser.add_argument(
        "--alert-file",
        default="~/Desktop/person_alerts.txt",
        help="Path to alert file on orin-nano (default: ~/Desktop/person_alerts.txt)"
    )
    
    args = parser.parse_args()
    
    print(f"Sending test alert to {args.host} (writing to {args.alert_file})...")
    success = send_ssh_notification(
        hostname=args.host,
        title=args.title,
        message=args.message,
        user=args.user,
        alert_file=args.alert_file
    )
    
    if success:
        print(f"✓ Alert written successfully to {args.alert_file} on {args.host}!")
        print(f"  View alerts: ssh {args.host} 'cat {args.alert_file}'")
        sys.exit(0)
    else:
        print("❌ Alert write failed!")
        sys.exit(1)

