#!/usr/bin/env python3
"""
Phase 1: Test SSH Connection and Desktop Notification

Tests SSH connection from old-nano to orin-nano and sends a test desktop notification.
"""
import subprocess
import sys
import argparse


def run_command(cmd, check=True):
    """Run shell command and return result."""
    try:
        # Use stdout/stderr and universal_newlines for Python 3.6 compatibility
        # (capture_output and text added in 3.7)
        result = subprocess.run(
            cmd,
            shell=True,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        stdout_str = result.stdout.strip() if result.stdout else ""
        stderr_str = result.stderr.strip() if result.stderr else ""
        return stdout_str, stderr_str, result.returncode
    except subprocess.CalledProcessError as e:
        stdout_str = e.stdout.strip() if e.stdout else ""
        stderr_str = e.stderr.strip() if e.stderr else ""
        return stdout_str, stderr_str, e.returncode


def test_ssh_connection(hostname, user=None):
    """Test SSH connection to remote host."""
    print(f"Testing SSH connection to {hostname}...")
    
    ssh_cmd = f"ssh -o ConnectTimeout=5"
    if user:
        ssh_cmd += f" {user}@{hostname}"
    else:
        ssh_cmd += f" {hostname}"
    ssh_cmd += " 'echo Connection successful!'"
    
    stdout, stderr, rc = run_command(ssh_cmd, check=False)
    
    if rc == 0:
        print(f"✓ SSH connection successful: {stdout}")
        return True
    else:
        print(f"❌ SSH connection failed:")
        print(f"   {stderr}")
        print(f"\nTroubleshooting:")
        print(f"   1. Ensure SSH server is running on {hostname}")
        print(f"   2. Check if hostname is resolvable: ping {hostname}")
        print(f"   3. Verify SSH key is in ~/.ssh/authorized_keys on {hostname}")
        return False


def test_desktop_notification(hostname, user=None):
    """Test desktop notification on remote host."""
    print(f"\nTesting desktop notification on {hostname}...")
    
    # Build SSH command - use bash -c with properly escaped quotes
    # This ensures proper shell interpretation on remote side
    ssh_target = f"{user}@{hostname}" if user else hostname
    
    # Use bash -c with double quotes and escaped inner quotes
    notify_cmd = 'bash -c "DISPLAY=:0 notify-send \\"TEST: SSH Notification\\" \\"This is a test notification from old-nano\\""'
    ssh_cmd = f"ssh -o ConnectTimeout=5 {ssh_target} {notify_cmd}"
    
    stdout, stderr, rc = run_command(ssh_cmd, check=False)
    
    if rc == 0:
        print("✓ Desktop notification sent successfully!")
        print("   Check the orin-nano desktop for a notification popup.")
        return True
    else:
        print(f"❌ Desktop notification failed:")
        print(f"   {stderr}")
        print(f"\nTroubleshooting:")
        print(f"   1. Ensure orin-nano has a desktop environment (X11/Wayland)")
        print(f"   2. Check if notify-send is installed: sudo apt-get install libnotify-bin")
        print(f"   3. Try with explicit DISPLAY: ssh {hostname} 'DISPLAY=:0 notify-send ...'")
        print(f"   4. Check if user has access to X display")
        return False


def check_notify_send_installed(hostname, user=None):
    """Check if notify-send is installed on remote host."""
    ssh_cmd = f"ssh -o ConnectTimeout=5"
    if user:
        ssh_cmd += f" {user}@{hostname}"
    else:
        ssh_cmd += f" {hostname}"
    ssh_cmd += " 'which notify-send'"
    
    stdout, stderr, rc = run_command(ssh_cmd, check=False)
    
    if rc == 0 and stdout:
        print(f"✓ notify-send is installed: {stdout}")
        return True
    else:
        print(f"⚠ notify-send not found on {hostname}")
        print(f"   Install with: sudo apt-get install libnotify-bin")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test SSH connection and desktop notifications between devices"
    )
    parser.add_argument(
        "--host",
        default="orin-nano",
        help="Hostname or IP of orin-nano (default: orin-nano)"
    )
    parser.add_argument(
        "--user",
        help="SSH username (optional, uses default if not specified)"
    )
    parser.add_argument(
        "--skip-notification",
        action="store_true",
        help="Skip desktop notification test"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Phase 1: Connection Testing")
    print("=" * 70)
    print(f"\nTesting connection to: {args.host}")
    if args.user:
        print(f"Username: {args.user}")
    print()
    
    # Test SSH connection
    if not test_ssh_connection(args.host, args.user):
        print("\n❌ SSH connection failed. Fix SSH setup before proceeding.")
        sys.exit(1)
    
    # Check notify-send installation
    if not args.skip_notification:
        check_notify_send_installed(args.host, args.user)
        
        # Test desktop notification
        if not test_desktop_notification(args.host, args.user):
            print("\n⚠ Desktop notification failed, but SSH connection works.")
            print("You can still proceed - notification setup can be fixed later.")
        else:
            print("\n✓ All tests passed! Ready for Phase 2.")
    else:
        print("\n✓ SSH connection works. Skipping notification test.")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

