#!/usr/bin/env python3
"""
Phase 1: Network Setup and SSH Configuration

This script helps set up SSH key-based authentication between old-nano and orin-nano
for desktop notifications.

Run this script on both devices to configure SSH access.
"""
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, check=True, capture_output=True):
    """Run shell command and return result."""
    try:
        # Use stdout/stderr and universal_newlines for Python 3.6 compatibility
        # (capture_output and text added in 3.7)
        result = subprocess.run(
            cmd,
            shell=True,
            check=check,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
            universal_newlines=True
        )
        stdout_str = result.stdout.strip() if result.stdout else ""
        stderr_str = result.stderr.strip() if result.stderr else ""
        return stdout_str, stderr_str, result.returncode
    except subprocess.CalledProcessError as e:
        stdout_str = e.stdout.strip() if e.stdout else ""
        stderr_str = e.stderr.strip() if e.stderr else ""
        return stdout_str, stderr_str, e.returncode


def check_ssh_installed():
    """Check if SSH is installed."""
    stdout, stderr, rc = run_command("which ssh", check=False)
    if rc != 0:
        print("❌ SSH is not installed. Please install OpenSSH:")
        print("   sudo apt-get update && sudo apt-get install openssh-client openssh-server")
        return False
    print("✓ SSH is installed")
    return True


def check_ssh_key_exists():
    """Check if SSH key already exists."""
    ssh_dir = Path.home() / ".ssh"
    key_file = ssh_dir / "id_rsa"
    pub_key_file = ssh_dir / "id_rsa.pub"
    
    if key_file.exists() and pub_key_file.exists():
        print(f"✓ SSH key already exists: {key_file}")
        return True, pub_key_file.read_text().strip()
    return False, None


def generate_ssh_key():
    """Generate SSH key pair if it doesn't exist."""
    ssh_dir = Path.home() / ".ssh"
    ssh_dir.mkdir(mode=0o700, exist_ok=True)
    
    key_file = ssh_dir / "id_rsa"
    
    if key_file.exists():
        print("SSH key already exists, skipping generation.")
        pub_key_file = ssh_dir / "id_rsa.pub"
        if pub_key_file.exists():
            return pub_key_file.read_text().strip()
    
    print("Generating SSH key pair...")
    stdout, stderr, rc = run_command(
        'ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N "" -q',
        check=False
    )
    
    if rc != 0:
        print(f"❌ Failed to generate SSH key: {stderr}")
        return None
    
    pub_key_file = ssh_dir / "id_rsa.pub"
    if pub_key_file.exists():
        pub_key = pub_key_file.read_text().strip()
        print("✓ SSH key generated successfully")
        return pub_key
    else:
        print("❌ SSH key generation failed - public key not found")
        return None


def get_public_key():
    """Get the current user's public SSH key."""
    exists, pub_key = check_ssh_key_exists()
    if exists:
        return pub_key
    return generate_ssh_key()


def display_instructions():
    """Display setup instructions."""
    print("\n" + "=" * 70)
    print("PHASE 1: SSH Setup for Desktop Notifications")
    print("=" * 70)
    print("\nThis script will help you set up SSH key-based authentication")
    print("between old-nano and orin-nano so that old-nano can send")
    print("desktop notifications to orin-nano when a person is detected.\n")


def main():
    """Main setup function."""
    display_instructions()
    
    # Check SSH installation
    if not check_ssh_installed():
        sys.exit(1)
    
    # Get or generate SSH key
    pub_key = get_public_key()
    if not pub_key:
        print("❌ Failed to get SSH public key")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("\n1. On OLD-NANO, copy your public key:")
    print(f"   {pub_key}")
    print("\n2. On ORIN-NANO, add the old-nano public key to authorized_keys:")
    print("   Run this on orin-nano:")
    print("   mkdir -p ~/.ssh")
    print("   chmod 700 ~/.ssh")
    print(f'   echo "{pub_key}" >> ~/.ssh/authorized_keys')
    print("   chmod 600 ~/.ssh/authorized_keys")
    print("\n3. On ORIN-NANO, ensure SSH server is running:")
    print("   sudo systemctl start ssh")
    print("   sudo systemctl enable ssh")
    print("\n4. Test SSH connection from old-nano to orin-nano:")
    print("   ssh orin-nano 'echo Connection successful!'")
    print("\n5. Test desktop notification:")
    print('   ssh orin-nano "DISPLAY=:0 notify-send \'ALERT: PERSON\' \'Person detected by old-nano\'"')
    print("\n" + "=" * 70)
    
    # Try to detect if we're on orin-nano or old-nano
    hostname_stdout, _, _ = run_command("hostname", check=False)
    hostname = hostname_stdout.strip() if hostname_stdout else "unknown"
    
    print(f"\nCurrent device hostname: {hostname}")
    
    if "orin" in hostname.lower() or "nano" not in hostname.lower():
        print("\nYou appear to be on orin-nano.")
        print("To complete setup, you need to run this on old-nano as well,")
        print("then copy old-nano's public key to this machine's ~/.ssh/authorized_keys")
    else:
        print("\nYou appear to be on old-nano.")
        print("Copy the public key above to orin-nano's ~/.ssh/authorized_keys")


if __name__ == "__main__":
    main()

