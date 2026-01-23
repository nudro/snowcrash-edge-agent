# Phase 1: Quick Start Guide - SSH Setup

## Overview

Phase 1 sets up SSH key-based authentication between **old-nano** and **orin-nano** so that old-nano can send desktop notifications to orin-nano when a person is detected.

## Prerequisites

- Both devices have SSH installed
- Both devices can reach each other on the network (or will be on Opal network)
- orin-nano has a desktop environment (X11/Wayland)

## Step-by-Step Setup

### Step 1: Run Setup Script on Both Devices

**On old-nano:**
```bash
cd ~/Documents/snowcrash-orin3
python3 scripts/phase1_setup_ssh.py
```

**On orin-nano:**
```bash
cd ~/Documents/snowcrash-orin3
python3 scripts/phase1_setup_ssh.py
```

This will:
- Check if SSH is installed
- Generate SSH key pair if one doesn't exist
- Display your public key

### Step 2: Copy Public Key from old-nano to orin-nano

**On old-nano**, note the public key displayed (looks like `ssh-rsa AAAAB3...`).

**On orin-nano**, add the old-nano's public key to authorized_keys:
```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
nano ~/.ssh/authorized_keys  # or use your preferred editor
```

Paste the public key from old-nano (one line), then save and set permissions:
```bash
chmod 600 ~/.ssh/authorized_keys
```

### Step 3: Ensure SSH Server is Running on orin-nano

```bash
# Check status
sudo systemctl status ssh

# Start if not running
sudo systemctl start ssh

# Enable auto-start on boot
sudo systemctl enable ssh
```

### Step 4: Install notify-send on orin-nano

The desktop notification requires `notify-send`:

```bash
sudo apt-get update
sudo apt-get install libnotify-bin
```

### Step 5: Test Connection and Notification

**On old-nano**, run the test script:
```bash
python3 scripts/phase1_test_connection.py --host orin-nano
```

This will:
- Test SSH connection to orin-nano
- Send a test desktop notification
- Verify everything works

You should see a notification popup on orin-nano's desktop saying "TEST: SSH Notification".

### Step 6: Test Notification Directly

You can also test manually from old-nano:
```bash
ssh orin-nano "DISPLAY=:0 notify-send 'ALERT: PERSON' 'Person detected by old-nano'"
```

## Troubleshooting

### SSH Connection Fails

**Problem**: `ssh orin-nano` times out or asks for password

**Solutions**:
1. Check hostname resolution:
   ```bash
   ping orin-nano
   # If this fails, try using IP address: ssh 192.168.x.x
   ```

2. Verify SSH server is running on orin-nano:
   ```bash
   # On orin-nano
   sudo systemctl status ssh
   ```

3. Check firewall:
   ```bash
   # On orin-nano
   sudo ufw status
   # If enabled, allow SSH:
   sudo ufw allow 22/tcp
   ```

4. Verify public key is in authorized_keys:
   ```bash
   # On orin-nano
   cat ~/.ssh/authorized_keys
   # Should contain the public key from old-nano
   ```

### Desktop Notification Fails

**Problem**: SSH works but no notification appears

**Solutions**:
1. Check if notify-send is installed:
   ```bash
   # On orin-nano
   which notify-send
   sudo apt-get install libnotify-bin
   ```

2. Try with explicit DISPLAY:
   ```bash
   ssh orin-nano "DISPLAY=:0 notify-send 'TEST' 'Message'"
   ```

3. Check DISPLAY variable on orin-nano:
   ```bash
   # On orin-nano (in a terminal on the desktop)
   echo $DISPLAY
   # Should output something like :0 or :1
   ```

4. Test notify-send locally on orin-nano:
   ```bash
   # On orin-nano desktop terminal
   notify-send "TEST" "This should work"
   ```

### Permission Denied

**Problem**: SSH asks for password even after adding key

**Solutions**:
1. Check file permissions on orin-nano:
   ```bash
   # On orin-nano
   chmod 700 ~/.ssh
   chmod 600 ~/.ssh/authorized_keys
   ```

2. Check SSH key permissions on old-nano:
   ```bash
   # On old-nano
   chmod 700 ~/.ssh
   chmod 600 ~/.ssh/id_rsa
   chmod 644 ~/.ssh/id_rsa.pub
   ```

3. Check SSH server config:
   ```bash
   # On orin-nano
   sudo nano /etc/ssh/sshd_config
   # Ensure these lines are present (or uncommented):
   # PubkeyAuthentication yes
   # AuthorizedKeysFile .ssh/authorized_keys
   # Then restart:
   sudo systemctl restart ssh
   ```

## What's Next?

Once Phase 1 is complete and you can successfully send desktop notifications from old-nano to orin-nano, you're ready for:

- **Phase 2**: Build YOLO detection service on old-nano
- **Phase 3**: Build remote tool on orin-nano
- **Phase 4**: Connect to Opal network and test end-to-end

## Quick Reference

```bash
# Test SSH connection
ssh orin-nano "echo Connection successful!"

# Test desktop notification
ssh orin-nano "DISPLAY=:0 notify-send 'ALERT: PERSON' 'Person detected'"

# Run automated test
python3 scripts/phase1_test_connection.py --host orin-nano

# Test notification utility directly
python3 tools/desktop_notification.py --host orin-nano --title "TEST" --message "Hello from old-nano"
```

