#!/bin/bash

# Use umask 007 so agent's files are readable by the contributors group
sudo runuser -u agent -- bash -c "umask 007; /home/agent/.opencode/bin/opencode $(pwd)"
