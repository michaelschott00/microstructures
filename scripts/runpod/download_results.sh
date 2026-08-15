#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <host> <port> <remote_path> <local_path>"
    exit 1
fi

rsync \
    --info=progress2 \
    -arv \
    -e "ssh -p $2" \
    "root@$1:$3" \
    "$4"