#!/bin/bash

PROJECT_ROOT="/home/michael/Projects/Personal/microstructures"

# groupadd -g 2000 -U michael contributors
# useradd -G 2000 -u 1001 -m agent

chown michael:contributors -R "$PROJECT_ROOT"

# Grant agent read-write permissions by default
# Exclude (potentially) sensitive files
chmod -R 'u=rwX,g=rwX,o=' "$PROJECT_ROOT" \
    && chmod 'u=rw,g=,o=' .env \
    && chmod 'u=rwX,g=,o=' archive \
    && chmod -R 'u=rwX,g=rX,o=' data results
