#!/bin/bash
set -euo pipefail

./scripts/ddp_ckpt_off.sh
./scripts/ddp_ckpt_on.sh
./scripts/fsdp_ckpt_off.sh
./scripts/fsdp_ckpt_on.sh