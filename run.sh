#!/usr/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate

pids=()

cleanup() {
	trap - EXIT
	kill "${pids[@]}" 2>/dev/null
	wait
}

trap cleanup EXIT
trap 'exit 130' INT TERM

python3 sam_server.py &
pids+=($!)
python3 server.py &
pids+=($!)

wait -n