#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PARAKEET_PYTHON="${PARAKEET_PYTHON:-/home/logan/Projects/parakeet-exp/.venv/bin/python}"
PARAKEET_DEVICE="${PARAKEET_DEVICE:-auto}"
SHERPA_MODEL_DIR="${SHERPA_MODEL_DIR:-/home/logan/Projects/sherpa-exp/sherpa-onnx-nemotron-speech-streaming-en-0.6b-int8-2026-01-14}"
VOXTRAL_URI="${VOXTRAL_URI:-ws://127.0.0.1:8000/v1/realtime}"
VOXTRAL_AUTOSTART="${VOXTRAL_AUTOSTART:-1}"
VOXTRAL_READY_TIMEOUT_S="${VOXTRAL_READY_TIMEOUT_S:-180}"
VOXTRAL_READY_POLL_S="${VOXTRAL_READY_POLL_S:-1}"
VOXTRAL_SERVE_SCRIPT="${VOXTRAL_SERVE_SCRIPT:-${ROOT_DIR}/scripts/serve_voxtral_optimized.sh}"

ARGS=(
  --providers deepgram voxtral sherpa parakeet
  --chunk-ms 40
  --voxtral-uri "${VOXTRAL_URI}"
  "$@"
)

VOXTRAL_PID=""
VOXTRAL_LOG=""

log() {
  printf '[system] %s\n' "$*"
}

cleanup() {
  if [[ -z "${VOXTRAL_PID}" ]]; then
    return
  fi
  if ! kill -0 "${VOXTRAL_PID}" 2>/dev/null; then
    return
  fi

  log "Stopping managed Voxtral server (pid=${VOXTRAL_PID})"
  kill "${VOXTRAL_PID}" 2>/dev/null || true
  wait "${VOXTRAL_PID}" 2>/dev/null || true
}

trap cleanup EXIT

providers_include_voxtral() {
  python3 - "${ARGS[@]}" <<'PY'
import sys

args = sys.argv[1:]
providers = []
i = 0
while i < len(args):
    arg = args[i]
    if arg == "--providers":
        providers = []
        i += 1
        while i < len(args) and not args[i].startswith("--"):
            providers.append(args[i])
            i += 1
        continue
    if arg.startswith("--providers="):
        providers = arg.split("=", 1)[1].split()
    i += 1

print("1" if "voxtral" in providers else "0")
PY
}

resolve_voxtral_uri() {
  python3 - "${ARGS[@]}" <<'PY'
import sys

args = sys.argv[1:]
uri = ""
i = 0
while i < len(args):
    arg = args[i]
    if arg == "--voxtral-uri" and i + 1 < len(args):
        uri = args[i + 1]
        i += 2
        continue
    if arg.startswith("--voxtral-uri="):
        uri = arg.split("=", 1)[1]
    i += 1

print(uri)
PY
}

uri_host_port() {
  python3 - "$1" <<'PY'
from urllib.parse import urlparse
import sys

parsed = urlparse(sys.argv[1])
host = parsed.hostname or "127.0.0.1"
port = parsed.port or (443 if parsed.scheme == "wss" else 80)
print(host)
print(port)
PY
}

is_local_host() {
  case "$1" in
    127.0.0.1|localhost|::1) return 0 ;;
    *) return 1 ;;
  esac
}

port_is_open() {
  python3 - "$1" "$2" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

try:
    with socket.create_connection((host, port), timeout=0.5):
        raise SystemExit(0)
except OSError:
    raise SystemExit(1)
PY
}

wait_for_port() {
  local host="$1"
  local port="$2"
  local timeout_s="$3"
  local poll_s="$4"
  local deadline=$((SECONDS + timeout_s))

  while (( SECONDS < deadline )); do
    if port_is_open "${host}" "${port}"; then
      return 0
    fi
    if [[ -n "${VOXTRAL_PID}" ]] && ! kill -0 "${VOXTRAL_PID}" 2>/dev/null; then
      return 1
    fi
    sleep "${poll_s}"
  done
  return 1
}

maybe_start_voxtral() {
  local use_voxtral
  use_voxtral="$(providers_include_voxtral)"
  if [[ "${use_voxtral}" != "1" ]]; then
    return
  fi

  local voxtral_uri
  voxtral_uri="$(resolve_voxtral_uri)"

  local host
  local port
  mapfile -t host_port < <(uri_host_port "${voxtral_uri}")
  host="${host_port[0]}"
  port="${host_port[1]}"

  if port_is_open "${host}" "${port}"; then
    log "Using existing Voxtral server at ${host}:${port}"
    return
  fi

  if [[ "${VOXTRAL_AUTOSTART}" != "1" ]]; then
    log "Voxtral is selected but no server is listening at ${host}:${port}"
    return
  fi

  if ! is_local_host "${host}"; then
    log "Skipping Voxtral autostart for non-local URI ${voxtral_uri}"
    return
  fi

  if [[ ! -x "${VOXTRAL_SERVE_SCRIPT}" ]]; then
    printf 'Voxtral serve script is not executable: %s\n' "${VOXTRAL_SERVE_SCRIPT}" >&2
    exit 1
  fi

  VOXTRAL_LOG="${VOXTRAL_LOG:-$(mktemp -t stt-exp-voxtral.XXXXXX.log)}"
  log "Starting optimized Voxtral server with ${VOXTRAL_SERVE_SCRIPT}"
  log "Voxtral logs: ${VOXTRAL_LOG}"
  "${VOXTRAL_SERVE_SCRIPT}" >"${VOXTRAL_LOG}" 2>&1 &
  VOXTRAL_PID=$!

  if ! wait_for_port "${host}" "${port}" "${VOXTRAL_READY_TIMEOUT_S}" "${VOXTRAL_READY_POLL_S}"; then
    printf 'Voxtral server did not become ready at %s:%s within %ss\n' "${host}" "${port}" "${VOXTRAL_READY_TIMEOUT_S}" >&2
    if [[ -f "${VOXTRAL_LOG}" ]]; then
      printf 'Last Voxtral log lines from %s:\n' "${VOXTRAL_LOG}" >&2
      tail -n 40 "${VOXTRAL_LOG}" >&2 || true
    fi
    exit 1
  fi

  log "Voxtral server ready at ${host}:${port}"
}

export PARAKEET_PYTHON
export PARAKEET_DEVICE
export SHERPA_MODEL_DIR

maybe_start_voxtral

uv run stt-exp live --parakeet-device "${PARAKEET_DEVICE}" "${ARGS[@]}"
