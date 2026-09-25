"""GEARS syringe pump bridge — kill/restart and readiness checks.

Handles per-task restart of GEARS.exe to clear stale TCP connections, then
verifies the syringe pump is detected via the UDP beacon before proceeding.

Environment variables:
  GEARS_EXE              Full path to the GEARS executable.
  GEARS_PORT             TCP port GEARS listens on (default: 50185).
  GEARS_BEACON_PORT      UDP port GEARS broadcasts on (default: 50184).
  GEARS_INSTRUMENT_NAME  Instrument name to find in beacon
                         (default: Verity 4120 Syringe Pump).
"""

import asyncio
import ctypes
import logging
import os
import pathlib
import socket
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

_GEARS_EXE = os.environ.get('GEARS_EXE')
_GEARS_PORT = int(os.environ.get('GEARS_PORT', '50185'))
_GEARS_BEACON_PORT = int(os.environ.get('GEARS_BEACON_PORT', '50184'))
_GEARS_INSTRUMENT_NAME = os.environ.get('GEARS_INSTRUMENT_NAME', 'Verity 4120 Syringe Pump')

if not _GEARS_EXE:
    logger.warning(
        "GEARS_EXE not set — GEARS will not be restarted before each task. "
        "Set GEARS_EXE to the full path of the GEARS executable."
    )


async def restart_gears(max_attempts: int = 3) -> None:
    """Kill and relaunch GEARS, retrying until the syringe pump appears in the beacon."""
    if not _GEARS_EXE:
        return

    exe_path = pathlib.Path(_GEARS_EXE)
    logger.info("GEARS exe: %s (exists: %s)", exe_path, exe_path.exists())

    for attempt in range(1, max_attempts + 1):
        await _kill_gears(exe_path)
        if not _launch_gears(exe_path):
            return

        logger.info("[%d/%d] Waiting for GEARS on port %d...", attempt, max_attempts, _GEARS_PORT)
        if not await _wait_for_port(_GEARS_PORT, timeout=30.0):
            logger.warning("[%d/%d] GEARS did not start listening — retrying", attempt, max_attempts)
            continue

        logger.info("[%d/%d] Waiting for '%s' in GEARS beacon...", attempt, max_attempts, _GEARS_INSTRUMENT_NAME)
        if await _wait_for_instrument_in_beacon(_GEARS_INSTRUMENT_NAME, timeout=30.0):
            logger.info("GEARS ready — proceeding with task")
            return

        logger.warning("[%d/%d] Pump not detected in beacon — restarting GEARS", attempt, max_attempts)

    logger.error("GEARS failed to detect pump after %d attempts — proceeding anyway", max_attempts)


async def _kill_gears(exe_path: pathlib.Path) -> None:
    # Force kill, then poll until the process disappears from tasklist.
    # Once it's gone the kernel has released all its handles, including the
    # USB device — so the new instance can enumerate the pump reliably.
    proc = await asyncio.create_subprocess_exec(
        "taskkill", "/IM", exe_path.name, "/F",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode == 128:
        logger.info("taskkill: GEARS was not running")
        return
    if proc.returncode != 0:
        logger.warning("taskkill returned %d: %s", proc.returncode, (stdout + stderr).decode().strip())
        return
    logger.info("taskkill: %s", stdout.decode().strip())

    deadline = asyncio.get_event_loop().time() + 10.0
    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.5)
        check = await asyncio.create_subprocess_exec(
            "tasklist", "/FI", f"IMAGENAME eq {exe_path.name}", "/NH",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, _ = await check.communicate()
        if exe_path.name.lower() not in out.decode().lower():
            logger.info("GEARS process gone — USB handles released")
            return

    logger.warning("GEARS still in tasklist after 10s — proceeding anyway")


def _launch_gears(exe_path: pathlib.Path) -> bool:
    # ShellExecute mirrors what Windows Explorer does: launches outside any
    # Job Object and with the correct desktop/window-station context.
    # subprocess.Popen inherits the process-compose Job Object, which can
    # prevent GEARS from enumerating USB devices.
    result = ctypes.windll.shell32.ShellExecuteW(
        None, "open", str(exe_path), None, str(exe_path.parent), 1
    )
    if result > 32:
        return True
    logger.error("ShellExecuteW failed for %s (error code %d)", exe_path, result)
    return False


async def _wait_for_port(port: int, timeout: float = 30.0) -> bool:
    """Poll until GEARS accepts a TCP connection on the given port."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection('localhost', port), timeout=2.0
            )
            writer.close()
            await writer.wait_closed()
            return True
        except (ConnectionRefusedError, OSError, asyncio.TimeoutError):
            await asyncio.sleep(1.0)
    return False


async def _wait_for_instrument_in_beacon(instrument_name: str, timeout: float = 30.0) -> bool:
    """Listen on UDP for GEARS beacon packets until the named instrument appears."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('', _GEARS_BEACON_PORT))
    sock.setblocking(False)

    try:
        while loop.time() < deadline:
            try:
                data = await asyncio.wait_for(loop.sock_recv(sock, 4096), timeout=5.0)
                if _beacon_has_instrument(data, instrument_name):
                    return True
            except asyncio.TimeoutError:
                pass
    finally:
        sock.close()

    return False


def _beacon_has_instrument(data: bytes, name: str) -> bool:
    try:
        root = ET.fromstring(data.decode('utf-8', errors='replace'))
        for instr in root.iter('Instrument'):
            instr_name = instr.findtext('Name') or ''
            if name.lower() in instr_name.lower():
                logger.debug("Beacon: found '%s'", instr_name)
                return True
    except ET.ParseError:
        pass
    return False
