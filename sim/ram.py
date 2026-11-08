"""Refuse to start a multi-process job that would not fit in free RAM (instead of freezing the machine)."""
import psutil


def check_ram(need_mb: float, hint: str):
    avail = psutil.virtual_memory().available / 2**20
    if need_mb > avail:
        raise MemoryError(f"needs ~{need_mb:.0f} MB but only {avail:.0f} MB is free. {hint}")
