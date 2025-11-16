"""Shared configuration helpers for the SPOTER Temporal Convolutional Network."""

from typing import List, Optional, Sequence

DEFAULT_TCN_CHANNELS = (1028, 512, 512, 1028)


def parse_tcn_channels(value: str) -> List[int]:
    """Parses a comma-separated list of integers describing TCN block widths."""
    if value is None:
        raise ValueError("TCN channels string must not be empty.")
    items = [item.strip() for item in value.split(",")]
    channels = []
    for item in items:
        if not item:
            continue
        try:
            channel = int(item)
        except ValueError as exc:
            raise ValueError(f"Unable to parse '{item}' as an integer.") from exc
        if channel <= 0:
            raise ValueError("TCN channels must be positive integers.")
        channels.append(channel)
    if not channels:
        raise ValueError("At least one TCN channel value must be provided.")
    return channels


def resolve_tcn_channels(channels: Optional[Sequence[int]]) -> List[int]:
    """Returns a validated list of channel widths, falling back to defaults when needed."""
    if channels is None:
        return list(DEFAULT_TCN_CHANNELS)
    resolved = [int(channel) for channel in channels]
    if not resolved:
        raise ValueError("TCN channel list cannot be empty.")
    if any(channel <= 0 for channel in resolved):
        raise ValueError("All TCN channels must be positive integers.")
    return resolved
