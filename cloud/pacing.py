"""The paced worker's CPU share, recomputed when the set of connected robots changes.

Pure logic: no asyncio, no sockets, no clock reading. The I/O layer calls update in the same
synchronous step as each robot's join or leave, through the session table's
on_membership_change callback.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType

from cloud.config import ServerConfig
from cloud.dispatch import ModelDispatch
from cloud.registry import RobotRecord
from cloud.scheduler import other_demand_cores, pacing_share


class Pacing:
    """The feed-forward pacing share of DESIGN.md 6.4, recomputed when a robot joins or leaves.

    The share is what the CPU quota leaves the paced worker after a headroom and the predicted
    demand of every other consumer, predicted from the frame rates the connected robots
    declared, the workers' calibrated CPU time per frame and the configured I/O cost. It is
    not a feedback loop: nothing measured while the service runs moves it.
    """

    def __init__(
        self,
        config: ServerConfig,
        quota_cores: float | None,
        dispatches: Mapping[str, ModelDispatch],
    ) -> None:
        """Pace nothing until the first update; quota_cores is None without a CPU quota."""
        self._config = config
        self._quota_cores = quota_cores
        self._dispatches = dispatches
        # The configuration allows at most one paced model.
        self._paced = next(
            (model for model, settings in config.models.items() if settings.paced), None
        )
        # Every offered frame is costed as a batch of one, at the CPU time of the first
        # calibration of the run, which a restarted worker does not replace.
        self._cpu_ns_per_frame = {
            model: dispatch.calibration.t_b_cpu_ns[1] for model, dispatch in dispatches.items()
        }
        self._shares: dict[str, float | None] = dict.fromkeys(config.models)
        self._shares_view = MappingProxyType(self._shares)

    def shares(self) -> Mapping[str, float | None]:
        """Return the share each model's worker is paced to now, None where it is not paced."""
        return self._shares_view

    def update(self, records: Iterable[RobotRecord]) -> None:
        """Recompute the paced worker's share for the records connected now, and apply it."""
        # ASSUMPTION: A19. The CPU limit is a quota, which freezes the whole container once its
        # demand passes it. Under a cpuset limit, or no limit, there is no freeze to prevent,
        # so nothing is paced.
        if self._quota_cores is None or self._paced is None:
            return
        share_cores = pacing_share(
            self._quota_cores,
            self._config.pacing_headroom_cores,
            self._config.pacing_min_share_cores,
            self._config.models[self._paced].threads,
            other_demand_cores(self._config, records, self._cpu_ns_per_frame, self._quota_cores),
        )
        self._dispatches[self._paced].scheduler.set_pacing_share(share_cores)
        self._shares[self._paced] = share_cores
