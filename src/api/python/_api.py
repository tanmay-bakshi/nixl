# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pickle
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import numpy as np
import torch

from . import _bindings as nixlBind  # type: ignore
from .logging import get_logger

# Get logger using centralized configuration
logger = get_logger(__name__)

DEFAULT_COMM_PORT = nixlBind.DEFAULT_COMM_PORT

nixl_xfer_attestation_snapshot = nixlBind.nixlXferAttestationSnapshot
nixl_xfer_attestation_transport = nixlBind.nixlXferAttestationTransport
nixl_xfer_completion_receipt = nixlBind.nixlXferCompletionReceipt
nixl_terminal_event = nixlBind.nixlTerminalEvent
nixl_terminal_event_batch = nixlBind.nixlTerminalEventBatch
nixl_terminal_channel_inventory = nixlBind.nixlTerminalChannelInventory
nixl_terminal_backend_lifecycle_inventory = (
    nixlBind.nixlTerminalBackendLifecycleInventory
)
nixl_terminal_source_delivery = nixlBind.nixlTerminalSourceDelivery
nixl_terminal_deadline = nixlBind.nixlTerminalDeadline
nixl_terminal_destination_delivery = nixlBind.nixlTerminalDestinationDelivery
nixl_terminal_subscription_info = nixlBind.nixlTerminalSubscriptionInfo
nixl_terminal_event_kind_t = nixlBind.nixl_terminal_event_kind_t
nixl_terminal_capability_state_t = nixlBind.nixl_terminal_capability_state_t
nixl_terminal_channel_fatal_t = nixlBind.nixl_terminal_channel_fatal_t
nixl_terminal_destination_phase_t = nixlBind.nixl_terminal_destination_phase_t

_REMOTE_HANDLE_CONSTRUCTION_TOKEN = object()
_TERMINAL_CHANNEL_CONSTRUCTION_TOKEN = object()
_TERMINAL_SUBSCRIPTION_CONSTRUCTION_TOKEN = object()
_TERMINAL_OWNER_PRODUCER_CONSTRUCTION_TOKEN = object()
_TERMINAL_OWNER_SUBSCRIPTION_CONSTRUCTION_TOKEN = object()


"""
@brief Opaque handle wrapper for a prepared transfer descriptor list.
       Use release() to explicitly free resources; __del__ performs best-effort cleanup.
@param agent Owning nixl_agent used to perform release operations.
@param value Internal handle
"""


class nixl_prepped_dlist_handle:
    __slots__ = ("_handle", "_agent", "_released")

    def __init__(self, agent, value: int):
        self._handle = int(value)
        self._agent = agent
        self._released = False

    def __repr__(self) -> str:
        return (
            f"nixl_prepped_dlist_handle(0x{self._handle:x}, released={self._released})"
        )

    def release(self):
        if not self._released:
            self._agent.releasedDlistH(self._handle)
            self._released = True

    def __del__(self):
        if not self._released:
            try:
                self._agent.releasedDlistH(self._handle)
            except Exception:
                try:
                    logger.error(
                        "nixl_prepped_dlist_handle finalization failed for 0x%x",
                        self._handle,
                    )
                except Exception:
                    pass


"""
@brief Opaque handle wrapper for a transfer request.
       Use release() to explicitly free resources. If transfer was not complete, this will initiate
       the abort process (if available) and will raise an exception.
       __del__ calls release() and if it fails, it logs the failure and defers release by queuing
       the handle in leaked xfer handles list, which will be re-released during agent destruction
@param agent Owning nixl_agent used to perform release operations.
@param value Internal handle
"""


class nixl_xfer_handle:
    __slots__ = ("_handle", "_agent", "_released")

    def __init__(self, agent, value: int):
        self._handle = int(value)
        self._agent = agent
        self._released = False

    def __repr__(self) -> str:
        return f"nixl_xfer_handle(0x{self._handle:x}, released={self._released})"

    def release(self):
        if not self._released:
            self._agent.releaseXferReq(self._handle)
            self._released = True

    def __del__(self):
        if not self._released:
            try:
                self._agent.releaseXferReq(self._handle)
            except Exception:
                try:
                    logger.error(
                        "nixl_xfer_handle finalization failed for 0x%x; keeping handle alive in agent leak list",
                        self._handle,
                    )
                except Exception:
                    pass
                try:
                    self._agent._leaked_xfer_handles.append(self._handle)
                except Exception:
                    pass
                return


class nixl_remote_agent_handle:
    """Canonical Python identity for an agent-owned native remote handle.

    :ivar name: Remote agent name authenticated by the loaded metadata.
    :ivar identity: Process-unique native handle identity.
    :ivar generation: Monotonic generation for this remote agent name.
    """

    __slots__ = ("_active", "_native_handle", "_owner")

    _active: bool
    _native_handle: nixlBind.nixlRemoteAgentH
    _owner: nixlBind.nixlAgent

    def __init__(
        self,
        owner: nixlBind.nixlAgent,
        native_handle: nixlBind.nixlRemoteAgentH,
        construction_token: object,
    ) -> None:
        """Create the canonical wrapper retained by one high-level agent.

        :param owner: Owning native NIXL agent.
        :param native_handle: Agent-owned native remote handle.
        :param construction_token: Module-private construction authority.
        """
        if construction_token is not _REMOTE_HANDLE_CONSTRUCTION_TOKEN:
            raise TypeError("remote-agent handles can only be created by nixl_agent")
        self._owner = owner
        self._native_handle = native_handle
        self._active = True

    @property
    def name(self) -> str:
        """Return the authenticated remote agent name.

        :returns: Remote agent name.
        """
        return self._native_handle.name

    @property
    def identity(self) -> int:
        """Return the process-unique native handle identity.

        :returns: Native handle identity.
        """
        return int(self._native_handle.identity)

    @property
    def generation(self) -> int:
        """Return the name-scoped handle generation.

        :returns: Remote handle generation.
        """
        return int(self._native_handle.generation)

    def __repr__(self) -> str:
        """Return a diagnostic representation of this opaque handle.

        :returns: Handle representation.
        """
        return (
            "nixl_remote_agent_handle("
            f"name={self.name!r}, identity={self.identity}, "
            f"generation={self.generation}, active={self._active})"
        )


class nixl_terminal_event_subscription:
    """Agent-owned exact-generation terminal-event subscription.

    The channel retains this wrapper until :meth:`release` succeeds. An
    asynchronous transfer cancellation returns ``NIXL_IN_PROG`` and preserves
    the native handle until its exact terminal event is delivered.
    """

    __slots__ = ("_channel", "_native_handle", "_released")

    _channel: "nixl_terminal_event_channel"
    _native_handle: nixlBind.nixlTerminalEventSubscriptionH
    _released: bool

    def __init__(
        self,
        channel: "nixl_terminal_event_channel",
        native_handle: nixlBind.nixlTerminalEventSubscriptionH,
        construction_token: object,
    ) -> None:
        """Create an agent-owned subscription wrapper.

        :param channel: Owning terminal-event channel.
        :param native_handle: Agent-owned native subscription handle.
        :param construction_token: Module-private construction authority.
        """
        if construction_token is not _TERMINAL_SUBSCRIPTION_CONSTRUCTION_TOKEN:
            raise TypeError(
                "terminal-event subscriptions can only be created by nixl_agent"
            )
        self._channel = channel
        self._native_handle = native_handle
        self._released = False

    def query(self) -> nixl_terminal_subscription_info:
        """Return an immutable snapshot of the exact binding and lifecycle.

        :returns: Current subscription information.
        :raises RuntimeError: If release already consumed the public handle.
        """
        if self._released:
            raise RuntimeError("terminal-event subscription has already been released")
        return self._channel._owner.agent.queryTerminalEventSubscription(
            self._native_handle
        )

    def release(self) -> nixlBind.nixl_status_t:
        """Request cancellation or consume an already-terminal public handle.

        :returns: ``NIXL_IN_PROG`` while exact transfer cancellation is pending,
            or ``NIXL_SUCCESS`` once release consumes the public handle.
        :raises RuntimeError: If release already consumed the public handle.
        """
        if self._released:
            raise RuntimeError("terminal-event subscription has already been released")
        status = self._channel._owner.agent.releaseTerminalEventSubscription(
            self._native_handle
        )
        if status == nixlBind.NIXL_SUCCESS:
            self._released = True
            self._channel._forget(self)
        return status

    def __repr__(self) -> str:
        """Return a diagnostic representation without dereferencing a stale handle.

        :returns: Subscription representation.
        """
        return f"nixl_terminal_event_subscription(released={self._released})"


class nixl_terminal_event_channel:
    """One bounded, agent-scoped channel for autonomous native events."""

    __slots__ = ("_native_handle", "_owner", "_subscriptions")

    _native_handle: nixlBind.nixlTerminalEventChannelH
    _owner: "nixl_agent"
    _subscriptions: dict[int, nixl_terminal_event_subscription]

    def __init__(
        self,
        owner: "nixl_agent",
        native_handle: nixlBind.nixlTerminalEventChannelH,
        construction_token: object,
    ) -> None:
        """Create the sole channel wrapper for one agent.

        :param owner: Owning high-level NIXL agent.
        :param native_handle: Agent-owned native channel handle.
        :param construction_token: Module-private construction authority.
        """
        if construction_token is not _TERMINAL_CHANNEL_CONSTRUCTION_TOKEN:
            raise TypeError("terminal-event channels can only be created by nixl_agent")
        self._owner = owner
        self._native_handle = native_handle
        self._subscriptions = {}

    def fileno(self) -> int:
        """Return the borrowed poll descriptor.

        The descriptor may be registered with a selector, but must not be read,
        duplicated as ownership, or closed by the caller.

        :returns: Borrowed nonblocking eventfd.
        """
        return int(
            self._owner.agent.getTerminalEventChannelFd(self._native_handle)
        )

    def drain(self) -> nixl_terminal_event_batch:
        """Drain all currently queued events without querying transfer handles.

        :returns: Immutable event batch and post-drain inventory.
        """
        return self._owner.agent.drainTerminalEvents(self._native_handle)

    def query_inventory(self) -> nixl_terminal_channel_inventory:
        """Return immutable channel and native-producer inventory.

        :returns: Current channel inventory and sticky fatal state.
        """
        return self._owner.agent.queryTerminalEventChannel(self._native_handle)

    def close(self) -> nixlBind.nixl_status_t:
        """Stop admission and close once all subscriptions are released.

        :returns: ``NIXL_SUCCESS`` for a clean close.
        :raises nixlBind.nixlNotAllowedError: If live subscriptions make the
            close fail closed.
        """
        return self._owner.agent.closeTerminalEventChannel(self._native_handle)

    def _retain(
        self, native_handle: nixlBind.nixlTerminalEventSubscriptionH
    ) -> nixl_terminal_event_subscription:
        """Retain one public handle until explicit release succeeds.

        :param native_handle: Agent-owned native subscription handle.
        :returns: Canonical high-level subscription wrapper.
        """
        subscription = nixl_terminal_event_subscription(
            self,
            native_handle,
            _TERMINAL_SUBSCRIPTION_CONSTRUCTION_TOKEN,
        )
        self._subscriptions[id(native_handle)] = subscription
        return subscription

    def _forget(self, subscription: nixl_terminal_event_subscription) -> None:
        """Drop a wrapper only after native release consumed its handle.

        :param subscription: Successfully released subscription.
        """
        key = id(subscription._native_handle)
        retained = self._subscriptions.get(key)
        if retained is not subscription:
            raise RuntimeError("terminal-event subscription ownership mismatch")
        del self._subscriptions[key]

    def __repr__(self) -> str:
        """Return a diagnostic channel representation.

        :returns: Channel representation.
        """
        return (
            "nixl_terminal_event_channel("
            f"retained_subscriptions={len(self._subscriptions)})"
        )


@dataclass(frozen=True, slots=True)
class nixl_terminal_owner_producer_inventory:
    """Complete direct NIXL-to-owner producer inventory.

    :ivar registering_count: Exact bindings whose backend arm is returning.
    :ivar submitted_count: Exact bindings awaiting terminal delivery.
    :ivar active_callback_count: Qualified terminal callbacks still outstanding.
    :ivar active_registration_count: Backend subscription calls still returning.
    :ivar total_subscriptions: Successfully armed transfer generations.
    :ivar total_delivered: Events admitted directly into the immutable owner.
    :ivar successful_terminal_event_count: Admitted successful transfer events.
    :ivar failure_terminal_event_count: Admitted local failure events.
    :ivar owner_submission_failure_count: Native owner submission failures.
    :ivar admission_open: Whether new exact bindings may be armed.
    :ivar retirement_requested: Whether ordered producer retirement entered.
    :ivar joined: Whether ordered retirement committed.
    :ivar closed: Whether exact-zero closure completed.
    :ivar fatal_code: Sticky first producer lifecycle failure.
    :ivar fatal_status: Native errno or NIXL status for the first failure.
    :ivar fatal_binding: Exact lifecycle binding associated with the failure.
    """

    registering_count: int
    submitted_count: int
    active_callback_count: int
    active_registration_count: int
    total_subscriptions: int
    total_delivered: int
    successful_terminal_event_count: int
    failure_terminal_event_count: int
    owner_submission_failure_count: int
    admission_open: bool
    retirement_requested: bool
    joined: bool
    closed: bool
    fatal_code: str
    fatal_status: int
    fatal_binding: bytes | None

    @classmethod
    def from_native(
        cls, value: dict[str, object]
    ) -> "nixl_terminal_owner_producer_inventory":
        """Parse one native producer inventory.

        :param value: Native inventory mapping.
        :returns: Validated typed inventory.
        """

        fatal_binding_value = value["fatal_binding"]
        fatal_binding: bytes | None = None
        if fatal_binding_value is not None:
            if type(fatal_binding_value) is not bytes:
                raise TypeError("fatal_binding must be bytes or None")
            if len(fatal_binding_value) != 32:
                raise ValueError("fatal_binding must contain 32 bytes")
            fatal_binding = fatal_binding_value
        return cls(
            registering_count=int(value["registering_count"]),
            submitted_count=int(value["submitted_count"]),
            active_callback_count=int(value["active_callback_count"]),
            active_registration_count=int(value["active_registration_count"]),
            total_subscriptions=int(value["total_subscriptions"]),
            total_delivered=int(value["total_delivered"]),
            successful_terminal_event_count=int(
                value["successful_terminal_event_count"]
            ),
            failure_terminal_event_count=int(value["failure_terminal_event_count"]),
            owner_submission_failure_count=int(
                value["owner_submission_failure_count"]
            ),
            admission_open=bool(value["admission_open"]),
            retirement_requested=bool(value["retirement_requested"]),
            joined=bool(value["joined"]),
            closed=bool(value["closed"]),
            fatal_code=str(value["fatal_code"]),
            fatal_status=int(value["fatal_status"]),
            fatal_binding=fatal_binding,
        )

    @property
    def retained_count(self) -> int:
        """Return every exact owner binding still retained.

        :returns: Registering and submitted binding count.
        """

        return self.registering_count + self.submitted_count


class nixl_terminal_owner_subscription:
    """Retained direct-owner subscription for one exact transfer generation."""

    __slots__ = ("_native_handle", "_producer", "_released")

    _native_handle: nixlBind.nixlTerminalEventSubscriptionH
    _producer: "nixl_terminal_owner_producer"
    _released: bool

    def __init__(
        self,
        producer: "nixl_terminal_owner_producer",
        native_handle: nixlBind.nixlTerminalEventSubscriptionH,
        construction_token: object,
    ) -> None:
        """Retain one agent-owned native subscription.

        :param producer: Direct owner producer which armed the binding.
        :param native_handle: Agent-owned native subscription handle.
        :param construction_token: Module-private construction authority.
        """

        if construction_token is not _TERMINAL_OWNER_SUBSCRIPTION_CONSTRUCTION_TOKEN:
            raise TypeError(
                "terminal-owner subscriptions can only be created by nixl_agent"
            )
        self._producer = producer
        self._native_handle = native_handle
        self._released = False

    def query(self) -> nixl_terminal_subscription_info:
        """Return the exact transfer identity and active state.

        :returns: Current immutable subscription snapshot.
        """

        if self._released:
            raise RuntimeError("terminal-owner subscription has already been released")
        return self._producer._owner.agent.queryTerminalEventSubscription(
            self._native_handle
        )

    def release(self) -> nixlBind.nixl_status_t:
        """Cancel or consume the exact terminal subscription.

        :returns: ``NIXL_IN_PROG`` while cancellation is pending, otherwise
            ``NIXL_SUCCESS`` after the public handle is consumed.
        """

        if self._released:
            raise RuntimeError("terminal-owner subscription has already been released")
        status = self._producer._owner.agent.releaseTerminalEventSubscription(
            self._native_handle
        )
        if status == nixlBind.NIXL_SUCCESS:
            self._released = True
            self._producer._forget(self)
        return status


class nixl_terminal_owner_producer:
    """Process-lifetime direct NIXL callback producer for one immutable owner."""

    __slots__ = ("_native_handle", "_owner", "_subscriptions")

    _native_handle: nixlBind.nixlTerminalOwnerProducerH
    _owner: "nixl_agent"
    _subscriptions: dict[int, nixl_terminal_owner_subscription]

    def __init__(
        self,
        owner: "nixl_agent",
        native_handle: nixlBind.nixlTerminalOwnerProducerH,
        construction_token: object,
    ) -> None:
        """Bind one native producer to a NIXL agent.

        :param owner: NIXL agent whose exact transfer generations are armed.
        :param native_handle: Capsule-bound native producer.
        :param construction_token: Module-private construction authority.
        """

        if construction_token is not _TERMINAL_OWNER_PRODUCER_CONSTRUCTION_TOKEN:
            raise TypeError("terminal-owner producers can only be created by nixl_agent")
        self._owner = owner
        self._native_handle = native_handle
        self._subscriptions = {}

    def _retain(
        self, native_handle: nixlBind.nixlTerminalEventSubscriptionH
    ) -> nixl_terminal_owner_subscription:
        """Retain one newly armed exact-generation subscription.

        :param native_handle: Agent-owned native subscription.
        :returns: Canonical Python lifetime wrapper.
        """

        subscription = nixl_terminal_owner_subscription(
            self,
            native_handle,
            _TERMINAL_OWNER_SUBSCRIPTION_CONSTRUCTION_TOKEN,
        )
        identity = id(native_handle)
        if identity in self._subscriptions:
            raise RuntimeError("native terminal-owner subscription identity collision")
        self._subscriptions[identity] = subscription
        return subscription

    def _forget(self, subscription: nixl_terminal_owner_subscription) -> None:
        """Forget one publicly consumed subscription.

        :param subscription: Consumed canonical wrapper.
        """

        identity = id(subscription._native_handle)
        if self._subscriptions.get(identity) is not subscription:
            raise RuntimeError("terminal-owner subscription retention mismatch")
        del self._subscriptions[identity]

    def stop_admission(self) -> None:
        """Permanently stop new transfer-generation bindings."""

        self._native_handle.stopAdmission()

    def join(self, timeout_seconds: float) -> bool:
        """Join callbacks and ordered producer retirement.

        :param timeout_seconds: Positive native wait bound.
        :returns: Whether ordered retirement committed within the bound.
        """

        if type(timeout_seconds) is not float or timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be a positive float")
        return bool(self._native_handle.join(timeout_seconds))

    def close(self) -> None:
        """Close after exact-zero subscriptions and ordered retirement."""

        if len(self._subscriptions) != 0:
            raise RuntimeError("terminal-owner producer retains public subscriptions")
        self._native_handle.close()

    def inventory(self) -> nixl_terminal_owner_producer_inventory:
        """Return complete binding, callback, and retirement inventory.

        :returns: Typed producer inventory.
        """

        return nixl_terminal_owner_producer_inventory.from_native(
            self._native_handle.inventory()
        )


def terminal_owner_producer_abi() -> dict[str, object]:
    """Return the independently compiled owner-producer ABI layout.

    :returns: ABI version, flags, structure sizes, offsets, and header digest.
    """

    value = nixlBind.terminalOwnerProducerAbi()
    if type(value) is not dict:
        raise TypeError("terminal-owner producer ABI must be a dictionary")
    offsets = value["event_offsets"]
    if type(offsets) is not dict:
        raise TypeError("terminal-owner producer ABI offsets must be a dictionary")
    return {
        "abi_version": int(value["abi_version"]),
        "api_struct_size": int(value["api_struct_size"]),
        "event_struct_size": int(value["event_struct_size"]),
        "required_flags": int(value["required_flags"]),
        "event_offsets": {str(key): int(item) for key, item in offsets.items()},
        "header_sha256": str(value["header_sha256"]),
    }


# Opaque handle for backend can be just int, as it's not passed to the user
nixl_backend_handle = int

"""
@brief Enumeration of supported thread synchronization modes.
"""


class nixl_thread_sync_t(Enum):
    NIXL_THREAD_SYNC_NONE = nixlBind.NIXL_THREAD_SYNC_NONE
    NIXL_THREAD_SYNC_STRICT = nixlBind.NIXL_THREAD_SYNC_STRICT
    NIXL_THREAD_SYNC_RW = nixlBind.NIXL_THREAD_SYNC_RW
    NIXL_THREAD_SYNC_DEFAULT = nixlBind.NIXL_THREAD_SYNC_DEFAULT


"""
@brief Configuration class for NIXL agent.

@param enable_prog_thread Whether to enable the progress thread, if available.
@param enable_listen_thread Whether to enable the listener thread for metadata communication.
@param listen_port Specify the port for the listener thread to listen on.
@param capture_telemetry Whether to enable telemetry capture.
@param num_threads Specify number of threads for the supported multi-threaded backends.
@param backends List of backend names for agent to initialize.
        Default is UCX, other backends can be added to the list, or after
        agent creation, can be initialized with create_backend.
@param sync_mode Thread synchronization mode to use for the agent.
        If None, sync_mode is set based on the enable_listen flag.
"""


class nixl_agent_config:
    def __init__(
        self,
        enable_prog_thread: bool = True,
        enable_listen_thread: bool = False,
        listen_port: int = DEFAULT_COMM_PORT,
        capture_telemetry: bool = False,
        num_threads: int = 0,
        backends: list[str] = ["UCX"],
        sync_mode: Optional[nixl_thread_sync_t] = None,
    ):
        # TODO: add backend init parameters
        self.backends = backends
        self.enable_pthread = enable_prog_thread
        self.enable_listen = enable_listen_thread
        self.port = listen_port
        self.capture_telemetry = capture_telemetry
        self.num_threads = num_threads
        if sync_mode is not None and not isinstance(sync_mode, nixl_thread_sync_t):
            raise TypeError(
                f"sync_mode must be a nixl_thread_sync_t (got {type(sync_mode).__name__!r})"
            )

        self.sync_mode = sync_mode


"""
@brief Main class for creating a NIXL agent and performing transfers.
        This class provides methods for initializing backends, creating descriptor lists,
        registering memory, performing data transfers, and destroying NIXL objects.

@param agent_name Name of the agent, should be unique for clarity.
@param nixl_conf Optional configuration for the agent, described in nixl_agent_config.
@param instantiate_all Whether to instantiate all available backend plugins.
"""


class nixl_agent:
    def __init__(
        self,
        agent_name: str,
        nixl_conf: Optional[nixl_agent_config] = None,
        instantiate_all: bool = False,
    ):
        if nixl_conf and instantiate_all:
            instantiate_all = False
            logger.warning(
                "Ignoring instantiate_all based on the provided config in agent creation."
            )
        if not nixl_conf:
            nixl_conf = nixl_agent_config()  # Using defaults set in nixl_agent_config

        default_sync_mode = (
            nixl_thread_sync_t.NIXL_THREAD_SYNC_STRICT
            if nixl_conf.enable_listen
            else nixl_thread_sync_t.NIXL_THREAD_SYNC_NONE
        )

        # Set agent config and instantiate an agent
        agent_config = nixlBind.nixlAgentConfig()
        agent_config.useProgThread = nixl_conf.enable_pthread
        agent_config.useListenThread = nixl_conf.enable_listen
        agent_config.listenPort = nixl_conf.port
        agent_config.syncMode = (nixl_conf.sync_mode or default_sync_mode).value
        agent_config.pthrDelay = 0
        agent_config.lthrDelay = 100000
        agent_config.captureTelemetry = nixl_conf.capture_telemetry
        self.agent = nixlBind.nixlAgent(agent_name, agent_config)

        self.name = agent_name
        self._leaked_xfer_handles: list[int] = []
        self._remote_agent_handles: dict[int, nixl_remote_agent_handle] = {}
        self._terminal_event_channel: nixl_terminal_event_channel | None = None
        self.notifs: dict[nixl_remote_agent_handle, list[bytes]] = {}
        self.backends: dict[str, nixl_backend_handle] = {}
        self.backend_mems: dict[str, list[str]] = {}
        self.backend_options: dict[str, dict[str, str]] = {}

        self.plugin_list = self.agent.getAvailPlugins()
        if len(self.plugin_list) == 0:
            logger.error("No plugins available, cannot start transfers!")
            raise RuntimeError("No plugins available for NIXL, cannot start transfers!")

        self.plugin_b_options: dict[str, dict[str, str]] = {}
        self.plugin_mem_types: dict[str, list[str]] = {}

        if instantiate_all:
            nixl_conf.backends = self.plugin_list

        for bknd in nixl_conf.backends:
            if bknd not in self.plugin_list:
                logger.warning(
                    "Skipping backend registration %s due to the missing plugin.",
                    bknd,
                )
            else:
                # TODO: improve population of init from nixl_conf
                init: dict[str, str] = {}
                if nixl_conf.num_threads > 0:
                    if bknd == "UCX" or bknd == "OBJ":
                        init["num_threads"] = str(nixl_conf.num_threads)
                    elif bknd == "GDS_MT":
                        init["thread_count"] = str(nixl_conf.num_threads)
                    elif bknd == "UCCL":
                        init["num_cpus"] = str(nixl_conf.num_threads)
                self.create_backend(bknd, init)

        self.nixl_mems = {
            "DRAM": nixlBind.DRAM_SEG,
            "VRAM": nixlBind.VRAM_SEG,
            "FILE": nixlBind.FILE_SEG,
            "BLOCK": nixlBind.BLK_SEG,
            "OBJ": nixlBind.OBJ_SEG,
            "cpu": nixlBind.DRAM_SEG,  # deprecated
            "cuda": nixlBind.VRAM_SEG,  # deprecated
        }
        self.nixl_ops = {
            "WRITE": nixlBind.NIXL_WRITE,
            "READ": nixlBind.NIXL_READ,
        }

        logger.info("Initialized NIXL agent: %s", agent_name)

    def _wrap_remote_agent(
        self, native_handle: nixlBind.nixlRemoteAgentH
    ) -> nixl_remote_agent_handle:
        """Resolve one native handle to its canonical high-level wrapper.

        :param native_handle: Agent-owned native remote handle.
        :returns: Canonical wrapper for the native identity.
        :raises RuntimeError: If an identity aliases a different native handle.
        """
        identity = int(native_handle.identity)
        existing = self._remote_agent_handles.get(identity)
        if existing is not None:
            if existing._native_handle != native_handle:
                raise RuntimeError("native remote-handle identity collision")
            return existing

        handle = nixl_remote_agent_handle(
            self.agent, native_handle, _REMOTE_HANDLE_CONSTRUCTION_TOKEN
        )
        self._remote_agent_handles[identity] = handle
        return handle

    def _unwrap_remote_agent(
        self,
        handle: nixl_remote_agent_handle,
        require_active: bool = True,
    ) -> nixlBind.nixlRemoteAgentH:
        """Validate ownership and return the native remote handle.

        :param handle: High-level remote handle to validate.
        :param require_active: Whether a tombstoned handle must be rejected locally.
        :returns: Agent-owned native remote handle.
        :raises TypeError: If ``handle`` is not a remote-agent handle.
        :raises ValueError: If ``handle`` belongs to another agent.
        :raises RuntimeError: If ``handle`` is tombstoned.
        """
        if not isinstance(handle, nixl_remote_agent_handle):
            raise TypeError("remote_agent must be a nixl_remote_agent_handle")
        if handle._owner is not self.agent:
            raise ValueError("remote-agent handle belongs to a different NIXL agent")
        if require_active and not handle._active:
            raise RuntimeError("remote-agent handle has been invalidated")
        return handle._native_handle

    def __del__(self):
        # Best-effort cleanup of any leaked xfer handles belonging to this agent
        if getattr(self, "_leaked_xfer_handles", None):
            for h in list(self._leaked_xfer_handles):
                try:
                    self.releaseXferReq(h)
                except Exception as e:
                    try:
                        logger.error(
                            "Failed to finalize leaked nixl_xfer_handle 0x%x: %s", h, e
                        )
                    except Exception:
                        pass
            self._leaked_xfer_handles.clear()

    """
    @brief Get the list of available plugins.

    @return List of plugin names.
    """

    def _load_plugin_params(self, plugin: str):
        if plugin not in self.plugin_list:
            return
        try:
            (backend_options, mem_types) = self.agent.getPluginParams(plugin)
            self.plugin_b_options[plugin] = backend_options
            self.plugin_mem_types[plugin] = mem_types
        except Exception:
            logger.warning("Failed to load params for plugin %s", plugin, exc_info=True)

    def get_plugin_list(self) -> list[str]:
        return self.plugin_list

    """
    @brief Get the memory types supported by a plugin.

    @param backend Name of the plugin.
    @return List of supported memory types.
    """

    def get_plugin_mem_types(self, backend: str) -> list[str]:
        if backend not in self.plugin_mem_types:
            self._load_plugin_params(backend)
        if backend not in self.plugin_mem_types:
            logger.warning(
                "Plugin %s is not available to get its supported mem types.", backend
            )
            return []
        return self.plugin_mem_types[backend]

    """
    @brief Get the initialization parameters of a plugin.
           This is a dictionary of strings (option name) to strings (default value for that option).

    @param backend Name of the plugin to get params for.
    @return Dictionary of plugin parameters, described above.
    """

    def get_plugin_params(self, backend: str) -> dict[str, str]:
        if backend not in self.plugin_b_options:
            self._load_plugin_params(backend)
        if backend not in self.plugin_b_options:
            logger.warning("Plugin %s is not available to get its parameters.", backend)
            return {}
        return self.plugin_b_options[backend]

    """
    @brief  Get the memory types supported by a backend.
            Here, a backend means an initialized plugin.
            After a plugin is initialized, the supported memory types might have changed.
            This function is for getting a refreshed list of those memory types.

    @param backend Name of the backend.
    @return List of supported memory types.
    """

    def get_backend_mem_types(self, backend: str) -> list[str]:
        if backend in self.backend_mems:
            return self.backend_mems[backend]
        else:
            logger.warning(
                "Backend %s not instantiated to get its supported mem types.", backend
            )
            return []

    """
    @brief  Get the parameters of a backend.
            Here, a backend means an initialized plugin.
            Available initialization parameters (described above) might have changed after initialization.
            This function is for getting a refreshed list of those parameters.

    @param backend Name of the backend.
    @return Dictionary of backend parameters, described in get_plugin_params.
    """

    def get_backend_params(self, backend: str) -> dict[str, str]:
        if backend in self.backend_options:
            return self.backend_options[backend]
        else:
            logger.warning(
                "Backend %s not instantiated to get its parameters.", backend
            )
            return {}

    """
    @brief  Initialize a backend with the specified initialization parameters, described above.

    @param backend Name of the backend.
    @param initParams Dictionary of initialization parameters.
    """

    def create_backend(self, backend: str, initParams: dict[str, str] = {}):
        self.backends[backend] = self.agent.createBackend(backend, initParams)

        (backend_options, mem_types) = self.agent.getBackendParams(
            self.backends[backend]
        )
        self.backend_mems[backend] = mem_types
        self.backend_options[backend] = backend_options
        logger.info("Backend %s was instantiated", backend)

    """
    @brief Register memory regions, optionally with specified backends.

    @param reg_list List of either memory regions, tensors, or nixlRegDList to register.
    @param mem_type Optional memory type, necessary if specifying a list of memory regions.
    @param backends Optional list of backend names for registration, otherwise NIXL will try to
            register with all backends that support this memory type.
    @return nixlRegDList for the registered memory, can be used with deregister_memory.
    """

    def register_memory(
        self,
        reg_list,
        mem_type: Optional[str] = None,
        backends: list[str] = [],
    ) -> nixlBind.nixlRegDList:
        reg_descs = self.get_reg_descs(reg_list, mem_type)

        handle_list = []
        for backend_string in backends:
            handle_list.append(self.backends[backend_string])
        self.agent.registerMem(reg_descs, handle_list)

        return reg_descs

    """
    @brief Deregister memory regions from the specified backends.

    @param dereg_list nixlRegDList of memory to deregister, received from register_memory or get_reg_descs.
    @param backends Optional list of backend names for deregistration, otherwise NIXL will deregister
            with all the backends that have these memory regions registered.
    """

    def deregister_memory(
        self, dereg_list: nixlBind.nixlRegDList, backends: list[str] = []
    ):
        handle_list = []
        for backend_string in backends:
            handle_list.append(self.backends[backend_string])
        self.agent.deregisterMem(dereg_list, handle_list)

    """
    @brief Query information about memory/storage for a specific backend.

    @param reg_list List of either memory regions, tensors, or nixlRegDList to query.
    @param backend Backend name for querying.
    @param mem_type Optional memory type, necessary if specifying a list of memory regions.
    @return List of query results where each item is either None if not found, or a dictionary with the info
    """

    def query_memory(
        self, reg_list, backend: str, mem_type: Optional[str] = None
    ) -> list[Optional[dict[str, str]]]:
        reg_descs = self.get_reg_descs(reg_list, mem_type)

        # Get the backend handle
        if backend not in self.backends:
            raise ValueError(
                f"Backend '{backend}' not found. Available backends: {list(self.backends.keys())}"
            )

        return self.agent.queryMem(reg_descs, self.backends[backend])

    """
    @brief  Proactively establish a connection with a remote agent,
            which will reduce the time spent in the first transfer between the two agents.
            NIXL will establish the connection for all the backends that talk to that remote
            agent, or limit to the set of backends passed through the backends argument.
            This function is optional.

    @param backends Optional list of backend names to limit the connections to specific backends
    @param remote_agent Name of the remote agent.
    """

    def make_connection(
        self,
        remote_agent: str | nixl_remote_agent_handle,
        backends: list[str] = [],
    ) -> None:
        """Proactively connect to an owned remote handle or local loopback.

        :param remote_agent: Owned remote handle, or this agent's name for loopback.
        :param backends: Backend names to use.
        """
        handle_list = []
        for backend_string in backends:
            handle_list.append(self.backends[backend_string])

        if isinstance(remote_agent, str):
            if remote_agent != self.name:
                raise TypeError(
                    "remote connections require a nixl_remote_agent_handle"
                )
            self.agent.makeConnection(remote_agent, handle_list)
            return

        self.agent.makeConnection(self._unwrap_remote_agent(remote_agent), handle_list)

    """
    @brief  Prepare a transfer descriptor list for data transfer.
            Later, elements from this list can be used to create a transfer request by index.
            It should be done on the initiator agent, and for both sides of a transfer.
            Considering loopback, there are 3 modes for agent_name:
              - For local descriptors, it is set to NIXL_INIT_AGENT,
                indicating that this is a local preparation to be used as local_side handle.
              - For remote descriptors, it is set to the remote name, indicating
                that this is remote side preparation to be used for remote_side handle.
              - For loopback descriptors, it is set to local agent's name, indicating that
                this is for a loopback (local) transfer to be used for remote_side handle
            Preparation succeeds if there exists at least one backend that can handle all
            elements in the descriptor list.

    @param agent_name Name of the agent. It can be "NIXL_INIT_AGENT", local agent name, or remote agent name
    @param xfer_list List of transfer descriptors, can be list of memory region tuples, tensors,
                     Nx3 numpy array, or nixlXferDList. See get_xfer_descs for more details on the structure.
    @param mem_type Optional memory type necessary for list of memory regions.
    @param backends Optional list of backend names to limit which backends are used during preparation
    @return Opaque handle to the prepared transfer descriptor list.
    """

    def prep_xfer_dlist(
        self,
        remote_agent: str | nixl_remote_agent_handle,
        xfer_list,
        mem_type: Optional[str] = None,
        backends: list[str] = [],
    ) -> nixl_prepped_dlist_handle:
        """Prepare a local, loopback, or handle-bound descriptor list.

        :param remote_agent: Empty/init sentinel for local descriptors, this agent's
            name for loopback, or an owned remote handle.
        :param xfer_list: Transfer descriptors.
        :param mem_type: Descriptor memory type.
        :param backends: Backend names to use.
        :returns: Prepared descriptor-list handle.
        """
        descs = self.get_xfer_descs(xfer_list, mem_type)

        is_initiator = isinstance(remote_agent, str) and remote_agent in (
            "NIXL_INIT_AGENT",
            "",
        )
        is_loopback = isinstance(remote_agent, str) and remote_agent == self.name
        if isinstance(remote_agent, str) and not (is_initiator or is_loopback):
            raise TypeError(
                "remote descriptor lists require a nixl_remote_agent_handle"
            )

        handle_list = []
        for backend_string in backends:
            handle_list.append(self.backends[backend_string])

        if is_initiator:
            handle = self.agent.prepXferDlist(descs, handle_list)
        elif is_loopback:
            handle = self.agent.prepXferDlist(self.name, descs, handle_list)
        else:
            handle = self.agent.prepXferDlist(
                self._unwrap_remote_agent(remote_agent), descs, handle_list
            )
        return nixl_prepped_dlist_handle(self.agent, handle)

    """
    @brief Estimate the cost of a transfer operation.
           Times are in microseconds and the method indicates how the estimation was performed.

    @param req_handle Handle to the transfer operation.
    @return Tuple of duration, error margin, method
    """

    def estimate_xfer_cost(self, req_handle: nixl_xfer_handle) -> tuple[int, int, int]:
        duration, err_margin, method = self.agent.estimateXferCost(req_handle._handle)
        if method == nixlBind.NIXL_COST_ANALYTICAL_BACKEND:
            method = "ANALYTICAL_BACKEND"
        else:
            method = "UNKNOWN"
        return duration, err_margin, method

    """
    @brief Prepare a transfer operation using prep_xfer_dlist handles.

    @param operation Type of operation ("WRITE" or "READ").
    @param local_xfer_side Handle to the local transfer descriptor list,
            received from prep_xfer_dlist.
    @param local_indices List or numpy array (dtype=int32) of indices for selecting local descriptors.
    @param remote_xfer_side Handle to the remote (or loopback) transfer descriptor list,
            received from prep_xfer_dlist.
    @param remote_indices List or numpy array (dtype=int32) of indices for selecting remote descriptors.
    @param notif_msg Optional notification message to send after transfer is done.
           notif_msg should be bytes, as that is what will be returned to the target, but will work with str too.
    @param backends Optional list of backend names to limit which backends NIXL can use.
    @param skip_desc_merge Deprecated: Whether to skip descriptor merging optimization.
    @return Opaque handle for posting/checking transfer.
            The handle can be released by calling release_xfer_handle from agent, or release() method on itself.
    """

    def make_prepped_xfer(
        self,
        operation: str,
        local_xfer_side: nixl_prepped_dlist_handle,
        local_indices: Union[list[int], np.ndarray],
        remote_xfer_side: nixl_prepped_dlist_handle,
        remote_indices: Union[list[int], np.ndarray],
        notif_msg: bytes = b"",
        backends: list[str] = [],
        skip_desc_merge: bool = False,
    ) -> nixl_xfer_handle:
        op = self.nixl_ops[operation]
        handle_list = []
        for backend_string in backends:
            handle_list.append(self.backends[backend_string])

        handle = self.agent.makeXferReq(
            op,
            local_xfer_side._handle,
            local_indices,
            remote_xfer_side._handle,
            remote_indices,
            notif_msg,
            handle_list,
            skip_desc_merge,
        )

        return nixl_xfer_handle(self.agent, handle)

    """
    @brief  Initialize a transfer operation. This is a combined API, to create a transfer request
            from two descriptor lists, where NIXL prepares the descriptor lists and then the transfer.
            If there are common descriptors across different transfer requests, using
            this combined API will result in repeated computation, such as validity checks and
            pre-processing done in the preparation step.

    @param operation Type of operation ("WRITE" or "READ").
    @param local_descs List of local transfer descriptors, from get_xfer_descs.
    @param remote_descs List of remote (or loopback) transfer descriptors, from get_xfer_descs.
    @param remote_agent Name of the remote agent.
    @param notif_msg Optional notification message.
           notif_msg should be bytes, as that is what will be returned to the target, but will work with str too.
    @param backends Optional list of backend names to limit which backends NIXL can use.
    @return Opaque handle for posting/checking transfer.
            The handle can be released by calling release_xfer_handle from agent, or release() method on itself.
    """

    def initialize_xfer(
        self,
        operation: str,
        local_descs: nixlBind.nixlXferDList,
        remote_descs: nixlBind.nixlXferDList,
        remote_agent: str | nixl_remote_agent_handle,
        notif_msg: bytes = b"",
        backends: list[str] = [],
    ) -> nixl_xfer_handle:
        """Create a transfer bound to an owned remote handle or local loopback.

        :param operation: Transfer operation name.
        :param local_descs: Local transfer descriptors.
        :param remote_descs: Target transfer descriptors.
        :param remote_agent: Owned remote handle, or this agent's name for loopback.
        :param notif_msg: Optional completion notification.
        :param backends: Backend names to use.
        :returns: Transfer handle.
        """
        op = self.nixl_ops[operation]
        handle_list = []
        for backend_string in backends:
            handle_list.append(self.backends[backend_string])

        if isinstance(remote_agent, str):
            if remote_agent != self.name:
                raise TypeError(
                    "remote transfers require a nixl_remote_agent_handle"
                )
            native_remote_agent = remote_agent
        else:
            native_remote_agent = self._unwrap_remote_agent(remote_agent)

        handle = self.agent.createXferReq(
            op,
            local_descs,
            remote_descs,
            native_remote_agent,
            notif_msg,
            handle_list,
        )

        return nixl_xfer_handle(self.agent, handle)

    """
    @brief  Initiate a data transfer operation.
            After calling this, the transfer state can be checked asynchronously till completion.
            In case of small transfers that are completed as part of the call itself, return value
            will be "DONE", otherwise "PROC" or "ERR".

    @param handle Handle to the transfer operation, from make_prepped_xfer, or initialize_xfer.
    @param notif_msg Optional notification message can be specified or updated per transfer call.
           notif_msg should be bytes, as that is what will be returned to the target, but will work with str too.
    @return Status of the transfer operation ("DONE", "PROC", "NOT_READY", or "ERR").
    """

    def transfer(self, handle: nixl_xfer_handle, notif_msg: bytes = b"") -> str:
        try:
            status = self.agent.postXferReq(handle._handle, notif_msg)
        except nixlBind.nixlNotReadyError:
            return "NOT_READY"
        if status == nixlBind.NIXL_SUCCESS:
            return "DONE"
        if status == nixlBind.NIXL_IN_PROG:
            return "PROC"
        return "ERR"

    """
    @brief Check the state of a transfer operation.

    @param handle Handle to the transfer operation, from make_prepped_xfer, or initialize_xfer.
    @return Status of the transfer operation ("DONE", "PROC", or "ERR").
    """

    def check_xfer_state(self, handle: nixl_xfer_handle) -> str:
        status = self.agent.getXferStatus(handle._handle)
        if status == nixlBind.NIXL_SUCCESS:
            return "DONE"
        elif status == nixlBind.NIXL_IN_PROG:
            return "PROC"
        else:
            return "ERR"

    """
    @brief Get telemetry information of a transfer request.
           The output object has three time values fields in microseconds
           (startTime, postDuration, xferDuration), as well as integer totalBytes transferred
           for the request, and integer descCount representing number of descriptors involved
           (for example if there was some merging of descriptors).

    @param handle Handle to the transfer operation, from make_prepped_xfer or initialize_xfer.
    @return nixlXferTelemetry object
    """

    def get_xfer_telemetry(
        self, handle: nixl_xfer_handle
    ) -> nixlBind.nixlXferTelemetry:
        return self.agent.getXferTelemetry(handle._handle)

    """
    @brief Query the backend that was chosen for a transfer operation.

    @param handle Handle to the transfer operation.
    @return Name of the backend decided for the transfer.
    """

    def query_xfer_backend(self, handle: nixl_xfer_handle) -> str:
        b_handle = self.agent.queryXferBackend(handle._handle)
        # this works because there should not be multiple matching handles in the Dict
        return next(
            backendS
            for backendS, backendH in self.backends.items()
            if backendH == b_handle
        )

    def query_xfer_attestation(
        self, handle: nixl_xfer_handle
    ) -> nixl_xfer_attestation_snapshot:
        """Query diagnostic transport evidence for a transfer generation.

        :param handle: Transfer handle owned by this agent.
        :returns: Read-only diagnostic snapshot. A snapshot never authorizes a
            lifecycle transition.
        """
        self._validate_xfer_attestation_handle(handle)
        return handle._agent.queryXferAttestation(handle._handle)

    def query_xfer_ucp_transports(
        self, handle: nixl_xfer_handle
    ) -> tuple[nixl_xfer_attestation_transport, ...]:
        """Query selected UCP data resources observed for a transfer generation.

        :param handle: Transfer handle owned by this agent.
        :returns: Canonical union of handle-bound selected transport and device
            pairs. The tuple is empty until the backend has posted segment
            evidence.
        """
        snapshot = self.query_xfer_attestation(handle)
        selected_by_resource: dict[
            tuple[str, str], nixl_xfer_attestation_transport
        ] = {}
        for segment in snapshot.segments:
            for transport in segment.selectedTransports:
                resource = (transport.transport, transport.device)
                selected_by_resource[resource] = transport
        return tuple(
            selected_by_resource[resource] for resource in sorted(selected_by_resource)
        )

    def take_xfer_completion_receipt(
        self, handle: nixl_xfer_handle
    ) -> nixl_xfer_completion_receipt | None:
        """Take unique completion authority for a successful transfer generation.

        :param handle: Transfer handle owned by this agent.
        :returns: A read-only, take-once completion receipt, or ``None`` while
            the whole transfer handle is still in progress.
        """
        self._validate_xfer_attestation_handle(handle)
        return handle._agent.takeXferCompletionAttestation(handle._handle)

    def _validate_xfer_attestation_handle(self, handle: nixl_xfer_handle) -> None:
        """Validate ownership and lifetime before a native attestation call.

        :param handle: Transfer handle to validate.
        """
        if not isinstance(handle, nixl_xfer_handle):
            raise TypeError("handle must be a nixl_xfer_handle")
        if handle._agent is not self.agent:
            raise ValueError("transfer handle belongs to a different NIXL agent")
        if handle._released:
            raise ValueError("transfer handle has already been released")

    def create_terminal_owner_producer(
        self,
        producer_api: object,
        producer_context: object,
    ) -> nixl_terminal_owner_producer:
        """Bind the immutable owner's native producer ABI to this NIXL agent.

        :param producer_api: Named capsule containing the versioned producer API.
        :param producer_context: Named capsule containing one registered producer.
        :returns: Process-lifetime direct terminal producer.
        """

        native_handle = self.agent.createTerminalOwnerProducer(
            producer_api, producer_context
        )
        return nixl_terminal_owner_producer(
            self,
            native_handle,
            _TERMINAL_OWNER_PRODUCER_CONSTRUCTION_TOKEN,
        )

    def subscribe_xfer_terminal_owner(
        self,
        producer: nixl_terminal_owner_producer,
        handle: nixl_xfer_handle,
        binding_digest: bytes,
    ) -> nixl_terminal_owner_subscription:
        """Arm direct owner delivery for one exact next transfer generation.

        :param producer: Capsule-bound direct owner producer for this agent.
        :param handle: Transfer request owned by this agent.
        :param binding_digest: Exact 32-byte immutable lifecycle binding.
        :returns: Retained exact-generation subscription.
        """

        if type(producer) is not nixl_terminal_owner_producer:
            raise TypeError("producer must be a nixl_terminal_owner_producer")
        if producer._owner is not self:
            raise ValueError("terminal-owner producer belongs to a different NIXL agent")
        self._validate_xfer_attestation_handle(handle)
        if type(binding_digest) is not bytes or len(binding_digest) != 32:
            raise ValueError("binding_digest must contain 32 bytes")
        native_handle = self.agent.subscribeXferTerminalOwner(
            producer._native_handle,
            handle._handle,
            binding_digest,
        )
        return producer._retain(native_handle)

    def create_terminal_event_channel(
        self, capacity: int
    ) -> nixl_terminal_event_channel:
        """Create this agent's sole bounded autonomous-event channel.

        :param capacity: Positive maximum number of queued events.
        :returns: Agent-owned terminal-event channel.
        :raises RuntimeError: If this agent already created a channel.
        """
        if self._terminal_event_channel is not None:
            raise RuntimeError("this NIXL agent already owns a terminal-event channel")
        native_handle = self.agent.createTerminalEventChannel(capacity)
        channel = nixl_terminal_event_channel(
            self, native_handle, _TERMINAL_CHANNEL_CONSTRUCTION_TOKEN
        )
        self._terminal_event_channel = channel
        return channel

    def subscribe_xfer_terminal(
        self,
        channel: nixl_terminal_event_channel,
        handle: nixl_xfer_handle,
        owner_cookie: int,
    ) -> nixl_terminal_event_subscription:
        """Arm autonomous terminal delivery for the transfer's next generation.

        :param channel: This agent's terminal-event channel.
        :param handle: Transfer handle owned by this agent.
        :param owner_cookie: Positive opaque owner correlation identity.
        :returns: Retained exact-generation subscription.
        """
        self._validate_terminal_event_channel(channel)
        self._validate_xfer_attestation_handle(handle)
        native_handle = self.agent.subscribeXferTerminal(
            channel._native_handle, handle._handle, owner_cookie
        )
        return channel._retain(native_handle)

    def subscribe_remote_notification_state(
        self,
        channel: nixl_terminal_event_channel,
        remote_agent: nixl_remote_agent_handle,
        backend: str,
        owner_cookie: int,
    ) -> nixl_terminal_event_subscription:
        """Subscribe to state transitions for one exact notification route.

        :param channel: This agent's terminal-event channel.
        :param remote_agent: Active remote-agent handle owned by this agent.
        :param backend: Instantiated backend name for the route.
        :param owner_cookie: Positive opaque owner correlation identity.
        :returns: Retained exact-route capability subscription.
        :raises ValueError: If the backend is not instantiated by this agent.
        """
        self._validate_terminal_event_channel(channel)
        native_remote_agent = self._unwrap_remote_agent(remote_agent)
        if backend not in self.backends:
            raise ValueError(f"backend {backend!r} is not instantiated by this agent")
        native_handle = self.agent.subscribeRemoteNotificationState(
            channel._native_handle,
            native_remote_agent,
            self.backends[backend],
            owner_cookie,
        )
        return channel._retain(native_handle)

    def _validate_terminal_event_channel(
        self, channel: nixl_terminal_event_channel
    ) -> None:
        """Validate the canonical channel wrapper for this agent.

        :param channel: Channel wrapper to validate.
        :raises TypeError: If the value is not a terminal-event channel.
        :raises ValueError: If another agent owns the channel.
        """
        if not isinstance(channel, nixl_terminal_event_channel):
            raise TypeError("channel must be a nixl_terminal_event_channel")
        if channel._owner is not self or self._terminal_event_channel is not channel:
            raise ValueError("terminal-event channel belongs to a different NIXL agent")

    """
    @brief  Releases a transfer handle, which internally frees the memory used for the handle.
            If the transfer is active, NIXL will attempt to cancel it.
            If it cannot be canceled, an error will be returned and the handle will not be freed.

    @param handle Handle to the transfer operation from initialize_xfer or make_xfer.
    """

    def release_xfer_handle(self, handle: nixl_xfer_handle):
        handle.release()

    """
    @brief Release a descriptor list handle, which internally frees the memory used for the handle.

    @param handle Handle to the descriptor list from make_prepped_dlist.
    """

    def release_dlist_handle(self, handle: nixl_prepped_dlist_handle):
        handle.release()

    """
    @brief Get new notifications that have come to the agent.

    @param backends Optional list of backend names to limit which backends are checked for notifications.
    @return Dictionary of new notifications.
            Return Dict is a map of remote agent names to a list of notification messages from that agent.
    """

    def get_new_notifs(
        self, backends: list[str] = []
    ) -> dict[nixl_remote_agent_handle, list[bytes]]:
        """Drain newly authenticated notifications keyed by canonical handle.

        :param backends: Backend names to poll.
        :returns: New notifications keyed by canonical remote handle.
        :raises RuntimeError: If native notification ownership is unknown or stale.
        """
        handle_list = []
        for backend_string in backends:
            handle_list.append(self.backends[backend_string])

        native_notifications = self.agent.getRemoteNotifs(handle_list)
        notifications: dict[nixl_remote_agent_handle, list[bytes]] = {}
        for native_handle, messages in native_notifications.items():
            identity = int(native_handle.identity)
            remote_handle = self._remote_agent_handles.get(identity)
            if remote_handle is None:
                raise RuntimeError(
                    "received a notification for an unknown remote handle"
                )
            if remote_handle._native_handle != native_handle:
                raise RuntimeError(
                    "notification remote-handle identity collision"
                )
            if not remote_handle._active:
                raise RuntimeError(
                    "received a notification for an invalidated remote handle"
                )
            notifications[remote_handle] = list(messages)
        return notifications

    """
    @brief Update notifications in a map
            Same as get_new_notifs, but returns all unhandled notifications in agent.

    @param backends Optional list of backend names to limit which backends are checked for notifications.
    @return Dictionary of updated notifications.
    """

    def update_notifs(
        self, backends: list[str] = []
    ) -> dict[nixl_remote_agent_handle, list[bytes]]:
        """Merge newly authenticated notifications into the retained map.

        :param backends: Backend names to poll.
        :returns: Retained notifications keyed by canonical remote handle.
        """
        for remote_handle, messages in self.get_new_notifs(backends).items():
            self.notifs.setdefault(remote_handle, []).extend(messages)
        return self.notifs

    """
    @brief Check if a remote transfer is done with a specific notification.
           Will only remove the notification that is found.

    @param remote_agent_name Name of the remote agent.
    @param lookup_tag A tag to match against available messages in the notification map.
           The tag Can be the same as the entire expected message.
    @param backends Optional list of backend names to limit which backends are checked for notifications.
    @param tag_is_prefix Optionally specify that the tag you want to search with is just a prefix, or can be search as a substring of the full message.
    @return True if the notification is found, False otherwise.
    """

    def check_remote_xfer_done(
        self,
        remote_agent: nixl_remote_agent_handle,
        lookup_tag: bytes,
        backends: list[str] = [],
        tag_is_prefix: bool = True,
    ) -> bool:
        """Consume one matching notification from an owned remote handle.

        :param remote_agent: Owned remote handle.
        :param lookup_tag: Notification tag to match.
        :param backends: Backend names to poll.
        :param tag_is_prefix: Whether the tag must be a prefix.
        :returns: Whether a matching notification was consumed.
        """
        self._unwrap_remote_agent(remote_agent)
        self.update_notifs(backends)
        found = False
        message: bytes | None = None

        if remote_agent in self.notifs:
            for msg in self.notifs[remote_agent]:
                if (tag_is_prefix and msg.startswith(lookup_tag)) or (
                    not tag_is_prefix and lookup_tag in msg
                ):
                    message = msg
                    found = True
                    break
        if message is not None:
            self.notifs[remote_agent].remove(message)
        return found

    """
    @brief Send a standalone notification to a remote agent, not bound to a transfer.

    @param remote_agent_name Name of the remote agent.
    @param notif_msg Message to send, it will be received as bytes.
           notif_msg should be bytes, as that is what will be returned to the target, but will work with str too.
    @param backends Optional a backend name to use to send the notifications.
    """

    def send_notif(
        self,
        remote_agent: str | nixl_remote_agent_handle,
        notif_msg: bytes,
        backend: Optional[str] = None,
    ) -> None:
        """Send a notification to an owned remote handle or local loopback.

        :param remote_agent: Owned remote handle, or this agent's name for loopback.
        :param notif_msg: Notification payload.
        :param backend: Optional backend name.
        """
        if isinstance(remote_agent, str):
            if remote_agent != self.name:
                raise TypeError(
                    "remote notifications require a nixl_remote_agent_handle"
                )
            native_remote_agent = remote_agent
        else:
            native_remote_agent = self._unwrap_remote_agent(remote_agent)
        if backend is None:
            self.agent.genNotif(native_remote_agent, notif_msg)
        else:
            self.agent.genNotif(
                native_remote_agent, notif_msg, [self.backends[backend]]
            )

    """
    @brief Get the full metadata of the local agent.

    @return Metadata of the local agent, in bytes.
    """

    def get_agent_metadata(self) -> bytes:
        return self.agent.getLocalMD()

    """
    @brief Get partial metadata of the local agent.

    @param descs         The list of descriptors to include metadata about.
                         List can be empty if only trying to send connection info.
    @param inc_conn_info Whether to include connection info in the metadata.
    @param backends      List of backends to consider when constructing partial metadata.

    @return Metadata of the local agent, in bytes.
    """

    def get_partial_agent_metadata(
        self,
        descs: nixlBind.nixlRegDList,
        inc_conn_info: bool = False,
        backends: list[str] = [],
    ) -> bytes:
        handle_list = []
        for backend_string in backends:
            handle_list.append(self.backends[backend_string])
        return self.agent.getLocalPartialMD(descs, inc_conn_info, handle_list)

    """
    @brief Add a remote agent using its metadata. After this call, current agent can
            initiate transfers towards the remote agent.

    @param metadata Metadata of the remote agent, received out-of-band in bytes.
    @return Name of the added remote agent.
    """

    def add_remote_agent(self, metadata: bytes) -> nixl_remote_agent_handle:
        """Load exact remote metadata and return its canonical handle.

        :param metadata: Exact serialized remote metadata.
        :returns: Canonical agent-owned remote handle.
        """
        native_handle = self.agent.loadRemoteMD(metadata)
        return self._wrap_remote_agent(native_handle)

    """
    @brief Remove a remote agent. After this call, current agent cannot initiate
            transfers towards the remote agent specified in the call anymore.
            This call will also result in a disconnect between the two agents.

    @param agent Name of the remote agent.
    """

    def remove_remote_agent(self, agent: nixl_remote_agent_handle) -> None:
        """Invalidate an exact agent-owned remote handle.

        :param agent: Remote handle to invalidate.
        """
        native_handle = self._unwrap_remote_agent(agent)
        try:
            self.agent.invalidateRemoteMD(native_handle)
        finally:
            agent._active = False

    """
    @brief Send all of your metadata to a peer or central metadata server.

    @param ip_addr If specified, will only send metadata to one peer by IP address.
                   Otherwise, metadata will be sent to central metadata server, if supported.
    @param port    If specified next to ip_addr, will try to send to this specific port of a peer.
                   Ignored when sending to a central metadata server.
    """

    def send_local_metadata(self, ip_addr: str = "", port: int = DEFAULT_COMM_PORT):
        self.agent.sendLocalMD(ip_addr, port)

    """
    @brief Send partial metadata of the local agent to a peer or central metadata server.

    @param descs         The list of descriptors to include metadata about.
                         List can be empty if only trying to send connection info.
    @param inc_conn_info Whether to include connection info in the metadata.
    @param backends      List of backends to consider when constructing partial metadata.
    @param ip_addr       If specified, will only send metadata to one peer by IP address.
                         Otherwise, metadata will be sent to central metadata server, if supported.
    @param port          If specified next to ip_addr, will try to send to this specific port of a peer.
                         Ignored when sending to a central metadata server.
    @param label         Label to use for the metadata when sending to central metadata server.
                         Ignored when sending to a peer.
    """

    def send_partial_agent_metadata(
        self,
        descs: nixlBind.nixlRegDList,
        inc_conn_info: bool = False,
        backends: list[str] = [],
        ip_addr: str = "",
        port: int = DEFAULT_COMM_PORT,
        label: str = "",
    ):
        handle_list = []
        for backend_string in backends:
            handle_list.append(self.backends[backend_string])
        self.agent.sendLocalPartialMD(
            descs, inc_conn_info, handle_list, ip_addr, port, label
        )

    """
    @brief Request metadata be retrieved from central metadata server or sent by peer.

    @param ip_addr If specified, will request metadata from one peer by IP address.
    @param port    If specified, will try to request on specific port.
    """

    def fetch_remote_metadata(
        self,
        remote_agent: str,
        ip_addr: str = "",
        port: int = DEFAULT_COMM_PORT,
        label: str = "",
    ):
        self.agent.fetchRemoteMD(remote_agent, ip_addr, port, label)

    """
    @brief Invalidate your own metadata in the central metadata server, or from a specific peer.

    @param ip_addr If specified, will only send invalidation to one peer by IP address.
    @param port    If specified, will try to send to specific port.
    """

    def invalidate_local_metadata(
        self, ip_addr: str = "", port: int = DEFAULT_COMM_PORT
    ):
        self.agent.invalidateLocalMD(ip_addr, port)

    """
    @brief Check if the remote metadata for a specific agent is available.
           When partial metadata methods are used, the descriptor list in question can be specified.

    @param agent Name of the remote agent.

    @return True if available, False otherwise
    """

    def check_remote_metadata(
        self, agent: str, descs: nixlBind.nixlXferDList = None
    ) -> bool:
        if descs is None:  # Just empty list, mem_type not important
            descs = nixlBind.nixlXferDList(nixlBind.DRAM_SEG)
        if self.agent.checkRemoteMD(agent, descs) == nixlBind.NIXL_SUCCESS:
            return True
        else:
            return False

    @staticmethod
    def _tensor_mem_type(tensor: torch.Tensor) -> str:
        return "DRAM" if tensor.get_device() == -1 else "VRAM"

    """
    @brief Get nixlXferDList from different input types:
            a) list of 3 element tuples (address, len, device ID) alongside a mandatory memory type
            b) a tensor
            c) a list of tensors
            d) a Nx3 2D numpy array, each row defines a single descriptor (address, len, device ID),
               alongside a mandatory memory type
            e) passes along if an xfer_dlist is given.

    @param descs List of any of the above types
    @param mem_type Optional memory type necessary for (a).
    @return Transfer descriptor list, nixlXferDList.
    """

    def get_xfer_descs(
        self,
        descs,
        mem_type: Optional[str] = None,
    ) -> nixlBind.nixlXferDList:
        # can add check for DLPack input

        if isinstance(descs, nixlBind.nixlXferDList):
            return descs
        elif isinstance(descs, nixlBind.nixlRegDList):
            logger.error("RegList type detected for transfer, please use XferList")
            new_descs = None
        elif isinstance(descs[0], tuple):
            if mem_type is not None and len(descs[0]) == 3:
                new_descs = nixlBind.nixlXferDList(self.nixl_mems[mem_type], descs)
            elif mem_type is None:
                logger.error("Please specify a mem type if not using Tensors")
                new_descs = None
            else:
                logger.error("3-tuple list needed for transfer")
                new_descs = None
        elif isinstance(descs, np.ndarray):
            if mem_type is not None and descs.ndim == 2 and descs.shape[1] == 3:
                new_descs = nixlBind.nixlXferDList(self.nixl_mems[mem_type], descs)
            elif mem_type is None:
                logger.error("Please specify a mem type if not using Tensors")
                new_descs = None
            else:
                logger.error(
                    "Nx3 shape required for transfer descriptor list from numpy array"
                )
                new_descs = None
        elif isinstance(descs, torch.Tensor):
            if descs.is_contiguous():
                mem_type = self._tensor_mem_type(descs)
                base_addr = descs.data_ptr()
                region_len = descs.numel() * descs.element_size()
                gpu_id = descs.get_device()
                if gpu_id == -1:  # DRAM
                    gpu_id = 0
                new_descs = nixlBind.nixlXferDList(
                    self.nixl_mems[mem_type], [(base_addr, region_len, gpu_id)]
                )
            else:
                logger.error("Please use a list of contiguous Tensors")
                new_descs = None
        elif isinstance(descs[0], torch.Tensor):  # List[torch.Tensor]:
            tensor_type = descs[0].device
            dlist = np.zeros((len(descs), 3), dtype=np.uint64)

            for i in range(len(descs)):
                if descs[i].device != tensor_type:
                    return None
                if not descs[i].is_contiguous():
                    logger.error("Please use a list of contiguous Tensors")
                    return None
                base_addr = descs[i].data_ptr()
                region_len = descs[i].numel() * descs[i].element_size()
                gpu_id = descs[i].get_device()
                if gpu_id == -1:  # DRAM
                    gpu_id = 0
                dlist[i, :] = (base_addr, region_len, gpu_id)
            mem_type = self._tensor_mem_type(descs[0])
            new_descs = nixlBind.nixlXferDList(self.nixl_mems[mem_type], dlist)
        else:
            new_descs = None

        return new_descs

    """
    @brief Get nixlRegDList from different input types:
            a) list of 4 element tuples (address, len, device ID, meta info) alongside a mandatory memory type
            b) a tensor
            c) a list of tensors
            d) a Nx3 2D numpy array, each row defines a single descriptor (address, len, device ID),
               alongside a mandatory memory type. Empty meta info will be considered for each descriptor.
            e) passes along if a reg_dlist is given.

    @param descs List of any of the above types
    @param mem_type Optional memory type necessary for (a).
    @return Registration descriptor list, nixlRegDList.
    """

    def get_reg_descs(
        self,
        descs,
        mem_type: Optional[str] = None,
    ) -> nixlBind.nixlRegDList:
        # can add check for DLPack input

        if isinstance(descs, nixlBind.nixlRegDList):
            return descs
        elif isinstance(descs, nixlBind.nixlXferDList):
            logger.error("XferList type detected for registration, please use RegList")
            new_descs = None
        elif isinstance(descs[0], tuple):
            if mem_type is not None and len(descs[0]) == 4:
                new_descs = nixlBind.nixlRegDList(self.nixl_mems[mem_type], descs)
            elif mem_type is None:
                logger.error("Please specify a mem type if not using Tensors")
                new_descs = None
            else:
                logger.error("4-tuple list needed for registration")
                new_descs = None
        elif isinstance(descs, np.ndarray):
            if mem_type is not None and descs.ndim == 2 and descs.shape[1] == 3:
                new_descs = nixlBind.nixlRegDList(self.nixl_mems[mem_type], descs)
            elif mem_type is None:
                logger.error("Please specify a mem type if not using Tensors")
                new_descs = None
            else:
                logger.error(
                    "Nx3 shape required for transfer descriptor list from numpy array"
                )
                new_descs = None
        elif isinstance(descs, torch.Tensor):
            if descs.is_contiguous():
                mem_type = self._tensor_mem_type(descs)
                base_addr = descs.data_ptr()
                region_len = descs.numel() * descs.element_size()
                gpu_id = descs.get_device()
                if gpu_id == -1:  # DRAM
                    gpu_id = 0
                new_descs = nixlBind.nixlRegDList(
                    self.nixl_mems[mem_type], [(base_addr, region_len, gpu_id, "")]
                )
            else:
                logger.error("Please use a list of contiguous Tensors")
                new_descs = None
        elif isinstance(descs[0], torch.Tensor):  # List[torch.Tensor]:
            tensor_type = descs[0].device
            dlist = np.zeros((len(descs), 3), dtype=np.uint64)

            for i in range(len(descs)):
                if descs[i].device != tensor_type:
                    return None
                if not descs[i].is_contiguous():
                    logger.error("Please use a list of contiguous Tensors")
                    return None
                base_addr = descs[i].data_ptr()
                region_len = descs[i].numel() * descs[i].element_size()
                gpu_id = descs[i].get_device()
                if gpu_id == -1:  # DRAM
                    gpu_id = 0
                dlist[i, :] = (base_addr, region_len, gpu_id)
            mem_type = self._tensor_mem_type(descs[0])
            new_descs = nixlBind.nixlRegDList(self.nixl_mems[mem_type], dlist)
        else:
            new_descs = None

        return new_descs

    """
    @brief Serialize NIXL descriptor list with pickle.

    @param descs NIXL list to serialize.
    @return Serialized descriptor list.
    """

    def get_serialized_descs(self, descs) -> bytes:
        return pickle.dumps(descs)

    """
    @brief Deserialize NIXL descriptor list.

    @param serialized_descs Serialized NIXL descriptor list.
    @return Deserialized NIXL descriptor list.
    """

    def deserialize_descs(self, serialized_descs: bytes):
        return pickle.loads(serialized_descs)
