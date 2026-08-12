# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

qualification_crypto_dep = dependency('openssl', modules: ['libcrypto'], required: true)

terminal_ucx_qualification_exe = executable(
    'terminal_ucx_qualification',
    sources: [
        'terminal_ucx_qualification.cpp',
        'terminal_ucx_api_adapter.cpp',
        'terminal_ucx_peer_fixture.cpp',
    ],
    include_directories: [nixl_inc_dirs, utils_inc_dirs],
    dependencies: [nixl_dep, nixl_infra, qualification_crypto_dep, thread_dep],
    install: false,
)

terminal_ucx_notification_failure_qualification_exe = executable(
    'terminal_ucx_notification_failure_qualification',
    sources: [
        'terminal_ucx_notification_failure_qualification.cpp',
        'terminal_ucx_api_adapter.cpp',
        'terminal_ucx_peer_fixture.cpp',
    ],
    include_directories: [nixl_inc_dirs, utils_inc_dirs],
    dependencies: [nixl_dep, nixl_infra, thread_dep],
    install: false,
)

terminal_ucx_receipt_qualification_exe = executable(
    'terminal_ucx_receipt_qualification',
    sources: [
        'terminal_ucx_receipt_qualification.cpp',
        'terminal_ucx_api_adapter.cpp',
        'terminal_ucx_peer_fixture.cpp',
    ],
    include_directories: [nixl_inc_dirs, utils_inc_dirs],
    dependencies: [nixl_dep, nixl_infra, thread_dep],
    install: false,
)

terminal_ucx_endpoint_failure_qualification_exe = executable(
    'terminal_ucx_endpoint_failure_qualification',
    sources: [
        'terminal_ucx_endpoint_failure_qualification.cpp',
        'terminal_ucx_api_adapter.cpp',
        'terminal_ucx_peer_fixture.cpp',
    ],
    include_directories: [nixl_inc_dirs, utils_inc_dirs],
    dependencies: [nixl_dep, nixl_infra, qualification_crypto_dep, thread_dep],
    install: false,
)

terminal_ucx_owner_lifecycle_qualification_exe = executable(
    'terminal_ucx_owner_lifecycle_qualification',
    sources: [
        'terminal_ucx_owner_lifecycle_qualification.cpp',
        'terminal_ucx_api_adapter.cpp',
        'terminal_ucx_peer_fixture.cpp',
    ],
    include_directories: [nixl_inc_dirs, utils_inc_dirs, ucx_backend_includes],
    dependencies: [
        nixl_dep,
        nixl_infra,
        qualification_crypto_dep,
        thread_dep,
        ucx_dep,
        ucx_backend_interface,
    ],
    install: false,
)
