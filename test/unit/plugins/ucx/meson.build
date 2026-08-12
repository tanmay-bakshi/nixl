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

ucx_test_dep = [ ucx_dep, nixl_common_deps, ucx_backend_interface]

if cuda_dep.found()
    ucx_test_dep += [ cuda_dep ]
    cpp_args = '-DHAVE_CUDA'
else
    cpp_args = '-UHAVE_CUDA'
endif

ucx_backend_test_dep = [nixl_dep, nixl_infra, thread_dep] + ucx_test_dep + nixl_test_utils_dep
ucx_test_include_directories = [nixl_inc_dirs, utils_inc_dirs]

ucx_backend_test = executable('ucx_backend_test',
        'ucx_backend_test.cpp',
        dependencies: ucx_backend_test_dep,
        include_directories: ucx_test_include_directories,
        cpp_args : cpp_args,
        install: true)

ucx_backend_multi = executable('ucx_backend_multi',
           'ucx_backend_multi.cpp',
           dependencies: ucx_backend_test_dep,
           include_directories: ucx_test_include_directories,
           cpp_args : cpp_args,
           install: true)

ucx_attestation_test = executable('ucx_attestation_test',
        'ucx_attestation_test.cpp',
        dependencies: ucx_backend_test_dep,
        include_directories: ucx_test_include_directories)

test('ucx_attestation', ucx_attestation_test)

ucx_connection_metadata_test = executable('ucx_connection_metadata_test',
        'ucx_connection_metadata_test.cpp',
        dependencies: ucx_backend_test_dep,
        include_directories: ucx_test_include_directories)

test('ucx_connection_metadata', ucx_connection_metadata_test)

ucx_notif_wire_test = executable('ucx_notif_wire_test',
        'ucx_notif_wire_test.cpp',
        dependencies: ucx_backend_test_dep,
        include_directories: ucx_test_include_directories)

test('ucx_notif_wire', ucx_notif_wire_test)

ucx_notif_state_test = executable('ucx_notif_state_test',
        'ucx_notif_state_test.cpp',
        dependencies: ucx_backend_test_dep,
        include_directories: ucx_test_include_directories)

test('ucx_notif_state', ucx_notif_state_test)

ucx_notif_delivery_test = executable('ucx_notif_delivery_test',
        'ucx_notif_delivery_test.cpp',
        dependencies: ucx_backend_test_dep,
        include_directories: ucx_test_include_directories)

test('ucx_notif_delivery', ucx_notif_delivery_test)

ucx_notif_ingress_test = executable('ucx_notif_ingress_test',
        'ucx_notif_ingress_test.cpp',
        dependencies: ucx_backend_test_dep,
        include_directories: ucx_test_include_directories)

test('ucx_notif_ingress', ucx_notif_ingress_test)

ucx_terminal_deadline_test = executable('ucx_terminal_deadline_test',
        'ucx_terminal_deadline_test.cpp',
        dependencies: ucx_backend_test_dep,
        include_directories: ucx_test_include_directories)

test('ucx_terminal_deadline', ucx_terminal_deadline_test)

if get_option('buildtype') != 'release'

    ucx_worker_bin = executable('ucx_worker_test',
               'ucx_worker_test.cpp',
               dependencies: ucx_test_dep,
               include_directories: ucx_test_include_directories,
               install: true)

    if cuda_dep.found()
        ucx_worker_cuda_bin = executable('ucx_worker_test_cuda',
                       'ucx_worker_test.cpp',
                       dependencies: ucx_test_dep,
                       include_directories: ucx_test_include_directories,
                       cpp_args : '-DUSE_VRAM',
                       install: true)
    endif

endif

ucx_terminal_progress_test = executable('ucx_terminal_progress_test',
        'ucx_terminal_progress_test.cpp',
        dependencies: ucx_backend_test_dep,
        include_directories: ucx_test_include_directories)

test('ucx_terminal_progress', ucx_terminal_progress_test)

ucx_terminal_cancellation_test = executable('ucx_terminal_cancellation_test',
        'ucx_terminal_cancellation_test.cpp',
        dependencies: ucx_backend_test_dep,
        include_directories: ucx_test_include_directories)

test('ucx_terminal_cancellation', ucx_terminal_cancellation_test, timeout: 60)
