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

gtest_inc_dirs = include_directories('..')

mock_backend_sources = ['gmock_engine.cpp', 'mock_backend_plugin.cpp', 'mock_backend_engine.cpp']
mock_backend_names = ['MOCK_BACKEND', 'MOCK_BACKEND_SECONDARY', 'MOCK_BACKEND_TERTIARY']
foreach mock_backend_name : mock_backend_names
    mock_backend_plugin = shared_library(mock_backend_name, mock_backend_sources,
                   cpp_args: ['-DMOCK_BACKEND_PLUGIN_NAME="' + mock_backend_name + '"'],
                   dependencies: [nixl_infra, nixl_common_dep, gmock_dep],
                   include_directories: [nixl_inc_dirs, utils_inc_dirs, gtest_inc_dirs],
                   name_prefix: 'libplugin_',
                   install: true,
                   install_dir: plugin_install_dir)
    run_command('sh', '-c',
                'echo "' + mock_backend_name + '=' + mock_backend_plugin.full_path() +
                '" >> ' + plugin_build_dir + '/pluginlist',
                check: true)
endforeach

source_root = meson.project_source_root()
mocks_dep = declare_dependency(variables : {'path' : meson.current_source_dir().split(source_root + '/')[1]})
