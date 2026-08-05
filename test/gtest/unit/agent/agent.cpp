/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <random>
#include <regex>

#include "common.h"
#include "nixl.h"
#include "agent_data.h"
#include "plugin_manager.h"
#include "mocks/gmock_engine.h"

namespace gtest {
namespace agent {
    static constexpr const char *local_agent_name = "LocalAgent";
    static constexpr const char *remote_agent_name = "RemoteAgent";
    static constexpr const char *nonexisting_plugin = "NonExistingPlugin";

    /* Generates a random number in [0,255] (byte range). */
    unsigned char
    GetRandomByte() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<unsigned int> distr(0, 255);
        return static_cast<unsigned char>(distr(gen));
    }

    class blob {
    protected:
        static constexpr size_t bufLen = 256;
        static constexpr uint32_t devId = 0;

        std::unique_ptr<char[]> buf_;
        const nixlBlobDesc desc_;
        const char buf_pattern_;

    public:
        blob()
            : buf_(std::make_unique<char[]>(bufLen)),
              desc_(reinterpret_cast<uintptr_t>(buf_.get()), bufLen, devId),
              buf_pattern_(GetRandomByte()) {
            memset(buf_.get(), buf_pattern_, bufLen);
        }

        nixlBlobDesc
        getDesc() const {
            return desc_;
        }
    };

    class agentHelper {
    protected:
        testing::NiceMock<mocks::GMockBackendEngine> gmock_engine_;
        std::unique_ptr<nixlAgent> agent_;

    public:
        agentHelper(const std::string &name)
            : agent_([&name]() {
                  nixlAgentConfig cfg;
                  cfg.useProgThread = true;
                  return std::make_unique<nixlAgent>(name, cfg);
              }()) {}

        ~agentHelper() {
            /* We must release nixlAgent first (i.e. explicitly in the destructor), as it calls
               cleanup functions in gmock_engine, which must stay alive during the process. */
            agent_.reset();
        }

        nixlAgent *
        getAgent() const {
            return agent_.get();
        }

        mocks::GMockBackendEngine &
        getGMockEngine() {
            return gmock_engine_;
        }

        nixl_status_t
        createBackendWithGMock(nixl_b_params_t &params,
                               nixlBackendH *&backend,
                               const nixl_backend_t &backend_name = GetMockBackendName()) {
            gmock_engine_.SetToParams(params);
            return agent_->createBackend(backend_name, params, backend);
        }

        nixl_status_t
        getAndLoadRemoteMd(nixlAgent *remote_agent, std::string &remote_agent_name_out) {
            std::string remote_metadata;
            EXPECT_EQ(remote_agent->getLocalMD(remote_metadata), NIXL_SUCCESS);
            return agent_->loadRemoteMD(remote_metadata, remote_agent_name_out);
        }

        nixl_status_t
        getAndLoadRemoteMd(nixlAgent *remote_agent,
                           nixlRemoteAgentH *&remote_agent_handle_out) {
            std::string remote_metadata;
            EXPECT_EQ(remote_agent->getLocalMD(remote_metadata), NIXL_SUCCESS);
            return agent_->loadRemoteMD(remote_metadata, remote_agent_handle_out);
        }

        nixl_status_t
        initAndRegisterMemory(blob &blob,
                              nixl_reg_dlist_t &reg_dlist,
                              nixl_opt_args_t &extra_params,
                              nixlBackendH *backend) {
            reg_dlist.addDesc(blob.getDesc());
            extra_params.backends.push_back(backend);
            return agent_->registerMem(reg_dlist, &extra_params);
        }
    };

    class singleAgentSessionFixture : public testing::Test {
    protected:
        std::unique_ptr<agentHelper> agent_helper_;
        nixlAgent *agent_;

        void
        SetUp() override {
            agent_helper_ = std::make_unique<agentHelper>(local_agent_name);
            agent_ = agent_helper_->getAgent();
        }
    };

    class dualAgentBridgeFixture : public testing::Test {
    protected:
        std::unique_ptr<agentHelper> local_agent_helper_, remote_agent_helper_;
        nixlAgent *local_agent_, *remote_agent_;

        void
        SetUp() override {
            local_agent_helper_ = std::make_unique<agentHelper>(local_agent_name);
            remote_agent_helper_ = std::make_unique<agentHelper>(remote_agent_name);
            local_agent_ = local_agent_helper_->getAgent();
            remote_agent_ = remote_agent_helper_->getAgent();
        }

        struct DualAgentSetup {
            nixlBackendH *local_backend = nullptr;
            nixlBackendH *remote_backend = nullptr;
            blob local_blob;
            blob remote_blob;
            nixl_reg_dlist_t local_reg_dlist;
            nixl_reg_dlist_t remote_reg_dlist;
            nixl_opt_args_t local_extra_params;
            nixl_opt_args_t remote_extra_params;
            nixl_b_params_t local_params;
            nixl_b_params_t remote_params;
            std::string remote_agent_name;

            explicit DualAgentSetup(nixl_mem_t mem_type)
                : local_reg_dlist(mem_type),
                  remote_reg_dlist(mem_type) {}
        };

        void
        setupDualAgent(DualAgentSetup &s, bool register_local = true, bool register_remote = true) {
            EXPECT_EQ(local_agent_helper_->createBackendWithGMock(s.local_params, s.local_backend),
                      NIXL_SUCCESS);
            EXPECT_EQ(
                remote_agent_helper_->createBackendWithGMock(s.remote_params, s.remote_backend),
                NIXL_SUCCESS);
            if (register_local) {
                EXPECT_EQ(
                    local_agent_helper_->initAndRegisterMemory(
                        s.local_blob, s.local_reg_dlist, s.local_extra_params, s.local_backend),
                    NIXL_SUCCESS);
            }
            if (register_remote) {
                EXPECT_EQ(
                    remote_agent_helper_->initAndRegisterMemory(
                        s.remote_blob, s.remote_reg_dlist, s.remote_extra_params, s.remote_backend),
                    NIXL_SUCCESS);
            }
            EXPECT_EQ(local_agent_helper_->getAndLoadRemoteMd(remote_agent_, s.remote_agent_name),
                      NIXL_SUCCESS);
        }
    };

    class singleAgentWithMemParamFixture : public testing::TestWithParam<nixl_mem_t> {
    protected:
        std::unique_ptr<agentHelper> agent_helper_;
        nixlAgent *agent_;

        void
        SetUp() override {
            agent_helper_ = std::make_unique<agentHelper>(local_agent_name);
            agent_ = agent_helper_->getAgent();
        }
    };

    TEST_F(singleAgentSessionFixture, GetNonExistingPluginTest) {
        nixl_mem_list_t mem;
        nixl_b_params_t params;

        EXPECT_NE(agent_->getPluginParams(nonexisting_plugin, mem, params), NIXL_SUCCESS);
    }

    TEST_F(singleAgentSessionFixture, GetExistingPluginTest) {
        std::vector<nixl_backend_t> plugins;
        EXPECT_EQ(agent_->getAvailPlugins(plugins), NIXL_SUCCESS);
        if (plugins.empty()) {
            GTEST_SKIP();
        }

        nixl_mem_list_t mem;
        nixl_b_params_t params;
        EXPECT_EQ(agent_->getPluginParams(plugins.front(), mem, params), NIXL_SUCCESS);
    }

    TEST_F(singleAgentSessionFixture, CreateNonExistingPluginBackendTest) {
        nixlPluginManager &plugin_manager = nixlPluginManager::getInstance();
        EXPECT_EQ(plugin_manager.loadBackendPlugin(nonexisting_plugin), nullptr);

        nixl_b_params_t params;
        nixlBackendH *backend;
        EXPECT_NE(agent_->createBackend(nonexisting_plugin, params, backend), NIXL_SUCCESS);
    }

    TEST_F(singleAgentSessionFixture, CreateExistingPluginBackendTest) {
        nixl_mem_list_t mem;
        nixl_b_params_t params;
        EXPECT_EQ(agent_->getPluginParams(GetMockBackendName(), mem, params), NIXL_SUCCESS);

        nixlBackendH *backend;
        EXPECT_EQ(agent_helper_->createBackendWithGMock(params, backend), NIXL_SUCCESS);
    }

    TEST_F(singleAgentSessionFixture, GetNonExistingBackendParamsTest) {
        nixl_mem_list_t mem;
        nixl_b_params_t params;
        EXPECT_NE(agent_->getBackendParams(nullptr, mem, params), NIXL_SUCCESS);
    }

    TEST_F(singleAgentSessionFixture, GetExistingBackendParamsTest) {
        nixl_mem_list_t mem;
        nixl_b_params_t params;
        nixlBackendH *backend;
        EXPECT_EQ(agent_helper_->createBackendWithGMock(params, backend), NIXL_SUCCESS);
        EXPECT_EQ(agent_->getBackendParams(backend, mem, params), NIXL_SUCCESS);
    }

    TEST_F(singleAgentSessionFixture, GetLocalMetadataTest) {
        nixl_b_params_t params;
        nixlBackendH *backend;
        EXPECT_EQ(agent_helper_->createBackendWithGMock(params, backend), NIXL_SUCCESS);

        std::string metadata;
        EXPECT_EQ(agent_->getLocalMD(metadata), NIXL_SUCCESS);
        EXPECT_FALSE(metadata.empty());
    }

    TEST_P(singleAgentWithMemParamFixture, RegisterMemoryTest) {
        nixl_b_params_t params;
        nixlBackendH *backend;
        EXPECT_EQ(agent_helper_->createBackendWithGMock(params, backend), NIXL_SUCCESS);

        blob blob;
        nixl_opt_args_t extra_params;
        nixl_reg_dlist_t reg_dlist(GetParam());
        EXPECT_EQ(agent_helper_->initAndRegisterMemory(blob, reg_dlist, extra_params, backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(agent_->deregisterMem(reg_dlist, &extra_params), NIXL_SUCCESS);
    }

    TEST_F(singleAgentSessionFixture, RegisterDeregisterMemRepeatedTest) {
        constexpr int kWarmupIters = 3;
        constexpr int kTimedIters = 64;
        constexpr size_t kPoolSize = 128;

        using clock = std::chrono::steady_clock;
        using time_span = std::chrono::nanoseconds;

        nixl_opt_args_t extra_params;
        nixl_b_params_t params;
        nixlBackendH *backend;

        EXPECT_EQ(agent_helper_->createBackendWithGMock(params, backend), NIXL_SUCCESS);
        extra_params.backends.push_back(backend);

        std::vector<std::unique_ptr<blob>> pool;
        pool.resize(kPoolSize);
        for (auto &p : pool) {
            p = std::make_unique<blob>();
        }

        // Each round: registerMem once per pool entry, then deregisterMem once per entry.
        auto run_batch = [&](int rounds = 1) {
            for (int r = 0; r < rounds; ++r) {
                for (auto &bp : pool) {
                    nixl_reg_dlist_t reg_dlist{DRAM_SEG};
                    reg_dlist.addDesc(bp->getDesc());
                    EXPECT_EQ(agent_->registerMem(reg_dlist, &extra_params), NIXL_SUCCESS);
                }
                for (auto &bp : pool) {
                    nixl_reg_dlist_t reg_dlist{DRAM_SEG};
                    reg_dlist.addDesc(bp->getDesc());
                    EXPECT_EQ(agent_->deregisterMem(reg_dlist, &extra_params), NIXL_SUCCESS);
                }
            }
        };

        // Warmup
        run_batch(kWarmupIters);

        // First measurement
        auto start = clock::now();
        run_batch();
        const int64_t timed1_ns =
            std::chrono::duration_cast<time_span>(clock::now() - start).count();

        // Many cycles
        run_batch(kTimedIters);

        // Second measurement
        start = clock::now();
        run_batch();
        const int64_t timed2_ns =
            std::chrono::duration_cast<time_span>(clock::now() - start).count();

        ASSERT_GT(timed1_ns, 0);
        ASSERT_GT(timed2_ns, 0);
        const double ratio = static_cast<double>(std::max(timed1_ns, timed2_ns)) /
            static_cast<double>(std::min(timed1_ns, timed2_ns));
        EXPECT_LE(ratio, 2.) << "timed batches differ by more than 100% "
                                "(ns1="
                             << timed1_ns << " ns2=" << timed2_ns << " ratio=" << ratio << ")";
    }

    INSTANTIATE_TEST_SUITE_P(DramRegisterMemoryInstantiation,
                             singleAgentWithMemParamFixture,
                             testing::Values(DRAM_SEG));
    INSTANTIATE_TEST_SUITE_P(VramRegisterMemoryInstantiation,
                             singleAgentWithMemParamFixture,
                             testing::Values(VRAM_SEG));
    INSTANTIATE_TEST_SUITE_P(BlkRegisterMemoryInstantiation,
                             singleAgentWithMemParamFixture,
                             testing::Values(BLK_SEG));
    INSTANTIATE_TEST_SUITE_P(ObjRegisterMemoryInstantiation,
                             singleAgentWithMemParamFixture,
                             testing::Values(OBJ_SEG));
    INSTANTIATE_TEST_SUITE_P(FileRegisterMemoryInstantiation,
                             singleAgentWithMemParamFixture,
                             testing::Values(FILE_SEG));

    TEST_F(dualAgentBridgeFixture, LoadRemoteMetadataTest) {
        nixl_b_params_t local_params, remote_params;
        nixlBackendH *local_backend, *remote_backend;
        EXPECT_EQ(local_agent_helper_->createBackendWithGMock(local_params, local_backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->createBackendWithGMock(remote_params, remote_backend),
                  NIXL_SUCCESS);

        std::string remote_agent_name_out;
        EXPECT_EQ(local_agent_helper_->getAndLoadRemoteMd(remote_agent_, remote_agent_name_out),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_name, remote_agent_name_out);
    }

    TEST_F(dualAgentBridgeFixture, InvalidateRemoteMetadataTest) {
        nixl_b_params_t local_params, remote_params;
        nixlBackendH *local_backend, *remote_backend;
        EXPECT_EQ(local_agent_helper_->createBackendWithGMock(local_params, local_backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->createBackendWithGMock(remote_params, remote_backend),
                  NIXL_SUCCESS);

        nixlRemoteAgentH *remote_agent_handle = nullptr;
        EXPECT_EQ(local_agent_helper_->getAndLoadRemoteMd(remote_agent_, remote_agent_handle),
                  NIXL_SUCCESS);

        EXPECT_EQ(local_agent_->invalidateRemoteMD(remote_agent_handle), NIXL_SUCCESS);
    }

    TEST_F(dualAgentBridgeFixture, XferReqTest) {
        const std::string msg = "notification";
        EXPECT_CALL(remote_agent_helper_->getGMockEngine(), getNotifs)
            .WillOnce([=](notif_list_t &notif_list) {
                notif_list.push_back(std::make_pair(local_agent_name, msg));
                return NIXL_SUCCESS;
            });

        nixl_b_params_t local_params, remote_params;
        nixlBackendH *local_backend, *remote_backend;
        EXPECT_EQ(local_agent_helper_->createBackendWithGMock(local_params, local_backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->createBackendWithGMock(remote_params, remote_backend),
                  NIXL_SUCCESS);

        nixl_reg_dlist_t local_reg_dlist(DRAM_SEG), remote_reg_dlist(DRAM_SEG);
        nixl_opt_args_t local_extra_params, remote_extra_params;
        blob local_blob, remote_blob;
        EXPECT_EQ(local_agent_helper_->initAndRegisterMemory(
                      local_blob, local_reg_dlist, local_extra_params, local_backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->initAndRegisterMemory(
                      remote_blob, remote_reg_dlist, remote_extra_params, remote_backend),
                  NIXL_SUCCESS);

        std::string remote_agent_name_out;
        EXPECT_EQ(local_agent_helper_->getAndLoadRemoteMd(remote_agent_, remote_agent_name_out),
                  NIXL_SUCCESS);

        nixl_xfer_dlist_t local_xfer_dlist(DRAM_SEG), remote_xfer_dlist(DRAM_SEG);
        local_xfer_dlist.addDesc(local_blob.getDesc());
        remote_xfer_dlist.addDesc(remote_blob.getDesc());

        nixlXferReqH *xfer_req;
        local_extra_params.notif = msg;
        EXPECT_EQ(local_agent_->createXferReq(NIXL_WRITE,
                                              local_xfer_dlist,
                                              remote_xfer_dlist,
                                              remote_agent_name_out,
                                              xfer_req,
                                              &local_extra_params),
                  NIXL_SUCCESS);
        EXPECT_EQ(local_agent_->postXferReq(xfer_req), NIXL_SUCCESS);
        EXPECT_EQ(local_agent_->getXferStatus(xfer_req), NIXL_SUCCESS);

        nixl_notifs_t notif_map;
        EXPECT_EQ(remote_agent_->getNotifs(notif_map), NIXL_SUCCESS);
        EXPECT_EQ(notif_map.size(), 1u);
        EXPECT_EQ(notif_map[local_agent_name].size(), 1u);
        EXPECT_EQ(notif_map[local_agent_name].front(), msg);

        EXPECT_EQ(local_agent_->releaseXferReq(xfer_req), NIXL_SUCCESS);
    }

    TEST_F(dualAgentBridgeFixture, PrepMemViewRemoteDRAM) {
        DualAgentSetup s(DRAM_SEG);
        setupDualAgent(s, /*register_local=*/false);

        nixl_remote_dlist_t remote_dlist(DRAM_SEG);
        remote_dlist.addDesc(nixlRemoteDesc(s.remote_blob.getDesc(), s.remote_agent_name));

        nixlMemViewH mvh = nullptr;
        EXPECT_EQ(local_agent_->prepMemView(remote_dlist, mvh), NIXL_SUCCESS);
        EXPECT_NE(mvh, nullptr);

        local_agent_->releaseMemView(mvh);
    }

    TEST_F(dualAgentBridgeFixture, XferReqSubFunctionsTest) {
        const std::string msg = "notification";
        EXPECT_CALL(remote_agent_helper_->getGMockEngine(), getNotifs)
            .WillOnce([=](notif_list_t &notif_list) {
                notif_list.push_back(std::make_pair(local_agent_name, msg));
                return NIXL_SUCCESS;
            });

        nixl_b_params_t local_params, remote_params;
        nixlBackendH *local_backend, *remote_backend;
        EXPECT_EQ(local_agent_helper_->createBackendWithGMock(local_params, local_backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->createBackendWithGMock(remote_params, remote_backend),
                  NIXL_SUCCESS);

        nixl_reg_dlist_t local_reg_dlist(DRAM_SEG), remote_reg_dlist(DRAM_SEG);
        nixl_opt_args_t local_extra_params, remote_extra_params;
        blob local_blob, remote_blob;
        EXPECT_EQ(local_agent_helper_->initAndRegisterMemory(
                      local_blob, local_reg_dlist, local_extra_params, local_backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->initAndRegisterMemory(
                      remote_blob, remote_reg_dlist, remote_extra_params, remote_backend),
                  NIXL_SUCCESS);

        std::string remote_agent_name_out;
        EXPECT_EQ(local_agent_helper_->getAndLoadRemoteMd(remote_agent_, remote_agent_name_out),
                  NIXL_SUCCESS);

        nixl_xfer_dlist_t local_xfer_dlist(DRAM_SEG), remote_xfer_dlist(DRAM_SEG);
        local_xfer_dlist.addDesc(local_blob.getDesc());
        remote_xfer_dlist.addDesc(remote_blob.getDesc());

        nixlDlistH *desc_hndl1, *desc_hndl2;
        EXPECT_EQ(local_agent_->prepXferDlist(local_xfer_dlist, desc_hndl1), NIXL_SUCCESS);
        EXPECT_EQ(local_agent_->prepXferDlist(remote_agent_name_out, remote_xfer_dlist, desc_hndl2),
                  NIXL_SUCCESS);

        std::vector<int> indices;
        for (int i = 0; i < local_xfer_dlist.descCount(); i++)
            indices.push_back(i);

        nixlXferReqH *xfer_req;
        local_extra_params.notif = msg;
        EXPECT_EQ(local_agent_->makeXferReq(NIXL_WRITE,
                                            desc_hndl1,
                                            indices,
                                            desc_hndl2,
                                            indices,
                                            xfer_req,
                                            &local_extra_params),
                  NIXL_SUCCESS);
        EXPECT_EQ(local_agent_->postXferReq(xfer_req), NIXL_SUCCESS);

        EXPECT_EQ(local_agent_->getXferStatus(xfer_req), NIXL_SUCCESS);

        nixl_notifs_t notif_map;
        EXPECT_EQ(remote_agent_->getNotifs(notif_map), NIXL_SUCCESS);
        EXPECT_EQ(notif_map.size(), 1u);
        EXPECT_EQ(notif_map[local_agent_name].size(), 1u);
        EXPECT_EQ(notif_map[local_agent_name].front(), msg);

        EXPECT_EQ(local_agent_->releaseXferReq(xfer_req), NIXL_SUCCESS);
        EXPECT_EQ(local_agent_->releasedDlistH(desc_hndl1), NIXL_SUCCESS);
        EXPECT_EQ(local_agent_->releasedDlistH(desc_hndl2), NIXL_SUCCESS);
    }

    TEST_F(dualAgentBridgeFixture, GenNotifTest) {
        const std::string msg = "notification";
        EXPECT_CALL(remote_agent_helper_->getGMockEngine(), getNotifs)
            .WillOnce([=](notif_list_t &notif_list) {
                notif_list.push_back(std::make_pair(local_agent_name, msg));
                return NIXL_SUCCESS;
            });

        ON_CALL(local_agent_helper_->getGMockEngine(), supportsAuthenticatedNotif())
            .WillByDefault(testing::Return(true));

        nixl_b_params_t local_params, remote_params;
        nixlBackendH *local_backend, *remote_backend;
        EXPECT_EQ(local_agent_helper_->createBackendWithGMock(local_params, local_backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->createBackendWithGMock(remote_params, remote_backend),
                  NIXL_SUCCESS);

        nixlRemoteAgentH *remote_agent_handle = nullptr;
        EXPECT_EQ(local_agent_helper_->getAndLoadRemoteMd(remote_agent_, remote_agent_handle),
                  NIXL_SUCCESS);
        EXPECT_EQ(local_agent_->genNotif(remote_agent_handle, msg), NIXL_SUCCESS);

        nixl_notifs_t notif_map;
        EXPECT_EQ(remote_agent_->getNotifs(notif_map), NIXL_SUCCESS);
        EXPECT_EQ(notif_map.size(), 1u);
        EXPECT_EQ(notif_map[local_agent_name].size(), 1u);
        EXPECT_EQ(notif_map[local_agent_name].front(), msg);
    }

    TEST_F(dualAgentBridgeFixture, ExactRemoteBindingLifecycleTest) {
        auto &backend = local_agent_helper_->getGMockEngine();
        ON_CALL(backend, supportsAuthenticatedNotif())
            .WillByDefault(testing::Return(true));

        EXPECT_CALL(backend, backendInit(local_agent_name, testing::_))
            .WillOnce([](const std::string &, const std::string &incarnation) {
                static const std::regex uuid_v4(
                    "[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}");
                EXPECT_TRUE(std::regex_match(incarnation, uuid_v4));
            });

        nixlRemoteAgentBinding installed_binding;
        EXPECT_CALL(backend, bindRemoteAgent(testing::_))
            .WillOnce([&installed_binding](const nixlRemoteAgentBinding &binding) {
                installed_binding = binding;
                return NIXL_SUCCESS;
            });

        nixl_b_params_t local_params, remote_params;
        nixlBackendH *local_backend, *remote_backend;
        EXPECT_EQ(local_agent_helper_->createBackendWithGMock(local_params, local_backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->createBackendWithGMock(remote_params, remote_backend),
                  NIXL_SUCCESS);

        nixlRemoteAgentH *remote_handle = nullptr;
        EXPECT_EQ(local_agent_helper_->getAndLoadRemoteMd(remote_agent_, remote_handle),
                  NIXL_SUCCESS);
        ASSERT_NE(remote_handle, nullptr);
        EXPECT_EQ(installed_binding.remoteAgent, remote_agent_name);
        EXPECT_EQ(installed_binding.authority.handleIdentity, remote_handle->getIdentity());
        EXPECT_EQ(installed_binding.authority.generation, remote_handle->getGeneration());

        const std::string message = "exact-generation";
        EXPECT_CALL(backend, queryRemoteNotificationState(testing::_))
            .WillOnce([&installed_binding](const nixlRemoteAgentBinding &binding) {
                EXPECT_EQ(binding.authority.handleIdentity,
                          installed_binding.authority.handleIdentity);
                EXPECT_EQ(binding.authority.generation,
                          installed_binding.authority.generation);
                return NIXL_SUCCESS;
            });
        EXPECT_CALL(backend,
                    genNotif(testing::A<const nixlRemoteAgentBinding &>(), message))
            .WillOnce([&installed_binding](const nixlRemoteAgentBinding &binding,
                                           const std::string &) {
                EXPECT_EQ(binding.authority.handleIdentity,
                          installed_binding.authority.handleIdentity);
                EXPECT_EQ(binding.authority.generation,
                          installed_binding.authority.generation);
                return NIXL_SUCCESS;
            });
        EXPECT_EQ(local_agent_->genNotif(remote_handle, message), NIXL_SUCCESS);

        {
            testing::InSequence sequence;
            EXPECT_CALL(backend, retireRemoteAgent(testing::_))
                .WillOnce([&installed_binding](const nixlRemoteAgentBinding &binding) {
                    EXPECT_EQ(binding.authority.handleIdentity,
                              installed_binding.authority.handleIdentity);
                    EXPECT_EQ(binding.authority.generation,
                              installed_binding.authority.generation);
                    return NIXL_SUCCESS;
                });
            EXPECT_CALL(backend, disconnect(remote_agent_name))
                .WillOnce(testing::Return(NIXL_SUCCESS));
        }
        EXPECT_EQ(local_agent_->invalidateRemoteMD(remote_handle), NIXL_SUCCESS);
    }

    TEST_F(dualAgentBridgeFixture, GenerationSpecificNotificationSendingTest) {
        auto &backend = local_agent_helper_->getGMockEngine();
        ON_CALL(backend, supportsAuthenticatedNotif())
            .WillByDefault(testing::Return(true));

        std::vector<nixlRemoteAgentBinding> installed_bindings;
        EXPECT_CALL(backend, bindRemoteAgent(testing::_))
            .Times(2)
            .WillRepeatedly([&installed_bindings](const nixlRemoteAgentBinding &binding) {
                installed_bindings.push_back(binding);
                return NIXL_SUCCESS;
            });

        nixl_b_params_t local_params, remote_params;
        nixlBackendH *local_backend, *remote_backend;
        EXPECT_EQ(local_agent_helper_->createBackendWithGMock(local_params, local_backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->createBackendWithGMock(remote_params, remote_backend),
                  NIXL_SUCCESS);

        std::string remote_metadata;
        EXPECT_EQ(remote_agent_->getLocalMD(remote_metadata), NIXL_SUCCESS);

        nixlRemoteAgentH *generation_one = nullptr;
        EXPECT_EQ(local_agent_->loadRemoteMD(remote_metadata, generation_one), NIXL_SUCCESS);
        ASSERT_NE(generation_one, nullptr);

        EXPECT_CALL(backend, queryRemoteNotificationState(testing::_))
            .Times(2)
            .WillRepeatedly(testing::Return(NIXL_SUCCESS));
        std::vector<uint64_t> sent_generations;
        EXPECT_CALL(backend,
                    genNotif(testing::A<const nixlRemoteAgentBinding &>(), testing::_))
            .Times(2)
            .WillRepeatedly([&sent_generations](const nixlRemoteAgentBinding &binding,
                                                const std::string &) {
                sent_generations.push_back(binding.authority.generation);
                return NIXL_SUCCESS;
            });

        EXPECT_EQ(local_agent_->genNotif(generation_one, "g1"), NIXL_SUCCESS);
        EXPECT_EQ(local_agent_->invalidateRemoteMD(generation_one), NIXL_SUCCESS);
        EXPECT_EQ(local_agent_->genNotif(generation_one, "retired"), NIXL_ERR_NOT_ALLOWED);

        nixlRemoteAgentH *generation_two = nullptr;
        EXPECT_EQ(local_agent_->loadRemoteMD(remote_metadata, generation_two), NIXL_SUCCESS);
        ASSERT_NE(generation_two, nullptr);
        EXPECT_NE(generation_one->getIdentity(), generation_two->getIdentity());
        EXPECT_EQ(generation_two->getGeneration(), generation_one->getGeneration() + 1);
        EXPECT_EQ(local_agent_->genNotif(generation_two, "g2"), NIXL_SUCCESS);
        EXPECT_EQ(local_agent_->invalidateRemoteMD(generation_two), NIXL_SUCCESS);

        ASSERT_EQ(sent_generations.size(), 2u);
        EXPECT_EQ(sent_generations[0], generation_one->getGeneration());
        EXPECT_EQ(sent_generations[1], generation_two->getGeneration());
    }

    TEST_F(dualAgentBridgeFixture, NotificationReadinessGatesTransferBeforePostTest) {
        auto &backend = local_agent_helper_->getGMockEngine();
        ON_CALL(backend, supportsAuthenticatedNotif())
            .WillByDefault(testing::Return(true));

        nixl_b_params_t local_params, remote_params;
        nixlBackendH *local_backend, *remote_backend;
        EXPECT_EQ(local_agent_helper_->createBackendWithGMock(local_params, local_backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->createBackendWithGMock(remote_params, remote_backend),
                  NIXL_SUCCESS);

        nixl_reg_dlist_t local_reg_dlist(DRAM_SEG), remote_reg_dlist(DRAM_SEG);
        nixl_opt_args_t local_mem_args, remote_mem_args;
        blob local_blob, remote_blob;
        EXPECT_EQ(local_agent_helper_->initAndRegisterMemory(
                      local_blob, local_reg_dlist, local_mem_args, local_backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->initAndRegisterMemory(
                      remote_blob, remote_reg_dlist, remote_mem_args, remote_backend),
                  NIXL_SUCCESS);

        nixlBackendMD loaded_remote_metadata(false);
        ON_CALL(backend, loadRemoteMD(testing::_, testing::_, testing::_, testing::_))
            .WillByDefault([&loaded_remote_metadata](const nixlBlobDesc &,
                                                     const nixl_mem_t &,
                                                     const std::string &,
                                                     nixlBackendMD *&output) {
                output = &loaded_remote_metadata;
                return NIXL_SUCCESS;
            });

        nixlRemoteAgentH *remote_handle = nullptr;
        EXPECT_EQ(local_agent_helper_->getAndLoadRemoteMd(remote_agent_, remote_handle),
                  NIXL_SUCCESS);

        nixl_xfer_dlist_t local_xfer_dlist(DRAM_SEG), remote_xfer_dlist(DRAM_SEG);
        local_xfer_dlist.addDesc(local_blob.getDesc());
        remote_xfer_dlist.addDesc(remote_blob.getDesc());
        nixl_opt_args_t xfer_args;
        xfer_args.notif = "not-ready";
        nixlXferReqH *request = nullptr;
        EXPECT_EQ(local_agent_->createXferReq(NIXL_WRITE,
                                              local_xfer_dlist,
                                              remote_xfer_dlist,
                                              remote_handle,
                                              request,
                                              &xfer_args),
                  NIXL_SUCCESS);

        EXPECT_CALL(backend, queryRemoteNotificationState(testing::_))
            .WillOnce(testing::Return(NIXL_ERR_NOT_READY));
        EXPECT_CALL(backend, postXfer(testing::_,
                                     testing::_,
                                     testing::_,
                                     testing::_,
                                     testing::_,
                                     testing::_))
            .Times(0);
        EXPECT_EQ(local_agent_->postXferReq(request), NIXL_ERR_NOT_READY);
        EXPECT_EQ(local_agent_->releaseXferReq(request), NIXL_SUCCESS);
        EXPECT_EQ(local_agent_->invalidateRemoteMD(remote_handle), NIXL_SUCCESS);
    }

    TEST_F(dualAgentBridgeFixture, TombstonedAndActiveNotificationsSurviveInvalidSiblingTest) {
        auto &backend = local_agent_helper_->getGMockEngine();
        ON_CALL(backend, supportsAuthenticatedNotif())
            .WillByDefault(testing::Return(true));

        nixl_b_params_t local_params, remote_params;
        nixlBackendH *local_backend, *remote_backend;
        EXPECT_EQ(local_agent_helper_->createBackendWithGMock(local_params, local_backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->createBackendWithGMock(remote_params, remote_backend),
                  NIXL_SUCCESS);

        std::string remote_metadata;
        EXPECT_EQ(remote_agent_->getLocalMD(remote_metadata), NIXL_SUCCESS);
        nixlRemoteAgentH *generation_one = nullptr;
        EXPECT_EQ(local_agent_->loadRemoteMD(remote_metadata, generation_one), NIXL_SUCCESS);
        EXPECT_EQ(local_agent_->invalidateRemoteMD(generation_one), NIXL_SUCCESS);

        nixlRemoteAgentH *generation_two = nullptr;
        EXPECT_EQ(local_agent_->loadRemoteMD(remote_metadata, generation_two), NIXL_SUCCESS);

        EXPECT_CALL(backend, getAuthenticatedNotifs(testing::_))
            .WillOnce([generation_one, generation_two](authenticated_notif_list_t &notifications) {
                notifications.push_back({.payload = "retired-generation",
                                         .handleIdentity = generation_one->getIdentity(),
                                         .generation = generation_one->getGeneration()});
                notifications.push_back({.payload = "invalid",
                                         .handleIdentity = UINT64_MAX,
                                         .generation = 1});
                notifications.push_back({.payload = "active-generation",
                                         .handleIdentity = generation_two->getIdentity(),
                                         .generation = generation_two->getGeneration()});
                return NIXL_SUCCESS;
            });

        nixl_remote_notifs_t notifications;
        EXPECT_EQ(local_agent_->getRemoteNotifs(notifications), NIXL_ERR_NOT_ALLOWED);
        ASSERT_EQ(notifications[generation_one].size(), 1u);
        EXPECT_EQ(notifications[generation_one][0], "retired-generation");
        ASSERT_EQ(notifications[generation_two].size(), 1u);
        EXPECT_EQ(notifications[generation_two][0], "active-generation");
        EXPECT_EQ(local_agent_->invalidateRemoteMD(generation_two), NIXL_SUCCESS);
    }

    TEST_F(dualAgentBridgeFixture, RemoteBindingRollbackIsReverseTest) {
        auto &backend = local_agent_helper_->getGMockEngine();
        ON_CALL(backend, supportsAuthenticatedNotif())
            .WillByDefault(testing::Return(true));

        uint64_t next_connection_identity = 10;
        ON_CALL(backend, queryRemoteAgentAuthority(testing::_, testing::_))
            .WillByDefault([&next_connection_identity](
                               const std::string &,
                               nixl_remote_agent_authority_t &authority) {
                authority.connectionIdentity = next_connection_identity++;
                authority.endpointIdentities = {authority.connectionIdentity + 100};
                return NIXL_SUCCESS;
            });

        std::vector<uint64_t> bound_connections;
        EXPECT_CALL(backend, bindRemoteAgent(testing::_))
            .Times(3)
            .WillRepeatedly([&bound_connections](const nixlRemoteAgentBinding &binding) {
                bound_connections.push_back(binding.authority.connectionIdentity);
                return bound_connections.size() == 3 ? NIXL_ERR_BACKEND : NIXL_SUCCESS;
            });
        std::vector<uint64_t> retired_connections;
        EXPECT_CALL(backend, retireRemoteAgent(testing::_))
            .Times(2)
            .WillRepeatedly([&retired_connections](const nixlRemoteAgentBinding &binding) {
                retired_connections.push_back(binding.authority.connectionIdentity);
                return NIXL_SUCCESS;
            });
        EXPECT_CALL(backend, disconnect(remote_agent_name))
            .Times(3)
            .WillRepeatedly(testing::Return(NIXL_SUCCESS));

        const std::array<nixl_backend_t, 3> backend_names = {
            GetMockBackendName(),
            GetSecondaryMockBackendName(),
            GetTertiaryMockBackendName(),
        };
        std::array<nixl_b_params_t, 3> local_params;
        std::array<nixl_b_params_t, 3> remote_params;
        std::array<nixlBackendH *, 3> local_backends;
        std::array<nixlBackendH *, 3> remote_backends;
        for (size_t i = 0; i < backend_names.size(); ++i) {
            EXPECT_EQ(local_agent_helper_->createBackendWithGMock(
                          local_params[i], local_backends[i], backend_names[i]),
                      NIXL_SUCCESS);
            EXPECT_EQ(remote_agent_helper_->createBackendWithGMock(
                          remote_params[i], remote_backends[i], backend_names[i]),
                      NIXL_SUCCESS);
        }

        std::string remote_metadata;
        EXPECT_EQ(remote_agent_->getLocalMD(remote_metadata), NIXL_SUCCESS);
        nixlRemoteAgentH *remote_handle = nullptr;
        EXPECT_EQ(local_agent_->loadRemoteMD(remote_metadata, remote_handle), NIXL_ERR_BACKEND);
        EXPECT_EQ(remote_handle, nullptr);

        ASSERT_EQ(bound_connections.size(), 3u);
        ASSERT_EQ(retired_connections.size(), 2u);
        EXPECT_EQ(retired_connections[0], bound_connections[1]);
        EXPECT_EQ(retired_connections[1], bound_connections[0]);
    }

    TEST_F(dualAgentBridgeFixture, QueryXferBackendTest) {
        nixl_b_params_t local_params, remote_params;
        nixlBackendH *local_backend, *remote_backend;
        EXPECT_EQ(local_agent_helper_->createBackendWithGMock(local_params, local_backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->createBackendWithGMock(remote_params, remote_backend),
                  NIXL_SUCCESS);

        nixl_reg_dlist_t local_reg_dlist(DRAM_SEG), remote_reg_dlist(DRAM_SEG);
        nixl_opt_args_t local_extra_params, remote_extra_params;
        blob local_blob, remote_blob;
        EXPECT_EQ(local_agent_helper_->initAndRegisterMemory(
                      local_blob, local_reg_dlist, local_extra_params, local_backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->initAndRegisterMemory(
                      remote_blob, remote_reg_dlist, remote_extra_params, remote_backend),
                  NIXL_SUCCESS);

        std::string remote_agent_name_out;
        EXPECT_EQ(local_agent_helper_->getAndLoadRemoteMd(remote_agent_, remote_agent_name_out),
                  NIXL_SUCCESS);

        nixl_xfer_dlist_t local_xfer_dlist(DRAM_SEG), remote_xfer_dlist(DRAM_SEG);
        local_xfer_dlist.addDesc(local_blob.getDesc());
        remote_xfer_dlist.addDesc(remote_blob.getDesc());

        nixlXferReqH *xfer_req;
        EXPECT_EQ(local_agent_->createXferReq(NIXL_WRITE,
                                              local_xfer_dlist,
                                              remote_xfer_dlist,
                                              remote_agent_name_out,
                                              xfer_req,
                                              &local_extra_params),
                  NIXL_SUCCESS);

        nixlBackendH *backend_out;
        EXPECT_EQ(local_agent_->queryXferBackend(xfer_req, backend_out), NIXL_SUCCESS);
        EXPECT_EQ(backend_out, local_backend);

        EXPECT_EQ(local_agent_->releaseXferReq(xfer_req), NIXL_SUCCESS);
    }

    TEST_F(dualAgentBridgeFixture, MakeConnectionTest) {
        nixl_b_params_t local_params, remote_params;
        nixlBackendH *local_backend, *remote_backend;
        EXPECT_EQ(local_agent_helper_->createBackendWithGMock(local_params, local_backend),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->createBackendWithGMock(remote_params, remote_backend),
                  NIXL_SUCCESS);

        std::string local_agent_name_out, remote_agent_name_out;
        EXPECT_EQ(local_agent_helper_->getAndLoadRemoteMd(remote_agent_, remote_agent_name_out),
                  NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_helper_->getAndLoadRemoteMd(local_agent_, local_agent_name_out),
                  NIXL_SUCCESS);

        EXPECT_EQ(local_agent_->makeConnection(remote_agent_name_out), NIXL_SUCCESS);
        EXPECT_EQ(remote_agent_->makeConnection(local_agent_name_out), NIXL_SUCCESS);
    }

} // namespace agent
} // namespace gtest
