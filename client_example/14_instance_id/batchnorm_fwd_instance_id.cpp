// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <functional>
#include <numeric>
#include <iomanip>
#include <iostream>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/device_reduce.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/batchnorm_forward.hpp"

using XDataType       = float;
using YDataType       = float;
using AccDataType     = float;
using ScaleDataType   = AccDataType;
using BiasDataType    = AccDataType;
using MeanVarDataType = AccDataType;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

constexpr int Rank                  = 4;
constexpr int NumBatchNormReduceDim = 3;

const double epsilon       = std::numeric_limits<float>::epsilon();
const double averageFactor = 0.1;

struct SimpleDeviceMem
{
    SimpleDeviceMem() = delete;

    SimpleDeviceMem(std::size_t mem_size) : p_mem_{}
    {
        (void)hipMalloc(static_cast<void**>(&p_mem_), mem_size);
    }

    void* GetDeviceBuffer() { return p_mem_; }

    ~SimpleDeviceMem() { (void)hipFree(p_mem_); }

    void* p_mem_;
};

// In the actual application, the instance index and name are usually from the perf db
static int instance_index = -1;
static std::string instance_name;

int main(int argc, char* argv[])
{
    std::array<ck::index_t, Rank> xyLengths{16, 8, 128, 256};
    std::array<ck::index_t, Rank> xyStrides{8 * 128 * 256, 128 * 256, 256, 1};
    std::array<ck::index_t, Rank - NumBatchNormReduceDim> scaleBiasMeanVarLengths{256};
    std::array<ck::index_t, Rank - NumBatchNormReduceDim> scaleBiasMeanVarStrides{1};
    std::array<int, NumBatchNormReduceDim> reduceDims{0, 1, 2};

    ck::index_t numXYElement =
        std::accumulate(xyLengths.begin(), xyLengths.end(), 1, std::multiplies<ck::index_t>());

    ck::index_t numScaleBiasMeanVarElement = std::accumulate(scaleBiasMeanVarLengths.begin(),
                                                             scaleBiasMeanVarLengths.end(),
                                                             1,
                                                             std::multiplies<ck::index_t>());

    SimpleDeviceMem x(sizeof(XDataType) * numXYElement);
    SimpleDeviceMem y(sizeof(YDataType) * numXYElement);
    SimpleDeviceMem scale(sizeof(ScaleDataType) * numScaleBiasMeanVarElement);
    SimpleDeviceMem bias(sizeof(BiasDataType) * numScaleBiasMeanVarElement);
    SimpleDeviceMem mean(sizeof(MeanVarDataType) * numScaleBiasMeanVarElement);
    SimpleDeviceMem invVariance(sizeof(MeanVarDataType) * numScaleBiasMeanVarElement);

    using DeviceOp = ck::tensor_operation::device::DeviceBatchNormFwd<XDataType,
                                                                      YDataType,
                                                                      AccDataType,
                                                                      ScaleDataType,
                                                                      BiasDataType,
                                                                      MeanVarDataType,
                                                                      PassThrough,
                                                                      Rank,
                                                                      NumBatchNormReduceDim>;

    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    bool found          = false;
    int best_op_index   = -1;
    float best_ave_time = std::numeric_limits<float>::max();

    // profile device operation instances and save the best performant instance index and instance
    // name
    std::cout << "Run all instances and do timing" << std::endl;

    for(int i = 0; i < op_ptrs.size(); ++i)
    {
        auto& op_ptr = op_ptrs[i];

        auto argument_ptr = op_ptr->MakeArgumentPointer(xyLengths,
                                                        xyStrides,
                                                        xyStrides,
                                                        reduceDims,
                                                        scaleBiasMeanVarLengths,
                                                        scaleBiasMeanVarStrides,
                                                        scaleBiasMeanVarStrides,
                                                        scaleBiasMeanVarStrides,
                                                        x.GetDeviceBuffer(),
                                                        scale.GetDeviceBuffer(),
                                                        bias.GetDeviceBuffer(),
                                                        epsilon,
                                                        PassThrough{},
                                                        y.GetDeviceBuffer(),
                                                        mean.GetDeviceBuffer(),
                                                        invVariance.GetDeviceBuffer(),
                                                        averageFactor,
                                                        nullptr,
                                                        nullptr);

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            size_t workspace_sz = op_ptr->GetWorkSpaceSize(argument_ptr.get());

            SimpleDeviceMem workspace(workspace_sz);

            op_ptr->SetWorkSpacePointer(argument_ptr.get(), workspace.GetDeviceBuffer());

            float ave_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});

            if(ave_time < best_ave_time)
            {
                found         = true;
                best_op_index = i;
                best_ave_time = ave_time;
            }
        }
    }

    if(found)
    {
        instance_index = best_op_index;
        instance_name  = op_ptrs[instance_index]->GetTypeIdHashCode();
    };

    // simulate the execution of the operation when the instance index and name are available
    const auto op_ptrs_2 = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    if(instance_index >= 0 && instance_index < op_ptrs_2.size())
    {
        auto& op_ptr = op_ptrs_2[instance_index];

        if(op_ptr->GetTypeIdHashCode() == instance_name)
        {

            auto argument_ptr = op_ptr->MakeArgumentPointer(xyLengths,
                                                            xyStrides,
                                                            xyStrides,
                                                            reduceDims,
                                                            scaleBiasMeanVarLengths,
                                                            scaleBiasMeanVarStrides,
                                                            scaleBiasMeanVarStrides,
                                                            scaleBiasMeanVarStrides,
                                                            x.GetDeviceBuffer(),
                                                            scale.GetDeviceBuffer(),
                                                            bias.GetDeviceBuffer(),
                                                            epsilon,
                                                            PassThrough{},
                                                            y.GetDeviceBuffer(),
                                                            mean.GetDeviceBuffer(),
                                                            invVariance.GetDeviceBuffer(),
                                                            averageFactor,
                                                            nullptr,
                                                            nullptr);

            auto invoker_ptr = op_ptr->MakeInvokerPointer();

            if(op_ptr->IsSupportedArgument(argument_ptr.get()))
            {
                size_t workspace_sz = op_ptr->GetWorkSpaceSize(argument_ptr.get());

                SimpleDeviceMem workspace(workspace_sz);

                op_ptr->SetWorkSpacePointer(argument_ptr.get(), workspace.GetDeviceBuffer());

                float exec_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});

                size_t num_bytes = numXYElement * (sizeof(XDataType) + sizeof(YDataType)) +
                                   numScaleBiasMeanVarElement *
                                       (sizeof(ScaleDataType) + sizeof(BiasDataType) +
                                        sizeof(MeanVarDataType) + sizeof(MeanVarDataType));

                float gb_per_sec = num_bytes / 1.E6 / exec_time;

                std::cout << "Kernel execution time: " << std::setw(10) << exec_time
                          << " ms,  effective data transfer bandwidth: " << gb_per_sec << " GB/s"
                          << std::endl;
            }
        };
    }

    return 0;
}
