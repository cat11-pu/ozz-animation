//----------------------------------------------------------------------------//
//                                                                            //
// ozz-animation is hosted at http://github.com/guillaumeblanc/ozz-animation  //
// and distributed under the MIT License (MIT).                               //
//                                                                            //
// Copyright (c) Guillaume Blanc                                              //
//                                                                            //
// Permission is hereby granted, free of charge, to any person obtaining a    //
// copy of this software and associated documentation files (the "Software"), //
// to deal in the Software without restriction, including without limitation  //
// the rights to use, copy, modify, merge, publish, distribute, sublicense,   //
// and/or sell copies of the Software, and to permit persons to whom the      //
// Software is furnished to do so, subject to the following conditions:       //
//                                                                            //
// The above copyright notice and this permission notice shall be included in //
// all copies or substantial portions of the Software.                        //
//                                                                            //
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR //
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   //
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    //
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER //
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    //
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        //
// DEALINGS IN THE SOFTWARE.                                                  //
//                                                                            //
//----------------------------------------------------------------------------//

#include "ozz/animation/runtime/blending_job.h"

#include <cassert>
#include <cstddef>

#include "ozz/animation/runtime/skeleton.h"
#include "ozz/base/maths/math_ex.h"
#include "ozz/base/maths/soa_transform.h"

namespace ozz {
namespace animation {

namespace {
bool ValidateLayer(const BlendingJob::Layer& _layer, size_t _min_range) {
  bool valid = true;

  // Tests transforms validity.
  valid &= _layer.transform.size() >= _min_range;

  // Joint weights are optional.
  if (!_layer.joint_weights.empty()) {
    valid &= _layer.joint_weights.size() >= _min_range;
  } else {
    valid &= _layer.joint_weights.empty();
  }
  return valid;
}
}  // namespace

bool BlendingJob::Validate() const {
  // Don't need any early out, as jobs are valid in most of the performance
  // critical cases.
  // Tests are written in multiple lines in order to avoid branches.
  bool valid = true;

  // Test for valid threshold).
  valid &= threshold > 0.f;

  // Test for nullptr begin pointers.
  // Blending layers are mandatory, additive aren't.
  valid &= !rest_pose.empty();
  valid &= !output.empty();

  // The rest pose size defines the ranges of transforms to blend, so all
  // other buffers should be bigger.
  const size_t min_range = rest_pose.size();
  valid &= output.size() >= min_range;

  // Validates layers.
  for (const Layer& layer : layers) {
    valid &= ValidateLayer(layer, min_range);
  }

  // Validates additive layers.
  for (const Layer& layer : additive_layers) {
    valid &= ValidateLayer(layer, min_range);
  }

  return valid;
}

namespace {

// Macro that defines the process of blending the 1st pass.
#define OZZ_BLEND_1ST_PASS(_in, _simd_weight, _out)    \
  do {                                                 \
    _out.translation = _in.translation * _simd_weight; \
    _out.rotation = _in.rotation * _simd_weight;       \
    _out.scale = _in.scale * _simd_weight;             \
  } while (void(0), 0)

// Macro that defines the process of blending any pass but the first.
#define OZZ_BLEND_N_PASS(_in, _simd_weight, _out)                             \
  do {                                                                        \
    /* Blends translation. */                                                 \
    _out.translation = _out.translation + _in.translation * _simd_weight;     \
    /* Blends rotations, negates opposed quaternions to be sure to choose*/   \
    /* the shortest path between the two.*/                                   \
    const math::SimdInt4 sign = math::Sign(Dot(_out.rotation, _in.rotation)); \
    const math::SoaQuaternion rotation = {                                    \
        math::Xor(_in.rotation.x, sign), math::Xor(_in.rotation.y, sign),     \
        math::Xor(_in.rotation.z, sign), math::Xor(_in.rotation.w, sign)};    \
    _out.rotation = _out.rotation + rotation * _simd_weight;                  \
    /* Blends scales.*/                                                       \
    _out.scale = _out.scale + _in.scale * _simd_weight;                       \
  } while (void(0), 0)

// Macro that defines the process of adding a pass.
#define OZZ_ADD_PASS(_in, _simd_weight, _out)                               \
  do {                                                                      \
    _out.translation = _out.translation + _in.translation * _simd_weight;   \
    /* Interpolate quaternion between identity and src.rotation.*/          \
    /* Quaternion sign is fixed up, so that lerp takes the shortest path.*/ \
    const math::SimdInt4 sign = math::Sign(_in.rotation.w);                 \
    const math::SoaQuaternion interp_quat = {                               \
        math::Xor(_in.rotation.x, sign) * _simd_weight,                     \
        math::Xor(_in.rotation.y, sign) * _simd_weight,                     \
        math::Xor(_in.rotation.z, sign) * _simd_weight,                     \
        (math::Xor(_in.rotation.w, sign) - one) * _simd_weight + one};      \
    _out.rotation = _out.rotation * NormalizeEst(interp_quat);              \
    _out.scale.x = _out.scale.x *                                           \
                   math::MAdd(_in.scale.x, _simd_weight, one_minus_weight); \
    _out.scale.y = _out.scale.y *                                           \
                   math::MAdd(_in.scale.y, _simd_weight, one_minus_weight); \
    _out.scale.z = _out.scale.z *                                           \
                   math::MAdd(_in.scale.z, _simd_weight, one_minus_weight); \
  } while (void(0), 0)

// Macro that defines the process of subtracting a pass.
#define OZZ_SUB_PASS(_in, _simd_weight, _out)                                  \
  do {                                                                         \
    _out.translation = _out.translation - _in.translation * _simd_weight;      \
    /* Interpolate quaternion between identity and src.rotation.*/             \
    /* Quaternion sign is fixed up, so that lerp takes the shortest path.*/    \
    const math::SimdInt4 sign = math::Sign(_in.rotation.w);                    \
    const math::SoaQuaternion interp_quat = {                                  \
        math::Xor(_in.rotation.x, sign) * _simd_weight,                        \
        math::Xor(_in.rotation.y, sign) * _simd_weight,                        \
        math::Xor(_in.rotation.z, sign) * _simd_weight,                        \
        (math::Xor(_in.rotation.w, sign) - one) * _simd_weight + one};         \
    _out.rotation = _out.rotation * Conjugate(NormalizeEst(interp_quat));      \
    _out.scale.x =                                                             \
        _out.scale.x *                                                         \
        math::RcpEst(math::MAdd(_in.scale.x, _simd_weight, one_minus_weight)); \
    _out.scale.y =                                                             \
        _out.scale.y *                                                         \
        math::RcpEst(math::MAdd(_in.scale.y, _simd_weight, one_minus_weight)); \
    _out.scale.z =                                                             \
        _out.scale.z *                                                         \
        math::RcpEst(math::MAdd(_in.scale.z, _simd_weight, one_minus_weight)); \
  } while (void(0), 0)

// Collects pass statistics without touching the output buffers.
struct PassStats {
  size_t num_passes;
  size_t num_partial_passes;
  float accumulated_weight;
};

OZZ_INLINE void AddLayers(const BlendingJob& _job, size_t _i,
                          const math::SimdFloat4& _one,
                          math::SoaTransform* _dest);

PassStats PreparePasses(const BlendingJob& _job) {
  PassStats stats = {0, 0, 0.f};
  for (const BlendingJob::Layer& layer : _job.layers) {
    if (layer.weight <= 0.f) {
      continue;
    }
    stats.accumulated_weight += layer.weight;
    stats.num_partial_passes += !layer.joint_weights.empty();
    ++stats.num_passes;
  }
  return stats;
}

// Blends normal layers, rest pose and normalization in a single output pass.
void BlendLayers(const BlendingJob& _job, const PassStats& _stats) {
  const size_t num_soa_joints = _job.rest_pose.size();
  assert(_job.output.size() >= num_soa_joints);
  const math::SimdFloat4 one = math::simd_float4::one();

  if (_stats.num_partial_passes == 0) {
    // No partial blending pass, threshold and normalization can be computed
    // once for all joints.
    const float blp_weight = _job.threshold - _stats.accumulated_weight;
    const math::SimdFloat4 ratio =
        math::simd_float4::Load1(1.f / (blp_weight > 0.f
                                           ? (_stats.num_passes == 0
                                                  ? 1.f
                                                  : _job.threshold)
                                           : _stats.accumulated_weight));
    const bool blend_rest = blp_weight > 0.f;
    const math::SimdFloat4 rest_weight =
        math::simd_float4::Load1(blp_weight);

    const size_t num_layers = _job.layers.size();
    size_t layer_index = 0;
    const auto next_layer = [&]() -> const BlendingJob::Layer* {
      while (layer_index < num_layers &&
             _job.layers[layer_index].weight <= 0.f) {
        ++layer_index;
      }
      return layer_index < num_layers ? &_job.layers[layer_index++] : nullptr;
    };

    bool first_pair = true;
    for (const BlendingJob::Layer* layer0 = next_layer(); layer0 != nullptr;
         first_pair = false) {
      assert(layer0->transform.size() >= num_soa_joints);
      const BlendingJob::Layer* layer1 = next_layer();
      if (layer1 != nullptr) {
        assert(layer1->transform.size() >= num_soa_joints);
      }

      const math::SimdFloat4 weight0 =
          math::simd_float4::Load1(layer0->weight);
      const math::SimdFloat4 weight1 =
          layer1 != nullptr ? math::simd_float4::Load1(layer1->weight)
                            : math::simd_float4::zero();
      if (first_pair) {
        for (size_t i = 0; i < num_soa_joints; ++i) {
          math::SoaTransform dest;
          OZZ_BLEND_1ST_PASS(layer0->transform[i], weight0, dest);
          if (layer1 != nullptr) {
            OZZ_BLEND_N_PASS(layer1->transform[i], weight1, dest);
          }
          _job.output[i] = dest;
        }
      } else if (layer1 != nullptr) {
        for (size_t i = 0; i < num_soa_joints; ++i) {
          math::SoaTransform dest = _job.output[i];
          OZZ_BLEND_N_PASS(layer0->transform[i], weight0, dest);
          OZZ_BLEND_N_PASS(layer1->transform[i], weight1, dest);
          _job.output[i] = dest;
        }
      } else {
        for (size_t i = 0; i < num_soa_joints; ++i) {
          math::SoaTransform dest = _job.output[i];
          OZZ_BLEND_N_PASS(layer0->transform[i], weight0, dest);
          _job.output[i] = dest;
        }
      }
      layer0 = next_layer();
    }

    if (blend_rest) {
      if (_stats.num_passes == 0) {
        size_t i = 0;
        for (; i + 1 < num_soa_joints; i += 2) {
          math::SoaTransform dest = _job.rest_pose[i];
          dest.rotation = NormalizeEst(dest.rotation);
          dest.translation = dest.translation * ratio;
          dest.scale = dest.scale * ratio;
          AddLayers(_job, i, one, &dest);
          _job.output[i] = dest;
          dest = _job.rest_pose[i + 1];
          dest.rotation = NormalizeEst(dest.rotation);
          dest.translation = dest.translation * ratio;
          dest.scale = dest.scale * ratio;
          AddLayers(_job, i + 1, one, &dest);
          _job.output[i + 1] = dest;
        }
        if (i < num_soa_joints) {
          math::SoaTransform dest = _job.rest_pose[i];
          dest.rotation = NormalizeEst(dest.rotation);
          dest.translation = dest.translation * ratio;
          dest.scale = dest.scale * ratio;
          AddLayers(_job, i, one, &dest);
          _job.output[i] = dest;
        }
      } else {
        size_t i = 0;
        for (; i + 1 < num_soa_joints; i += 2) {
          math::SoaTransform dest = _job.output[i];
          OZZ_BLEND_N_PASS(_job.rest_pose[i], rest_weight, dest);
          dest.rotation = NormalizeEst(dest.rotation);
          dest.translation = dest.translation * ratio;
          dest.scale = dest.scale * ratio;
          AddLayers(_job, i, one, &dest);
          _job.output[i] = dest;
          dest = _job.output[i + 1];
          OZZ_BLEND_N_PASS(_job.rest_pose[i + 1], rest_weight, dest);
          dest.rotation = NormalizeEst(dest.rotation);
          dest.translation = dest.translation * ratio;
          dest.scale = dest.scale * ratio;
          AddLayers(_job, i + 1, one, &dest);
          _job.output[i + 1] = dest;
        }
        if (i < num_soa_joints) {
          math::SoaTransform dest = _job.output[i];
          OZZ_BLEND_N_PASS(_job.rest_pose[i], rest_weight, dest);
          dest.rotation = NormalizeEst(dest.rotation);
          dest.translation = dest.translation * ratio;
          dest.scale = dest.scale * ratio;
          AddLayers(_job, i, one, &dest);
          _job.output[i] = dest;
        }
      }
    } else {
      size_t i = 0;
      for (; i + 1 < num_soa_joints; i += 2) {
        math::SoaTransform& dest = _job.output[i];
        dest.rotation = NormalizeEst(dest.rotation);
        dest.translation = dest.translation * ratio;
        dest.scale = dest.scale * ratio;
        AddLayers(_job, i, one, &dest);
        math::SoaTransform& dest1 = _job.output[i + 1];
        dest1.rotation = NormalizeEst(dest1.rotation);
        dest1.translation = dest1.translation * ratio;
        dest1.scale = dest1.scale * ratio;
        AddLayers(_job, i + 1, one, &dest1);
      }
      if (i < num_soa_joints) {
        math::SoaTransform& dest = _job.output[i];
        dest.rotation = NormalizeEst(dest.rotation);
        dest.translation = dest.translation * ratio;
        dest.scale = dest.scale * ratio;
        AddLayers(_job, i, one, &dest);
      }
    }
  } else {
    // At least one partial pass requires a per-joint normalized weight.
    const math::SimdFloat4 threshold =
        math::simd_float4::Load1(_job.threshold);
    assert(_stats.num_passes != 0);

    for (size_t i = 0; i < num_soa_joints; ++i) {
      math::SoaTransform dest;
      math::SimdFloat4 accumulated_weight = math::simd_float4::zero();
      bool first_pass = true;
      for (const BlendingJob::Layer& layer : _job.layers) {
        assert(layer.transform.size() >= num_soa_joints);
        if (layer.weight <= 0.f) {
          continue;
        }
        const math::SimdFloat4 layer_weight =
            math::simd_float4::Load1(layer.weight);
        const math::SimdFloat4 weight =
            layer.joint_weights.empty()
                ? layer_weight
                : layer_weight * math::Max0(layer.joint_weights[i]);
        if (first_pass) {
          accumulated_weight = weight;
          OZZ_BLEND_1ST_PASS(layer.transform[i], weight, dest);
          first_pass = false;
        } else {
          accumulated_weight = accumulated_weight + weight;
          OZZ_BLEND_N_PASS(layer.transform[i], weight, dest);
        }
      }

      const math::SimdFloat4 rest_weight =
          math::Max0(threshold - accumulated_weight);
      accumulated_weight = math::Max(threshold, accumulated_weight);
      OZZ_BLEND_N_PASS(_job.rest_pose[i], rest_weight, dest);

      const math::SimdFloat4 ratio = one / accumulated_weight;
      dest.rotation = NormalizeEst(dest.rotation);
      dest.translation = dest.translation * ratio;
      dest.scale = dest.scale * ratio;
      AddLayers(_job, i, one, &dest);
      _job.output[i] = dest;
    }
  }
}

// Applies all additive layers to one already normalized joint.
OZZ_INLINE void AddLayers(const BlendingJob& _job, size_t _i,
                          const math::SimdFloat4& _one,
                          math::SoaTransform* _dest) {
  const math::SimdFloat4 one = _one;
  for (const BlendingJob::Layer& layer : _job.additive_layers) {
    assert(layer.transform.size() >= _job.rest_pose.size());
    if (layer.weight == 0.f) {
      continue;
    }

    const bool positive = layer.weight > 0.f;
    const math::SimdFloat4 layer_weight =
        math::simd_float4::Load1(positive ? layer.weight : -layer.weight);
    if (layer.joint_weights.empty()) {
      const math::SimdFloat4 one_minus_weight = _one - layer_weight;
      if (positive) {
        OZZ_ADD_PASS(layer.transform[_i], layer_weight, (*_dest));
      } else {
        OZZ_SUB_PASS(layer.transform[_i], layer_weight, (*_dest));
      }
    } else {
      const math::SimdFloat4 weight =
          layer_weight * math::Max0(layer.joint_weights[_i]);
      const math::SimdFloat4 one_minus_weight = _one - weight;
      if (positive) {
        OZZ_ADD_PASS(layer.transform[_i], weight, (*_dest));
      } else {
        OZZ_SUB_PASS(layer.transform[_i], weight, (*_dest));
      }
    }
  }
}
}  // namespace

bool BlendingJob::Run() const {
  if (!Validate()) {
    return false;
  }

  // Computes pass counts and the global accumulated weight.
  const PassStats pass_stats = PreparePasses(*this);

  // Blends all layers, rest pose and normalization to the job output.
  BlendLayers(*this, pass_stats);

  return true;
}
}  // namespace animation
}  // namespace ozz
