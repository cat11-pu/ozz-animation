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

#include <cmath>
#include <iostream>
#include <limits>

#include "gtest/gtest.h"
#include "ozz/animation/offline/raw_track.h"
#include "ozz/animation/offline/track_builder.h"
#include "ozz/animation/runtime/track.h"
#include "ozz/animation/runtime/track_sampling_job.h"
#include "ozz/base/maths/gtest_math_helper.h"
#include "ozz/base/memory/unique_ptr.h"

using ozz::animation::Float2Track;
using ozz::animation::Float3Track;
using ozz::animation::Float4Track;
using ozz::animation::FloatTrack;
using ozz::animation::Float2TrackSamplingJob;
using ozz::animation::Float3TrackSamplingJob;
using ozz::animation::Float4TrackSamplingJob;
using ozz::animation::FloatTrackSamplingJob;
using ozz::animation::QuaternionTrack;
using ozz::animation::QuaternionTrackSamplingJob;
using ozz::animation::offline::RawFloatTrack;
using ozz::animation::offline::RawTrackInterpolation;
using ozz::animation::offline::TrackBuilder;

TEST(JobValidity, TrackSamplingJob) {
  // Instantiates a builder objects with default parameters.
  TrackBuilder builder;

  // Building default RawFloatTrack succeeds.
  RawFloatTrack raw_float_track;
  EXPECT_TRUE(raw_float_track.Validate());

  // Builds track
  ozz::unique_ptr<FloatTrack> track(builder(raw_float_track));
  ASSERT_TRUE(track);

  {  // Empty/default job
    FloatTrackSamplingJob job;
    EXPECT_FALSE(job.Validate());
    EXPECT_FALSE(job.Run());
  }

  {  // Invalid output
    FloatTrackSamplingJob job;
    job.track = track.get();
    EXPECT_FALSE(job.Validate());
    EXPECT_FALSE(job.Run());
  }

  {  // Invalid track.
    FloatTrackSamplingJob job;
    float result;
    job.result = &result;
    EXPECT_FALSE(job.Validate());
    EXPECT_FALSE(job.Run());
  }

  {  // Valid
    FloatTrackSamplingJob job;
    job.track = track.get();
    float result;
    job.result = &result;
    EXPECT_TRUE(job.Validate());
    EXPECT_TRUE(job.Run());
  }
}

TEST(Default, TrackSamplingJob) {
  FloatTrack default_track;
  FloatTrackSamplingJob job;
  job.track = &default_track;
  float result = 1.f;
  job.result = &result;
  EXPECT_TRUE(job.Validate());
  EXPECT_TRUE(job.Run());
  EXPECT_FLOAT_EQ(result, 0.f);
}

TEST(Constant, TrackSamplingJob) {
  TrackBuilder builder;
  float result;

  RawFloatTrack raw_float_track;

  const RawFloatTrack::Keyframe key0 = {RawTrackInterpolation::kLinear, 0.f,
                                        46.f};
  raw_float_track.keyframes.push_back(key0);

  // Builds track
  ozz::unique_ptr<FloatTrack> track(builder(raw_float_track));
  ASSERT_TRUE(track);

  FloatTrackSamplingJob sampling;
  sampling.track = track.get();
  sampling.result = &result;

  sampling.ratio = .5f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT_EQ(result, 46.f);
}

TEST(Bounds, TrackSamplingJob) {
  TrackBuilder builder;
  float result;

  RawFloatTrack raw_float_track;

  const RawFloatTrack::Keyframe key0 = {RawTrackInterpolation::kLinear, 0.f,
                                        0.f};
  raw_float_track.keyframes.push_back(key0);
  const RawFloatTrack::Keyframe key1 = {RawTrackInterpolation::kStep, .5f,
                                        46.f};
  raw_float_track.keyframes.push_back(key1);
  const RawFloatTrack::Keyframe key2 = {RawTrackInterpolation::kLinear, .7f,
                                        0.f};
  raw_float_track.keyframes.push_back(key2);

  // Builds track
  ozz::unique_ptr<FloatTrack> track(builder(raw_float_track));
  ASSERT_TRUE(track);

  // Samples to verify build output.
  FloatTrackSamplingJob sampling;
  sampling.track = track.get();
  sampling.result = &result;

  sampling.ratio = 0.f - 1e-7f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT_EQ(result, 0.f);

  sampling.ratio = 0.f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT_EQ(result, 0.f);

  sampling.ratio = .5f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT_EQ(result, 46.f);

  sampling.ratio = 1.f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT_EQ(result, 0.f);

  sampling.ratio = 1.f + 1e-7f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT_EQ(result, 0.f);

  sampling.ratio = 1.5f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT_EQ(result, 0.f);
}

TEST(Float, TrackSamplingJob) {
  TrackBuilder builder;
  float result;

  ozz::animation::offline::RawFloatTrack raw_track;

  const ozz::animation::offline::RawFloatTrack::Keyframe key0 = {
      RawTrackInterpolation::kLinear, 0.f, 0.f};
  raw_track.keyframes.push_back(key0);
  const ozz::animation::offline::RawFloatTrack::Keyframe key1 = {
      RawTrackInterpolation::kStep, .5f, 4.6f};
  raw_track.keyframes.push_back(key1);
  const ozz::animation::offline::RawFloatTrack::Keyframe key2 = {
      RawTrackInterpolation::kLinear, .7f, 9.2f};
  raw_track.keyframes.push_back(key2);
  const ozz::animation::offline::RawFloatTrack::Keyframe key3 = {
      RawTrackInterpolation::kLinear, .9f, 0.f};
  raw_track.keyframes.push_back(key3);

  // Builds track
  ozz::unique_ptr<FloatTrack> track(builder(raw_track));
  ASSERT_TRUE(track);

  // Samples to verify build output.
  ozz::animation::FloatTrackSamplingJob sampling;
  sampling.track = track.get();
  sampling.result = &result;

  sampling.ratio = 0.f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT_EQ(result, 0.f);

  sampling.ratio = .25f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT_EQ(result, 2.3f);

  sampling.ratio = .5f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT_EQ(result, 4.6f);

  sampling.ratio = .6f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT_EQ(result, 4.6f);

  sampling.ratio = .7f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT_EQ(result, 9.2f);

  sampling.ratio = .8f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT_EQ(result, 4.6f);

  sampling.ratio = .9f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT_EQ(result, 0.f);

  sampling.ratio = 1.f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT_EQ(result, 0.f);
}

TEST(Float2, TrackSamplingJob) {
  TrackBuilder builder;
  ozz::math::Float2 result;

  ozz::animation::offline::RawFloat2Track raw_track;

  const ozz::animation::offline::RawFloat2Track::Keyframe key0 = {
      RawTrackInterpolation::kLinear, 0.f, ozz::math::Float2(0.f, 0.f)};
  raw_track.keyframes.push_back(key0);
  const ozz::animation::offline::RawFloat2Track::Keyframe key1 = {
      RawTrackInterpolation::kStep, .5f, ozz::math::Float2(2.3f, 4.6f)};
  raw_track.keyframes.push_back(key1);
  const ozz::animation::offline::RawFloat2Track::Keyframe key2 = {
      RawTrackInterpolation::kLinear, .7f, ozz::math::Float2(4.6f, 9.2f)};
  raw_track.keyframes.push_back(key2);
  const ozz::animation::offline::RawFloat2Track::Keyframe key3 = {
      RawTrackInterpolation::kLinear, .9f, ozz::math::Float2(0.f, 0.f)};
  raw_track.keyframes.push_back(key3);

  // Builds track
  ozz::unique_ptr<Float2Track> track(builder(raw_track));
  ASSERT_TRUE(track);

  // Samples to verify build output.
  ozz::animation::Float2TrackSamplingJob sampling;
  sampling.track = track.get();
  sampling.result = &result;

  sampling.ratio = 0.f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT2_EQ(result, 0.f, 0.f);

  sampling.ratio = .25f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT2_EQ(result, 1.15f, 2.3f);

  sampling.ratio = .5f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT2_EQ(result, 2.3f, 4.6f);

  sampling.ratio = .6f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT2_EQ(result, 2.3f, 4.6f);

  sampling.ratio = .7f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT2_EQ(result, 4.6f, 9.2f);

  sampling.ratio = .8f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT2_EQ(result, 2.3f, 4.6f);

  sampling.ratio = .9f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT2_EQ(result, 0.f, 0.f);

  sampling.ratio = 1.f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT2_EQ(result, 0.f, 0.f);
}

TEST(Float3, TrackSamplingJob) {
  TrackBuilder builder;
  ozz::math::Float3 result;

  ozz::animation::offline::RawFloat3Track raw_track;

  const ozz::animation::offline::RawFloat3Track::Keyframe key0 = {
      RawTrackInterpolation::kLinear, 0.f, ozz::math::Float3(0.f, 0.f, 0.f)};
  raw_track.keyframes.push_back(key0);
  const ozz::animation::offline::RawFloat3Track::Keyframe key1 = {
      RawTrackInterpolation::kStep, .5f, ozz::math::Float3(0.f, 2.3f, 4.6f)};
  raw_track.keyframes.push_back(key1);
  const ozz::animation::offline::RawFloat3Track::Keyframe key2 = {
      RawTrackInterpolation::kLinear, .7f, ozz::math::Float3(0.f, 4.6f, 9.2f)};
  raw_track.keyframes.push_back(key2);
  const ozz::animation::offline::RawFloat3Track::Keyframe key3 = {
      RawTrackInterpolation::kLinear, .9f, ozz::math::Float3(0.f, 0.f, 0.f)};
  raw_track.keyframes.push_back(key3);

  // Builds track
  ozz::unique_ptr<Float3Track> track(builder(raw_track));
  ASSERT_TRUE(track);

  // Samples to verify build output.
  ozz::animation::Float3TrackSamplingJob sampling;
  sampling.track = track.get();
  sampling.result = &result;

  sampling.ratio = 0.f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT3_EQ(result, 0.f, 0.f, 0.f);

  sampling.ratio = .25f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT3_EQ(result, 0.f, 1.15f, 2.3f);

  sampling.ratio = .5f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT3_EQ(result, 0.f, 2.3f, 4.6f);

  sampling.ratio = .6f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT3_EQ(result, 0.f, 2.3f, 4.6f);

  sampling.ratio = .7f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT3_EQ(result, 0.f, 4.6f, 9.2f);

  sampling.ratio = .8f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT3_EQ(result, 0.f, 2.3f, 4.6f);

  sampling.ratio = .9f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT3_EQ(result, 0.f, 0.f, 0.f);

  sampling.ratio = 1.f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT3_EQ(result, 0.f, 0.f, 0.f);
}

TEST(Float4, TrackSamplingJob) {
  TrackBuilder builder;
  ozz::math::Float4 result;

  ozz::animation::offline::RawFloat4Track raw_track;

  const ozz::animation::offline::RawFloat4Track::Keyframe key0 = {
      RawTrackInterpolation::kLinear, 0.f,
      ozz::math::Float4(0.f, 0.f, 0.f, 0.f)};
  raw_track.keyframes.push_back(key0);
  const ozz::animation::offline::RawFloat4Track::Keyframe key1 = {
      RawTrackInterpolation::kStep, .5f,
      ozz::math::Float4(0.f, 2.3f, 0.f, 4.6f)};
  raw_track.keyframes.push_back(key1);
  const ozz::animation::offline::RawFloat4Track::Keyframe key2 = {
      RawTrackInterpolation::kLinear, .7f,
      ozz::math::Float4(0.f, 4.6f, 0.f, 9.2f)};
  raw_track.keyframes.push_back(key2);
  const ozz::animation::offline::RawFloat4Track::Keyframe key3 = {
      RawTrackInterpolation::kLinear, .9f,
      ozz::math::Float4(0.f, 0.f, 0.f, 0.f)};
  raw_track.keyframes.push_back(key3);

  // Builds track
  ozz::unique_ptr<Float4Track> track(builder(raw_track));
  ASSERT_TRUE(track);

  // Samples to verify build output.
  ozz::animation::Float4TrackSamplingJob sampling;
  sampling.track = track.get();
  sampling.result = &result;

  sampling.ratio = 0.f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT4_EQ(result, 0.f, 0.f, 0.f, 0.f);

  sampling.ratio = .25f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT4_EQ(result, 0.f, 1.15f, 0.f, 2.3f);

  sampling.ratio = .5f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT4_EQ(result, 0.f, 2.3f, 0.f, 4.6f);

  sampling.ratio = .6f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT4_EQ(result, 0.f, 2.3f, 0.f, 4.6f);

  sampling.ratio = .7f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT4_EQ(result, 0.f, 4.6f, 0.f, 9.2f);

  sampling.ratio = .8f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT4_EQ(result, 0.f, 2.3f, 0.f, 4.6f);

  sampling.ratio = .9f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT4_EQ(result, 0.f, 0.f, 0.f, 0.f);

  sampling.ratio = 1.f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_FLOAT4_EQ(result, 0.f, 0.f, 0.f, 0.f);
}

TEST(Quaternion, TrackSamplingJob) {
  TrackBuilder builder;
  ozz::math::Quaternion result;

  ozz::animation::offline::RawQuaternionTrack raw_track;

  const ozz::animation::offline::RawQuaternionTrack::Keyframe key0 = {
      RawTrackInterpolation::kLinear, 0.f,
      ozz::math::Quaternion(.70710677f, 0.f, 0.f, .70710677f)};
  raw_track.keyframes.push_back(key0);
  const ozz::animation::offline::RawQuaternionTrack::Keyframe key1 = {
      RawTrackInterpolation::kStep, .5f,
      ozz::math::Quaternion(0.f, .70710677f, 0.f, .70710677f)};
  raw_track.keyframes.push_back(key1);
  const ozz::animation::offline::RawQuaternionTrack::Keyframe key2 = {
      RawTrackInterpolation::kLinear, .7f,
      ozz::math::Quaternion(.70710677f, 0.f, 0.f, .70710677f)};
  raw_track.keyframes.push_back(key2);
  const ozz::animation::offline::RawQuaternionTrack::Keyframe key3 = {
      RawTrackInterpolation::kLinear, .9f, ozz::math::Quaternion::identity()};
  raw_track.keyframes.push_back(key3);

  // Builds track
  ozz::unique_ptr<QuaternionTrack> track(builder(raw_track));
  ASSERT_TRUE(track);
  // Samples to verify build output.
  ozz::animation::QuaternionTrackSamplingJob sampling;
  sampling.track = track.get();
  sampling.result = &result;

  sampling.ratio = 0.f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_QUATERNION_EQ(result, .70710677f, 0.f, 0.f, .70710677f);

  sampling.ratio = .1f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_QUATERNION_EQ(result, .61721331f, .15430345f, 0.f, .77151674f);

  sampling.ratio = .4999999f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_QUATERNION_EQ(result, 0.f, .70710677f, 0.f, .70710677f);

  sampling.ratio = .5f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_QUATERNION_EQ(result, 0.f, .70710677f, 0.f, .70710677f);

  sampling.ratio = .6f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_QUATERNION_EQ(result, 0.f, .70710677f, 0.f, .70710677f);

  sampling.ratio = .7f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_QUATERNION_EQ(result, .70710677f, 0.f, 0.f, .70710677f);

  sampling.ratio = .8f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_QUATERNION_EQ(result, .38268333f, 0.f, 0.f, .92387962f);

  sampling.ratio = .9f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_QUATERNION_EQ(result, 0.f, 0.f, 0.f, 1.f);

  sampling.ratio = 1.f;
  ASSERT_TRUE(sampling.Run());
  EXPECT_QUATERNION_EQ(result, 0.f, 0.f, 0.f, 1.f);
}

TEST(NonFiniteRatio, TrackSamplingJob) {
  TrackBuilder builder;
  RawFloatTrack raw_track;
  raw_track.keyframes.push_back({RawTrackInterpolation::kLinear, 0.f, 0.f});
  raw_track.keyframes.push_back({RawTrackInterpolation::kLinear, 1.f, 10.f});

  ozz::unique_ptr<FloatTrack> track(builder(raw_track));
  ASSERT_TRUE(track);

  float result = 0.f;
  FloatTrackSamplingJob sampling;
  sampling.track = track.get();
  sampling.result = &result;

  struct {
    float ratio;
    float expected;
  } cases[] = {
      {std::numeric_limits<float>::quiet_NaN(), 0.f},
      {std::numeric_limits<float>::infinity(), 10.f},
      {-std::numeric_limits<float>::infinity(), 0.f},
  };

  for (const auto& test : cases) {
    sampling.ratio = test.ratio;
    result = 93.f;
    EXPECT_TRUE(sampling.Validate());
    EXPECT_TRUE(sampling.Run());
    EXPECT_FLOAT_EQ(result, test.expected);
    EXPECT_TRUE(std::isfinite(result));
  }
}

TEST(EmptyTrack, TrackSamplingJob) {
  FloatTrack track;
  float result = 93.f;
  FloatTrackSamplingJob sampling;
  sampling.track = &track;
  sampling.result = &result;
  sampling.ratio = std::numeric_limits<float>::quiet_NaN();

  EXPECT_TRUE(sampling.Validate());
  EXPECT_TRUE(sampling.Run());
  EXPECT_FLOAT_EQ(result, 0.f);
  EXPECT_TRUE(std::isfinite(result));
}

namespace {

bool IsFinite(float _value) { return std::isfinite(_value); }
bool IsFinite(const ozz::math::Float2& _value) {
  return std::isfinite(_value.x) && std::isfinite(_value.y);
}
bool IsFinite(const ozz::math::Float3& _value) {
  return std::isfinite(_value.x) && std::isfinite(_value.y) &&
         std::isfinite(_value.z);
}
bool IsFinite(const ozz::math::Float4& _value) {
  return std::isfinite(_value.x) && std::isfinite(_value.y) &&
         std::isfinite(_value.z) && std::isfinite(_value.w);
}
bool IsFinite(const ozz::math::Quaternion& _value) {
  return std::isfinite(_value.x) && std::isfinite(_value.y) &&
         std::isfinite(_value.z) && std::isfinite(_value.w);
}

template <typename _RawTrack, typename _Track, typename _Job>
void ScanBoundaryRatios(bool _empty, const typename _Track::ValueType& _value0,
                        const typename _Track::ValueType& _value1,
                        int* _scanned, int* _non_finite, int* _failed) {
  ozz::animation::offline::TrackBuilder builder;
  _RawTrack raw_track;
  if (!_empty) {
    raw_track.keyframes.push_back(
        {RawTrackInterpolation::kLinear, 0.f, _value0});
    raw_track.keyframes.push_back(
        {RawTrackInterpolation::kLinear, 1.f, _value1});
  }

  ozz::unique_ptr<_Track> built_track(builder(raw_track));
  ASSERT_TRUE(built_track);
  _Track default_track;
  const _Track* track = _empty ? &default_track : built_track.get();

  const float ratios[] = {
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      -1.f,
      -.25f,
      0.f,
      .25f,
      1.f,
      1.25f,
  };

  for (float ratio : ratios) {
    typename _Track::ValueType result = _value0;
    _Job job;
    job.track = track;
    job.result = &result;
    job.ratio = ratio;
    if (job.Run()) {
      ++*_scanned;
      *_non_finite += IsFinite(result) ? 0 : 1;
    } else {
      ++*_failed;
    }
  }
}

}  // namespace

TEST(BoundaryScan, TrackSamplingJob) {
  int scanned = 0;
  int non_finite = 0;
  int failed = 0;

  for (bool empty : {false, true}) {
    ScanBoundaryRatios<RawFloatTrack, FloatTrack, FloatTrackSamplingJob>(
        empty, 1.f, 2.f, &scanned, &non_finite, &failed);
    ScanBoundaryRatios<ozz::animation::offline::RawFloat2Track, Float2Track,
                       Float2TrackSamplingJob>(
        empty, ozz::math::Float2(1.f, 2.f), ozz::math::Float2(3.f, 4.f),
        &scanned, &non_finite, &failed);
    ScanBoundaryRatios<ozz::animation::offline::RawFloat3Track, Float3Track,
                       Float3TrackSamplingJob>(
        empty, ozz::math::Float3(1.f, 2.f, 3.f),
        ozz::math::Float3(4.f, 5.f, 6.f), &scanned, &non_finite, &failed);
    ScanBoundaryRatios<ozz::animation::offline::RawFloat4Track, Float4Track,
                       Float4TrackSamplingJob>(
        empty, ozz::math::Float4(1.f, 2.f, 3.f, 4.f),
        ozz::math::Float4(5.f, 6.f, 7.f, 8.f), &scanned, &non_finite, &failed);
    ScanBoundaryRatios<ozz::animation::offline::RawQuaternionTrack,
                       QuaternionTrack, QuaternionTrackSamplingJob>(
        empty, ozz::math::Quaternion::identity(),
        ozz::math::Quaternion(0.f, .70710677f, 0.f, .70710677f), &scanned,
        &non_finite, &failed);
  }

  std::cout << "track boundary scan: scanned=" << scanned
            << " non_finite=" << non_finite << " failed=" << failed
            << std::endl;
  EXPECT_EQ(scanned, 90);
  EXPECT_EQ(non_finite, 0);
  EXPECT_EQ(failed, 0);
}
