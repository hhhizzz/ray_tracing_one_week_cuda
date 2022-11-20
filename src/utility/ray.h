#pragma once
#include "vec3.h"

class Ray {
 public:
  __device__ Ray() = default;
  __device__ Ray(const Vec3& a, const Vec3& b) {
    this->a = a;
    this->b = b;
  }
  __device__ Vec3 Origin() const { return a; }
  __device__ Vec3 Direction() const { return b; }
  __device__ Vec3 PointAtParameter(float t) const { return a + t * b; }

  Vec3 a;
  Vec3 b;
};

#pragma endregion