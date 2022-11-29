#pragma once

#include "utility/ray.h"

struct HitRecord {
  Point3 point;
  Vec3 normal;
  double time;
  bool front_face;

  __device__ void setFaceNormal(const Ray& r, const Vec3& outward_normal) {
    front_face = Dot(r.Direction(), outward_normal) < 0;
    normal = front_face ? outward_normal : -outward_normal;
  }
};

class HitTable {
 public:
  __device__ virtual bool Hit(const Ray& r, double t_min, double t_max,
                              HitRecord* rec) const = 0;
};

#pragma endregion