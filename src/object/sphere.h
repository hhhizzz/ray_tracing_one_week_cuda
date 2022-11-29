#pragma once

#include "object/hittable.h"
#include "utility/vec3.h"

class Sphere : public HitTable {
 public:
  __device__ Sphere() = default;
  __device__ Sphere(Point3 cen, double r) : center(cen), radius(r){};

  __device__ virtual bool Hit(const Ray& r, double t_min, double t_max,
                              HitRecord* rec) const override;

 public:
  Point3 center{};
  double radius{};
};

__device__ bool Sphere::Hit(const Ray& r, double t_min, double t_max,
                            HitRecord* rec) const {
  Vec3 oc = r.Origin() - center;
  auto a = r.Direction().SquaredLength();
  auto half_b = Dot(oc, r.Direction());
  auto c = oc.SquaredLength() - radius * radius;

  auto discriminant = half_b * half_b - a * c;
  if (discriminant < 0) return false;
  auto sqrt_d = sqrt(discriminant);

  // Find the nearest root that lies in the acceptable range.
  auto root = (-half_b - sqrt_d) / a;
  if (root < t_min || t_max < root) {
    root = (-half_b + sqrt_d) / a;
    if (root < t_min || t_max < root) return false;
  }

  rec->time = root;
  rec->point = r.At(rec->time);
  rec->normal = (rec->point - center) / radius;

  auto outward_normal = (rec->point - center) / radius;
  rec->setFaceNormal(r, outward_normal);

  return true;
}

#pragma endregion