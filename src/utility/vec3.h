#pragma once

#include <cmath>
#include <cstdlib>
#include <iostream>

class Vec3 {
 public:
  __host__ __device__ Vec3() = default;
  __host__ __device__ Vec3(double e0, double e1, double e2) {
    e[0] = e0;
    e[1] = e1;
    e[2] = e2;
  }
  __host__ __device__ inline double X() const { return e[0]; }
  __host__ __device__ inline double Y() const { return e[1]; }
  __host__ __device__ inline double Z() const { return e[2]; }
  __host__ __device__ inline double R() const { return e[0]; }
  __host__ __device__ inline double G() const { return e[1]; }
  __host__ __device__ inline double B() const { return e[2]; }

  __host__ __device__ inline const Vec3& operator+() const { return *this; }
  __host__ __device__ inline Vec3 operator-() const {
    return {-e[0], -e[1], -e[2]};
  }
  __host__ __device__ inline double operator[](int i) const { return e[i]; }
  __host__ __device__ inline double& operator[](int i) { return e[i]; };

  __host__ __device__ inline Vec3& operator+=(const Vec3& v2);
  __host__ __device__ inline Vec3& operator-=(const Vec3& v2);
  __host__ __device__ inline Vec3& operator*=(const Vec3& v2);
  __host__ __device__ inline Vec3& operator/=(const Vec3& v2);
  __host__ __device__ inline Vec3& operator*=(double t);
  __host__ __device__ inline Vec3& operator/=(double t);

  __host__ __device__ inline double Length() const {
    return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
  }
  __host__ __device__ inline double SquaredLength() const {
    return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
  }
  __host__ __device__ inline void MakeUnitVector();

  double e[3];
};

inline std::istream& operator>>(std::istream& is, Vec3& t) {
  is >> t.e[0] >> t.e[1] >> t.e[2];
  return is;
}

inline std::ostream& operator<<(std::ostream& os, const Vec3& t) {
  os << t.e[0] << " " << t.e[1] << " " << t.e[2];
  return os;
}

__host__ __device__ inline void Vec3::MakeUnitVector() {
  double k = 1.0 / std::sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
  e[0] *= k;
  e[1] *= k;
  e[2] *= k;
}

__host__ __device__ inline Vec3 operator+(const Vec3& v1, const Vec3& v2) {
  return {v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]};
}

__host__ __device__ inline Vec3 operator-(const Vec3& v1, const Vec3& v2) {
  return {v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]};
}

__host__ __device__ inline Vec3 operator*(const Vec3& v1, const Vec3& v2) {
  return {v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]};
}

__host__ __device__ inline Vec3 operator/(const Vec3& v1, const Vec3& v2) {
  return {v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]};
}

__host__ __device__ inline Vec3 operator*(double t, const Vec3& v) {
  return {t * v.e[0], t * v.e[1], t * v.e[2]};
}

__host__ __device__ inline Vec3 operator/(Vec3 v, double t) {
  return {v.e[0] / t, v.e[1] / t, v.e[2] / t};
}

__host__ __device__ inline Vec3 operator*(const Vec3& v, double t) {
  return {t * v.e[0], t * v.e[1], t * v.e[2]};
}

__host__ __device__ inline double Dot(const Vec3& v1, const Vec3& v2) {
  return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

__host__ __device__ inline Vec3 Cross(const Vec3& v1, const Vec3& v2) {
  return {(v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]),
          (-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
          (v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0])};
}

__host__ __device__ inline Vec3& Vec3::operator+=(const Vec3& v) {
  e[0] += v.e[0];
  e[1] += v.e[1];
  e[2] += v.e[2];
  return *this;
}

__host__ __device__ inline Vec3& Vec3::operator*=(const Vec3& v) {
  e[0] *= v.e[0];
  e[1] *= v.e[1];
  e[2] *= v.e[2];
  return *this;
}

__host__ __device__ inline Vec3& Vec3::operator/=(const Vec3& v) {
  e[0] /= v.e[0];
  e[1] /= v.e[1];
  e[2] /= v.e[2];
  return *this;
}

__host__ __device__ inline Vec3& Vec3::operator-=(const Vec3& v) {
  e[0] -= v.e[0];
  e[1] -= v.e[1];
  e[2] -= v.e[2];
  return *this;
}

__host__ __device__ inline Vec3& Vec3::operator*=(const double t) {
  e[0] *= t;
  e[1] *= t;
  e[2] *= t;
  return *this;
}

__host__ __device__ inline Vec3& Vec3::operator/=(const double t) {
  double k = 1.0 / t;

  e[0] *= k;
  e[1] *= k;
  e[2] *= k;
  return *this;
}

__host__ __device__ inline Vec3 UnitVector(Vec3 v) { return v / v.Length(); }

using Color = Vec3;  // RGB color
using Point3 = Vec3;

#pragma endregion
