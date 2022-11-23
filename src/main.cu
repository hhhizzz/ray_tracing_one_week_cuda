#include <cmath>
#include <fstream>
#include <iostream>

#include "utility/ray.h"
#include "utility/vec3.h"

// limited version of CheckCudaErrors from helper_cuda.h in CUDA examples
#define CHECK_CUDA_ERRORS(val) CheckCuda((val), #val, __FILE__, __LINE__)

void CheckCuda(cudaError_t result, char const* const func,
               const char* const file, int const line) {
  if (result) {
    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at "
              << file << ":" << line << " '" << func << "' \n";
    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    exit(99);
  }
}

__device__ double HitSphere(const Vec3& center, float radius, const Ray& r) {
  Vec3 oc = r.Origin() - center;
  float a = Dot(r.Direction(), r.Direction());
  float b = 2.0f * Dot(oc, r.Direction());
  float c = Dot(oc, oc) - radius * radius;
  float discriminant = b * b - 4 * a * c;
  if (discriminant < 0) {
    return -1.0;
  } else {
    return (-b - std::sqrt(discriminant)) / (2.0 * a);
  }
}

__device__ Color RayColor(const Ray& r) {
  auto t = HitSphere({0, 0, -1}, 0.5, r);
  if (t > 0.0) {
    Vec3 N = UnitVector(r.At(t) - Vec3(0, 0, -1));
    return 0.5 * Color(N.X() + 1, N.Y() + 1, N.Z() + 1);
  }
  Vec3 unit_direction = UnitVector(r.Direction());
  t = 0.5 * (unit_direction.Y() + 1.0);
  return (1.0f - t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0);
}

__global__ void Render(Vec3* fb, int max_x, int max_y, Vec3 lower_left_corner,
                       Vec3 horizontal, Vec3 vertical, Vec3 origin) {
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ((i >= max_x) || (j >= max_y)) {
    return;
  }

  float u = float(i) / float(max_x);
  float v = float(j) / float(max_y);
  Ray r(origin, lower_left_corner + u * horizontal + v * vertical);

  unsigned int pixel_index = j * max_x + i;
  fb[pixel_index] = RayColor(r);
}

int main() {
  int nx = 1200;
  int ny = 600;
  int tx = 8;
  int ty = 8;

  std::cerr << "Rendering a " << nx << "x" << ny << " image ";
  std::cerr << "in " << tx << "x" << ty << " blocks.\n";

  int num_pixels = nx * ny;
  size_t fb_size = num_pixels * sizeof(Vec3);

  // allocate FB
  Vec3* fb;
  CHECK_CUDA_ERRORS(cudaMallocManaged((void**)&fb, fb_size));

  clock_t start, stop;
  start = clock();
  // Render our buffer
  dim3 blocks(nx / tx + 1, ny / ty + 1);
  dim3 threads(tx, ty);
  Render<<<blocks, threads>>>(fb, nx, ny, Vec3(-2.0, -1.0, -1.0),
                              Vec3(4.0, 0.0, 0.0), Vec3(0.0, 2.0, 0.0),
                              Vec3(0.0, 0.0, 0.0));
  CHECK_CUDA_ERRORS(cudaGetLastError());
  CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

  stop = clock();
  double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
  std::cerr << "took " << timer_seconds << " seconds.\n";

  std::string scene_name = "Sphere";
  // Output
  std::ofstream ofs(scene_name + ".ppm");
  // Output FB as Image
  ofs << "P3\n" << nx << " " << ny << "\n255\n";
  for (int j = ny - 1; j >= 0; j--) {
    for (int i = 0; i < nx; i++) {
      size_t pixel_index = j * nx + i;
      int ir = int(255.99 * fb[pixel_index].R());
      int ig = int(255.99 * fb[pixel_index].G());
      int ib = int(255.99 * fb[pixel_index].B());
      ofs << ir << " " << ig << " " << ib << "\n";
    }
  }

  CHECK_CUDA_ERRORS(cudaFree(fb));
}
