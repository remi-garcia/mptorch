#include "quant.h"
#include <cmath>
#include <cstdint>
#include <vector>
#include <ATen/ATen.h>

void fixed_min_max(int wl, int fl, bool symmetric, float *t_min, float *t_max)
{
  int sigma = -fl;
  *t_min = -ldexp(1.0, wl - fl - 1);
  *t_max = -*t_min - ldexp(1.0, sigma);
  if (symmetric)
    *t_min = *t_min + ldexp(1.0, sigma);
}

float round_helper(float a, float r) {
  // return floor(a+r);
  return nearbyint(a + r - 0.5);
}

float round(float a, float r, int sigma) {
  a = ldexp(a, -sigma);
  a = round_helper(a, r);
  a = ldexp(a, sigma);
  return a;
}

float round(float a, int sigma) {
  a = ldexp(a, -sigma);
  a = round(a);
  a = ldexp(a, sigma);
  return a;
}

// Round toward the closest value in the integer subset
float round(float a, float r, int sigma, int *subset_int, int n) {
    if (subset_int == nullptr || n <= 0) {
        return round(a, r, sigma);
    }
    float x = ldexp(a, -sigma);

    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (static_cast<float>(subset_int[mid]) < x) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }

    float chosen;
    if (lo == 0) {
        chosen = static_cast<float>(subset_int[0]);
    } else if (lo >= n) {
        chosen = static_cast<float>(subset_int[n - 1]);
    } else {
        int lower = subset_int[lo - 1];
        int upper = subset_int[lo];
        int diff  = upper - lower;

        // normalize to [0,1]
        float frac = (x - static_cast<float>(lower)) / static_cast<float>(diff);

        // round_helper(frac, r) returns 0 or 1
        float rounded = static_cast<float>(round_helper(frac, r));

        // scale back
        chosen = lower + static_cast<int>(rounded) * diff;
    }

    return ldexp(chosen, sigma);
}

float nearest_round(float a, int sigma, int *subset_int, int n) {
  if (subset_int == nullptr || n <= 0) {
        return round(a, sigma);
    }
    float x = ldexp(a, -sigma);

    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (static_cast<float>(subset_int[mid]) < x) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }

    float chosen;
    if (lo == 0) {
        chosen = static_cast<float>(subset_int[0]);
    } else if (lo >= n) {
        chosen = static_cast<float>(subset_int[n - 1]);
    } else {
        int lower = subset_int[lo - 1];
        int upper = subset_int[lo];
        int diff  = upper - lower;

        // normalize to [0,1]
        float frac = (x - static_cast<float>(lower)) / static_cast<float>(diff);

        // round returns 0 or 1
        float rounded = static_cast<float>(round(frac));

        // scale back
        chosen = lower + static_cast<int>(rounded) * diff;
    }

    return ldexp(chosen, sigma);
}

DimSizes partition_tensor(Tensor input, std::vector<int> &dims){
  DimSizes sizes;
  std::vector<int> real_dims(dims.size());
  for (int i = 0; i < dims.size(); i++){
    real_dims[i] = (input.dim() + (dims[i] % input.dim())) % input.dim();
  }

  sizes.channel = 1;
  for (int dim : real_dims){
    sizes.channel *= input.size(dim);
  }

  int min_dim = real_dims.back();
  int max_dim = real_dims.front();

  sizes.outer = 1;
  for (int i = 0; i < min_dim; i++){
    sizes.outer *= input.size(i);
  }

  sizes.inner = 1;
  for (int i = max_dim + 1; i < input.dim(); i++){
    sizes.inner *= input.size(i);
  }
  return sizes;
}

DimSizes partition_tensor(Tensor a, int dim) {
  DimSizes sizes;
  int real_dim = (a.dim() + (dim % a.dim())) % a.dim();
  sizes.outer = 1;
  sizes.channel = a.size(real_dim);
  sizes.inner = 1;
  for (int i = 0; i < real_dim; ++i) {
    sizes.outer *= a.size(i);
  }
  for (int i = real_dim + 1; i < a.dim(); ++i) {
    sizes.inner *= a.size(i);
  }
  return sizes;
}