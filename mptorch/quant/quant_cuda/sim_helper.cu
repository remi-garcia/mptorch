#include "quant_kernel.h"
#include <cmath>

__device__ __forceinline__ float round_helper(float a, float r) {
  // return floor(a+r);
  return nearbyint(a + r - 0.5);
}

__device__ __forceinline__ float round(float a, float r, int sigma) {
  a = ldexp(a, -sigma);
  a = round_helper(a, r);
  a = ldexp(a, sigma);
  return a;
}

__device__ __forceinline__ float round(float a, int sigma) {
  a = ldexp(a, -sigma);
  a = nearbyint(a);
  a = ldexp(a, sigma);
  return a;
}

// Round toward the closest value in the integer subset
__device__ __forceinline__ float round(float a, float r, int sigma, int *subset_int, int n) {
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

__device__ __forceinline__ float nearest_round(float a, int sigma) {
  a = ldexp(a, -sigma);
  // a = nearbyint(a);
  a = round(a);
  // a = floor(a+0.5);
  // a = ceil(a-0.5);
  a = ldexp(a, sigma);
  return a;
}

__device__ __forceinline__ float nearest_round(float a, int sigma, int *subset_int, int n) {
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
