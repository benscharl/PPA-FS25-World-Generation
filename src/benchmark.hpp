#include <chrono>
#include <glm/glm.hpp>
#include <iostream>
#include <numeric>
#include <vector>

#include "cmd_parser.hpp"
#include "heightmap_gen.hpp"

inline constexpr double WARMUP_TIME_THRESHOLD = 3.0;
inline constexpr int BENCHMARK_QUANTITY = 10;

bool test_correctness(const HeightMap& ref, const HeightMap& candidate,
                      float delta = 1e-4) {
  if (ref.data.size() != candidate.data.size()) {
    std::cerr << "Size mismatch: Rows " << ref.data.size() << " vs "
              << candidate.data.size() << std::endl;
    return false;
  }

  const size_t n = ref.data.size();

  for (size_t i = 0; i < n; ++i) {
    if (std::fabs(ref.data[i] - candidate.data[i]) > delta) {
      std::cout << "FAILED" << std::endl;
      return false;
    }
  }
  std::cout << "PASSED" << std::endl;
  return true;
}

void warmup() {
  std::cout << "Perlin generation only:" << std::endl;
  std::cout << "Warming up for 3 seconds..." << std::endl;

  auto threshold = std::chrono::duration<double>(WARMUP_TIME_THRESHOLD);
  auto warmup_start = std::chrono::high_resolution_clock::now();

  while (std::chrono::high_resolution_clock::now() - warmup_start < threshold) {
  }
}

template <typename F>
inline HeightMap benchmark_heightmap(const std::string& label, F&& generate) {
  std::cout << "Starting " << label << " heightmap generation benchmark"
            << std::endl;
  for (int r = 0; r < BENCHMARK_QUANTITY; ++r) {
    generate();
  }
  return generate();
}

inline HeightMap benchmark_heightmap_seq(MapGenerator& mapGenerator,
                                         const PerlinNoise& perlinNoise,
                                         const HeightmapConfig& config) {
  return benchmark_heightmap("sequential", [&] {
    return mapGenerator.generate_heightmap_seq(perlinNoise, config);
  });
}

inline HeightMap benchmark_heightmap_par(MapGenerator& mapGenerator,
                                         const PerlinNoise& perlinNoise,
                                         const HeightmapConfig& config) {
  return benchmark_heightmap("parallel", [&] {
    return mapGenerator.generate_heightmap_par(perlinNoise, config);
  });
}

inline HeightMap benchmark_heightmap_cuda(MapGenerator& mapGenerator,
                                          const PerlinNoiseCuda& perlinNoise,
                                          const HeightmapConfig& config) {
  return benchmark_heightmap("cuda", [&] {
    return mapGenerator.generate_heightmap_cuda(perlinNoise, config);
  });
}

inline HeightMap benchmark_heightmap_hybrid(MapGenerator& mapGenerator,
                                            PerlinNoiseHybrid& perlinNoise,
                                            const HeightmapConfig& config) {
  return benchmark_heightmap("hybrid", [&] {
    return mapGenerator.generate_heightmap_hybrid(perlinNoise, config);
  });
}

void benchmark(CMDSettings& settings) {
  PerlinNoise perlinNoise(settings.seed);
  HeightmapConfig config{settings.dimension, settings.dimension};
  MapGenerator mapGenerator;

  // std::cout << "[Verification] Generating Sequential Reference" << std::endl;
  // HeightMap referenceResult =
  //     mapGenerator.generate_heightmap_par(perlinNoise, config);
  // std::cout << "[Verification] Reference generated" << std::endl;

  switch (settings.mode) {
    case GenerationMode::SEQUENTIAL: {
      auto seq_res_height =
          benchmark_heightmap_seq(mapGenerator, perlinNoise, config);
      break;
    }
    case GenerationMode::PARALLEL: {
      auto par_res_height =
          benchmark_heightmap_par(mapGenerator, perlinNoise, config);
      // test_correctness(referenceResult, par_res_height);
      break;
    }
    case GenerationMode::CUDA: {
      PerlinNoiseCuda perlinNoiseCuda(settings.seed);
      auto cuda_res_height =
          benchmark_heightmap_cuda(mapGenerator, perlinNoiseCuda, config);
      // test_correctness(referenceResult, cuda_res_height);
      break;
    }
    case GenerationMode::HYBRID: {
      PerlinNoiseHybrid perlinNoiseHybrid(
          settings.seed, settings.hybrid_gen_split, settings.hybrid_norm_split);
      auto hybrid_res_height =
          benchmark_heightmap_hybrid(mapGenerator, perlinNoiseHybrid, config);
      // test_correctness(referenceResult, hybrid_res_height);
      break;
    }
    default:
      break;
  }
}
