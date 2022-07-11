#include <cstdint>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>

#include <exotic_hashing.hpp>
#include <hashing.hpp>
#include <learned_hashing.hpp>

#include "support/datasets.hpp"

const std::vector<std::int64_t> dataset_sizes{200'000'000};
const std::vector<std::int64_t> overallocations{100, 150, 200};
const std::vector<std::int64_t> datasets{static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::SEQUENTIAL),
                                         static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::GAPPED_10),
                                         static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::UNIFORM),
                                         static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::NORMAL),
                                         static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::BOOKS),
                                         static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::FB),
                                         static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::OSM),
                                         static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::WIKI)};

template<class Fn>
static void BM_items_per_slot(benchmark::State& state) {
   const auto ds_size = state.range(0);
   const auto ds_id = static_cast<dataset::ID>(state.range(1));
   const double overallocation = static_cast<double>(state.range(2)) / 100.0;

   // load dataset
   const auto dataset = dataset::load_cached(ds_id, ds_size);
   if (dataset.empty())
      throw std::runtime_error("benchmark dataset empty");

   std::vector<size_t> counters(static_cast<size_t>(overallocation * static_cast<double>(dataset.size())), 0);

   Fn hashfn(dataset.begin(), dataset.end(), counters.size());

   for (auto _ : state) {
      for (const auto& key : dataset) {
         const auto hash = hashfn(key);
         const auto addr = std::min(counters.size() - 1, hash);
         counters[addr]++;
      }
   }

   size_t empty_buckets = 0;
   size_t colliding_elems = 0;
   size_t winner_elems = 0;
   std::array<size_t, 10> n_buckets{}; // buckets with exactly one, two, three etc elements
   std::fill(n_buckets.begin(), n_buckets.end(), 0);

   for (const auto& cnt : counters) {
      if (cnt == 0)
         empty_buckets++;

      if (cnt == 1)
         winner_elems++;

      if (cnt >= 1 && cnt < 1 + n_buckets.size()) {
         n_buckets[cnt - 1]++;
      }

      if (cnt >= 2)
         colliding_elems += cnt;
   }

   state.counters["empty_buckets"] = static_cast<double>(empty_buckets);
   state.counters["colliding_elems"] = static_cast<double>(colliding_elems);
   state.counters["winner_elems"] = static_cast<double>(winner_elems);

   for (size_t i = 0; i < n_buckets.size(); i++)
      state.counters["n_buckets_" + std::to_string(i)] = static_cast<double>(n_buckets[i]);

   state.counters["overallocation"] = overallocation;
   state.counters["dataset_size"] = static_cast<double>(dataset.size());

   state.SetLabel(Fn::name() + ":" + dataset::name(ds_id));

   state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
   state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                           static_cast<int64_t>(sizeof(typename decltype(dataset)::value_type)));
}

#define SINGLE_ARG(...) __VA_ARGS__

#define BM(Hashfn)                                              \
   BENCHMARK_TEMPLATE(BM_items_per_slot, Hashfn)                \
      ->ArgsProduct({dataset_sizes, datasets, overallocations}) \
      ->Iterations(1);

using Data = std::uint64_t;

template<class H>
struct Learned {
   template<class It>
   Learned(const It& begin, const It& end, const size_t N) : hashfn(begin, end, N) {}

   template<class T>
   size_t operator()(const T& key) {
      return hashfn(key);
   }

   static std::string name() {
      return H::name();
   }

  private:
   const H hashfn;
} __attribute__((aligned(128)));

template<class H>
struct Biased {
   template<class It>
   Biased(const It&, const It&, const size_t N) : hashfn(N) {}

   template<class T>
   size_t operator()(const T& key) {
      return hashfn(key);
   }

   static std::string name() {
      return H::name();
   }

  private:
   const H hashfn;
};

template<class H>
struct Universal {
   template<class It>
   Universal(const It&, const It&, const size_t N) : reductionfn(N) {}

   template<class T>
   size_t operator()(const T& key) {
      const auto hash = hashfn(key);
      const auto offs = reductionfn(hash);
      return offs;
   }

   static std::string name() {
      return H::name();
   }

  private:
   const H hashfn;
   const hashing::reduction::FastModulo<Data> reductionfn;
};

BM(SINGLE_ARG(Learned<learned_hashing::RMIHash<Data, 1'000'000>>))
BM(SINGLE_ARG(Learned<learned_hashing::PGMHash<Data, 4>>));
BM(SINGLE_ARG(Learned<learned_hashing::CHTHash<Data, 16>>));
BM(SINGLE_ARG(Learned<learned_hashing::RadixSplineHash<Data, 18, 4>>))
BM(SINGLE_ARG(Learned<learned_hashing::TrieSplineHash<Data, 4>>));

BM(SINGLE_ARG(Biased<hashing::Fibonacci64>));
BM(SINGLE_ARG(Universal<hashing::MurmurFinalizer<Data>>));

BENCHMARK_MAIN();
