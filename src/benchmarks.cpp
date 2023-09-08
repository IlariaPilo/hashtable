#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>

#include <exotic_hashing.hpp>
#include <hashing.hpp>
#include <hashtable.hpp>
#include <learned_hashing.hpp>

#include "support/convenience.hpp"
#include "support/datasets.hpp"
#include "support/probing_set.hpp"

using Key = std::uint64_t;
using Payload = std::uint64_t;

const std::vector<std::int64_t> dataset_sizes{200'000'000};
const std::vector<std::int64_t> overallocations{100, 150, 200};
const std::vector<std::int64_t> cuckoo_overallocations{105, 110, 125};
const std::vector<std::int64_t> datasets{static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::SEQUENTIAL),
                                         static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::GAPPED_10),
                                         static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::UNIFORM),
                                         static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::NORMAL),
                                         static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::BOOKS),
                                         static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::FB),
                                         static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::OSM),
                                         static_cast<std::underlying_type_t<dataset::ID>>(dataset::ID::WIKI)};
const std::vector<std::int64_t> probe_distributions{
   static_cast<std::underlying_type_t<dataset::ProbingDistribution>>(dataset::ProbingDistribution::UNIFORM),
   static_cast<std::underlying_type_t<dataset::ProbingDistribution>>(dataset::ProbingDistribution::EXPONENTIAL)};

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

template<class Hashtable, class HashFn, bool Presorted = true>
static void BM_hashtable(benchmark::State& state) {
   const auto ds_size = state.range(0);
   const auto ds_id = static_cast<dataset::ID>(state.range(1));
   const double overallocation = static_cast<double>(state.range(2)) / 100.0;

   // load dataset
   auto dataset = dataset::load_cached(ds_id, ds_size);
   if (dataset.empty())
      throw std::runtime_error("benchmark dataset empty");

   // generate random payloads
   std::vector<Payload> payloads;
   payloads.reserve(dataset.size());
   std::random_device rd;
   std::default_random_engine rng(rd());
   std::uniform_int_distribution<Payload> dist(std::numeric_limits<Payload>::min(),
                                               std::numeric_limits<Payload>::max());
   for (size_t i = 0; i < dataset.size(); i++)
      payloads.push_back(dist(rng));
   if (payloads.size() != dataset.size())
      throw std::runtime_error("O(hno)");

   const auto address_space = overallocation * static_cast<double>(dataset.size());
   const auto capacity = Hashtable::directory_address_count(address_space);

   // make sure we actually copy & sort during measurment
   if constexpr (!Presorted)
      std::random_shuffle(dataset.begin(), dataset.end());

   const auto sample_start_time = std::chrono::steady_clock::now();
   std::vector<typename decltype(dataset)::value_type> sorted_ds(dataset.begin(), dataset.end());
   std::sort(sorted_ds.begin(), sorted_ds.end());
   const auto sample_end_time = std::chrono::steady_clock::now();

   // create hashtable and insert all keys
   const auto ht_build_start = std::chrono::steady_clock::now();
   Hashtable table(address_space, HashFn(sorted_ds.begin(), sorted_ds.end(), capacity));
   bool failed = false;
   size_t failed_at = 0;
   try {
      for (size_t i = 0; i < sorted_ds.size(); i++) {
         const auto& key = sorted_ds[i];
         const auto& payload = payloads[i];
         table.insert(key, payload);
         failed_at++;
      }
   } catch (const std::runtime_error& e) { failed = true; }
   const auto ht_build_end = std::chrono::steady_clock::now();

   // probe in random order to limit caching effects
   const auto probing_dist = static_cast<dataset::ProbingDistribution>(state.range(3));
   const auto probing_set = dataset::generate_probing_set(dataset, probing_dist);

   size_t i = 0;
   for (auto _ : state) {
      if (failed)
         continue;

      while (unlikely(i >= probing_set.size()))
         i -= probing_set.size();
      const auto& key = probing_set[i++];

      const auto payload_opt = table.lookup(key);
      const auto payload = payload_opt.value();
      benchmark::DoNotOptimize(payload);

      __sync_synchronize();

      i++;
   }

   state.counters["sample_time"] = std::chrono::duration<double>(sample_end_time - sample_start_time).count();
   state.counters["build_time"] = std::chrono::duration<double>(ht_build_end - ht_build_start).count();
   state.counters["failed"] = failed ? 1.0 : 0.0;
   state.counters["failed_at"] = static_cast<double>(failed_at);

   state.counters["overallocation"] = overallocation;
   state.counters["table_capacity"] = capacity;
   state.counters["dataset_size"] = static_cast<double>(dataset.size());
   state.counters["hashtable_bytes"] = table.byte_size();

   if (!failed) {
      for (const auto& stats : table.lookup_statistics(dataset))
         state.counters[stats.first] = stats.second;
   }

   state.SetLabel(Hashtable::name() + ":" + dataset::name(ds_id) + ":" + dataset::name(probing_dist) + ":" +
                  std::to_string(Presorted));
   state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

#define SINGLE_ARG(...) __VA_ARGS__

#define BM_Cuckoo(Hashfn, Kickingfn)                                                        \
   BENCHMARK_TEMPLATE(BM_hashtable,                                                         \
                      hashtable::Cuckoo<Key,                                                \
                                        Payload,                                            \
                                        4,                                                  \
                                        Hashfn,                                             \
                                        hashing::XXHash3<Key>,                              \
                                        hashing::reduction::DoNothing<Key>,                 \
                                        hashing::reduction::FastModulo<Key>,                \
                                        Kickingfn>,                                         \
                      Hashfn,                                                               \
                      false)                                                                \
      ->ArgsProduct({dataset_sizes, datasets, cuckoo_overallocations, probe_distributions}) \
      ->Iterations(10'000'000);

#define BM_Probing(Hashfn, Probingfn)                                                                          \
   BENCHMARK_TEMPLATE(BM_hashtable,                                                                            \
                      hashtable::Probing<Key, Payload, Hashfn, hashing::reduction::DoNothing<Key>, Probingfn>, \
                      Hashfn,                                                                                  \
                      false)                                                                                   \
      ->ArgsProduct({dataset_sizes, datasets, overallocations, probe_distributions})                           \
      ->Iterations(10'000'000);                                                                                \
   BENCHMARK_TEMPLATE(                                                                                         \
      BM_hashtable,                                                                                            \
      hashtable::RobinhoodProbing<Key, Payload, Hashfn, hashing::reduction::DoNothing<Key>, Probingfn>,        \
      Hashfn,                                                                                                  \
      false)                                                                                                   \
      ->ArgsProduct({dataset_sizes, datasets, overallocations, probe_distributions})                           \
      ->Iterations(10'000'000);

#define BM_Build(Hashfn, Sorted)                                                                       \
   BENCHMARK_TEMPLATE(BM_build,                                                                        \
                      hashtable::Chained<Key, Payload, 2, Hashfn, hashing::reduction::DoNothing<Key>>, \
                      Hashfn,                                                                          \
                      Sorted)                                                                          \
      ->ArgsProduct({dataset_sizes, datasets, overallocations})                                        \
      ->Iterations(1);

#define BM(Hashfn)                                                                                     \
   BENCHMARK_TEMPLATE(BM_hashtable,                                                                    \
                      hashtable::Chained<Key, Payload, 2, Hashfn, hashing::reduction::DoNothing<Key>>, \
                      Hashfn,                                                                          \
                      false)                                                                           \
      ->ArgsProduct({dataset_sizes, datasets, overallocations, probe_distributions})                   \
      ->Iterations(10'000'000);                                                                        \
   BM_Cuckoo(SINGLE_ARG(Hashfn), SINGLE_ARG(hashtable::BalancedKicking));                              \
   BM_Cuckoo(SINGLE_ARG(Hashfn), SINGLE_ARG(hashtable::BiasedKicking<20>));                            \
   BM_Cuckoo(SINGLE_ARG(Hashfn), SINGLE_ARG(hashtable::BiasedKicking<80>));                            \
   BM_Cuckoo(SINGLE_ARG(Hashfn), SINGLE_ARG(hashtable::UnbiasedKicking));                              \
   BM_Probing(SINGLE_ARG(Hashfn), SINGLE_ARG(hashtable::LinearProbingFunc));                           \
   BM_Probing(SINGLE_ARG(Hashfn), SINGLE_ARG(hashtable::QuadraticProbingFunc));                        \
//   BENCHMARK_TEMPLATE(BM_items_per_slot, Hashfn)                                                       \
//      ->ArgsProduct({dataset_sizes, datasets, overallocations})                                        \
//      ->Iterations(1);                                                                                 \

template<class H>
struct Learned {
   template<class It>
   Learned(const It& begin, const It& end, const size_t N) : hashfn(begin, end, N) {}

   template<class T>
   size_t operator()(const T& key) const {
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
   size_t operator()(const T& key) const {
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
   size_t operator()(const T& key) const {
      const auto hash = hashfn(key);
      const auto offs = reductionfn(hash);
      return offs;
   }

   static std::string name() {
      return H::name();
   }

  private:
   const H hashfn;
   const hashing::reduction::FastModulo<Key> reductionfn;
};

BM(SINGLE_ARG(Learned<learned_hashing::RMIHash<Key, 1'000'000>>))
BM(SINGLE_ARG(Learned<learned_hashing::TrieSplineHash<Key, 4>>));
BM(SINGLE_ARG(Universal<hashing::MurmurFinalizer<Key>>));
BM(SINGLE_ARG(Biased<hashing::Fibonacci64>));
BM(SINGLE_ARG(Learned<learned_hashing::RadixSplineHash<Key, 18, 4>>))
// BM(SINGLE_ARG(Learned<learned_hashing::PGMHash<Key, 4>>));
BM(SINGLE_ARG(Learned<learned_hashing::CHTHash<Key, 16>>));

BENCHMARK_MAIN();
