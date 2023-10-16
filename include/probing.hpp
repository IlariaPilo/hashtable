#pragma once

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "convenience/builtins.hpp"
#include "thirdparty/libdivide.h"

namespace hashtable {
   struct LinearProbingFunc {
     private:
      const size_t directory_size;

     public:
      LinearProbingFunc(const size_t& directory_size) : directory_size(directory_size) {}

      static std::string name() {
         return "linear";
      }

      forceinline size_t operator()(const size_t& index, const size_t& probing_step) const {
         auto next = index + probing_step;
         // TODO(dominik): benchmark whether this really is the fastest implementation
         while (unlikely(next >= directory_size))
            next -= directory_size;
         return next;
      }
   };

   struct QuadraticProbingFunc {
     private:
      const libdivide::divider<std::uint64_t> magic_div;
      const size_t directory_size;

     public:
      QuadraticProbingFunc(const size_t& directory_size) : magic_div(directory_size), directory_size(directory_size) {}

      static std::string name() {
         return "quadratic";
      }

      forceinline size_t operator()(const size_t& index, const size_t& probing_step) const {
         const auto next_ind = index + probing_step * probing_step;

         const auto div = static_cast<std::uint64_t>(next_ind) / magic_div;
         const auto remainder = next_ind - div * directory_size;
         assert(remainder < directory_size);
         return remainder;
      }
   };

   template<class Key,
            class Payload,
            class HashFn,
            class ReductionFn,
            class ProbingFn,
            size_t MaxProbingSteps = 500,
            size_t BucketSize = 1,
            Key Sentinel = std::numeric_limits<Key>::max()>
   struct Probing {
     public:
      using KeyType = Key;
      using PayloadType = Payload;

     private:
      const HashFn hashfn;
      const ReductionFn reductionfn;
      const ProbingFn probingfn;
      const size_t capacity;

     public:
      explicit Probing(const size_t& capacity, const HashFn hashfn = HashFn())
         : hashfn(hashfn), reductionfn(ReductionFn(directory_address_count(capacity))),
           probingfn(ProbingFn(directory_address_count(capacity))), capacity(capacity),
           buckets(directory_address_count(capacity)), locks(directory_address_count(capacity)) {
         // initialize locks
         for (int i=0; i<directory_address_count(capacity); i++)
            omp_init_lock(&(locks[i]));
      }

      Probing(Probing&&) noexcept = default;

      /**
       * Inserts a key, value/payload pair into the hashtable
       *
       * Note: Will throw a runtime error iff the probing function produces a
       * cycle and all buckets along that cycle are full.
       *
       * @param key
       * @param payload
       * @return whether or not the key, payload pair was inserted. Insertion will fail
       *    iff the same key already exists or if key == Sentinel value
       */
      bool insert(const Key& key, const Payload payload) {
         if (unlikely(key == Sentinel)) {
            assert(false); // TODO(dominik): this must never happen in practice
            return false;
         }

         // Using template functor should successfully inline actual hash computation
         const auto orig_slot_index = reductionfn(hashfn(key));
         auto slot_index = orig_slot_index;
         size_t probing_step = 0;

         for (;;) {
            if (probing_step > MaxProbingSteps)
               throw std::runtime_error("Maximum probing step count (" + std::to_string(MaxProbingSteps) +
                                        ") exceeded");
            omp_set_lock(&(locks[slot_index]));
            auto& bucket = buckets[slot_index];
            for (size_t i = 0; i < BucketSize; i++) {
               if (bucket.slots[i].key == Sentinel) {
                  bucket.slots[i] = {.key = key, .payload = payload};
                  omp_unset_lock(&(locks[slot_index]));
                  return true;
               } else if (bucket.slots[i].key == key) {
                  // key already exists
                  omp_unset_lock(&(locks[slot_index]));
                  return false;
               }
            }
            omp_unset_lock(&(locks[slot_index]));

            // Slot is full, choose a new slot index based on probing function
            slot_index = probingfn(orig_slot_index, ++probing_step);
            if (unlikely(slot_index == orig_slot_index))
               throw std::runtime_error("Building " + this->name() +
                                        " failed: detected cycle during probing, all buckets along the way are full");
         }
      }

      /**
       * Retrieves the associated payload/value for a given key.
       *
       * @param key
       * @return the payload or std::nullopt if key was not found in the Hashtable
       */
      std::optional<Payload> lookup(const Key& key) const {
         if (unlikely(key == Sentinel)) {
            assert(false); // TODO: this must never happen in practice
            return std::nullopt;
         }

         // Using template functor should successfully inline actual hash computation
         const auto orig_slot_index = reductionfn(hashfn(key));
         auto slot_index = orig_slot_index;
         size_t probing_step = 0;

         for (;;) {
            auto& bucket = buckets[slot_index];
            for (size_t i = 0; i < BucketSize; i++) {
               if (bucket.slots[i].key == key)
                  return std::make_optional(bucket.slots[i].payload);

               if (bucket.slots[i].key == Sentinel)
                  return std::nullopt;
            }

            // Slot is full, choose a new slot index based on probing function
            slot_index = probingfn(orig_slot_index, ++probing_step);
            if (unlikely(slot_index == orig_slot_index))
               return std::nullopt;
         }
      }

      std::map<std::string, double> lookup_statistics(const std::vector<Key>& dataset) {
         size_t min_psl = 0, max_psl = 0, total_psl = 0;
         long double average_psl = 0;

         for (const auto& key : dataset) {
            // Using template functor should successfully inline actual hash computation
            const auto orig_slot_index = reductionfn(hashfn(key));
            auto slot_index = orig_slot_index;
            size_t probing_step = 0;

            for (;;) {
               auto& bucket = buckets[slot_index];
               for (size_t i = 0; i < BucketSize; i++) {
                  if (bucket.slots[i].key == key) {
                     average_psl += static_cast<long double>(probing_step);
                     min_psl = std::min(min_psl, probing_step);
                     max_psl = std::max(max_psl, probing_step);
                     total_psl += probing_step;
                     goto next;
                  }

                  if (bucket.slots[i].key == Sentinel)
                     goto next;
               }

               // Slot is full, choose a new slot index based on probing function
               slot_index = probingfn(orig_slot_index, ++probing_step);
               if (unlikely(slot_index == orig_slot_index))
                  goto next;
            }

         next:
            continue;
         }

         average_psl /= static_cast<long double>(dataset.size());

         return {{"min_psl", min_psl}, {"max_psl", max_psl}, {"average_psl", average_psl}, {"total_psl", total_psl}};
      }

      size_t byte_size() const {
         return sizeof(*this) + buckets.size() * bucket_byte_size();
      }

      static constexpr forceinline size_t bucket_byte_size() {
         return sizeof(Bucket);
      }

      static forceinline std::string name() {
         return ProbingFn::name() + "_probing";
      }

      static forceinline std::string hash_name() {
         return HashFn::name();
      }

      static forceinline std::string reducer_name() {
         return ReductionFn::name();
      }

      static constexpr forceinline size_t bucket_size() {
         return BucketSize;
      }

      static constexpr forceinline size_t directory_address_count(const size_t& capacity) {
         return (capacity + BucketSize - 1) / BucketSize;
      }

      /**
       * Clears all keys from the hashtable. Note that payloads are technically
       * still in memory (i.e., might leak if sensitive).
       */
      void clear() {
         for (auto& bucket : buckets) {
            for (auto& slot : bucket.slots) {
               slot.key = Sentinel;
            }
         }
      }

      ~Probing() {
         clear();
         // destroy locks
         for (size_t i=0; i<locks.size(); i++)
            omp_destroy_lock(&(locks[i]));
      }

     protected:
      struct Bucket {
         struct Slot {
            Key key = Sentinel;
            Payload payload;
         } packed;

         std::array<Slot, BucketSize> slots /*__attribute((aligned(sizeof(Key) * 8)))*/;
      } packed;

      std::vector<Bucket> buckets;
      std::vector<omp_lock_t> locks;
   };

   template<class Key,
            class Payload,
            class HashFn,
            class ReductionFn,
            class ProbingFn,
            size_t BucketSize = 1,
            Key Sentinel = std::numeric_limits<Key>::max()>
   struct RobinhoodProbing {
     public:
      using KeyType = Key;
      using PayloadType = Payload;

     private:
      const HashFn hashfn;
      const ReductionFn reductionfn;
      const ProbingFn probingfn;
      const size_t capacity;

     public:
      explicit RobinhoodProbing(const size_t& capacity, const HashFn hashfn = HashFn())
         : hashfn(hashfn), reductionfn(ReductionFn(directory_address_count(capacity))),
           probingfn(ProbingFn(directory_address_count(capacity))), capacity(capacity),
           buckets(directory_address_count(capacity)), locks(directory_address_count(capacity)) {
         // initialize locks
         for (size_t i=0; i<directory_address_count(capacity); i++)
            omp_init_lock(&(locks[i]));
      }

      RobinhoodProbing(RobinhoodProbing&&) noexcept = default;

      /**
       * Inserts a key, value/payload pair into the hashtable
       *
       * Note: Will throw a runtime error iff the probing function produces a
       * cycle and all buckets along that cycle are full.
       *
       * @param key
       * @param payload
       * @return whether or not the key, payload pair was inserted. Insertion will fail
       *    iff the same key already exists or if key == Sentinel value
       */
      bool insert(const Key& k, const Payload& p) {
         // r+w variables (required to avoid insert recursion+issues)
         auto key = k;
         auto payload = p;

         const auto orig_key = key;

         if (unlikely(key == Sentinel)) {
            assert(false); // TODO: this must never happen in practice
            return false;
         }

         // Using template functor should successfully inline actual hash computation
         auto orig_slot_index = reductionfn(hashfn(key));
         auto slot_index = orig_slot_index;
         size_t probing_step = 0;

         for (;;) {
            omp_set_lock(&(locks[slot_index]));
            auto& bucket = buckets[slot_index];
            for (size_t i = 0; i < BucketSize; i++) {
               if (bucket.slots[i].key == Sentinel) {
                  bucket.slots[i] = {.key = key, .payload = payload, .psl = probing_step};
                  omp_unset_lock(&(locks[slot_index]));
                  return true;
               } else if (bucket.slots[i].key == key) {
                  // key already exists
                  omp_unset_lock(&(locks[slot_index]));
                  return false;
               } else if (bucket.slots[i].psl < probing_step) {
                  const auto rich_slot = bucket.slots[i];

                  if (unlikely(orig_key == rich_slot.key))
                     throw std::runtime_error("insertion failed, infinite loop detected");

                  bucket.slots[i] = {.key = key, .payload = payload, .psl = probing_step};

                  key = rich_slot.key;
                  payload = rich_slot.payload;
                  probing_step = rich_slot.psl;

                  // This is important to guarantee lookup success, e.g.,
                  // for quadratic probing.
                  orig_slot_index = reductionfn(hashfn(key));
               }
            }
            omp_unset_lock(&(locks[slot_index]));
            // Slot is full, choose a new slot index based on probing function
            slot_index = probingfn(orig_slot_index, ++probing_step);
            if (unlikely(slot_index == orig_slot_index))
               throw std::runtime_error("Building " + this->name() +
                                        " failed: detected cycle during probing, all buckets along the way are full");
         }
      }

      /**
       * Retrieves the associated payload/value for a given key.
       *
       * @param key
       * @return the payload or std::nullopt if key was not found in the Hashtable
       */
      std::optional<Payload> lookup(const Key& key) const {
         if (unlikely(key == Sentinel)) {
            assert(false); // TODO: this must never happen in practice
            return std::nullopt;
         }

         // Using template functor should successfully inline actual hash computation
         const auto orig_slot_index = reductionfn(hashfn(key));
         auto slot_index = orig_slot_index;
         size_t probing_step = 0;

         for (;;) {
            auto& bucket = buckets[slot_index];
            for (size_t i = 0; i < BucketSize; i++) {
               if (bucket.slots[i].key == key)
                  return std::make_optional(bucket.slots[i].payload);

               if (bucket.slots[i].key == Sentinel)
                  return std::nullopt;
            }

            // Slot is full, choose a new slot index based on probing function
            slot_index = probingfn(orig_slot_index, ++probing_step);
            if (unlikely(slot_index == orig_slot_index))
               return std::nullopt;
         }
      }

      std::map<std::string, double> lookup_statistics(const std::vector<Key>& dataset) {
         size_t min_psl = 0, max_psl = 0, total_psl = 0;
         long double average_psl = 0;

         for (const auto& key : dataset) {
            // Using template functor should successfully inline actual hash computation
            const auto orig_slot_index = reductionfn(hashfn(key));
            auto slot_index = orig_slot_index;
            size_t probing_step = 0;

            for (;;) {
               auto& bucket = buckets[slot_index];
               for (size_t i = 0; i < BucketSize; i++) {
                  if (bucket.slots[i].key == key) {
                     min_psl = std::min(min_psl, probing_step);
                     max_psl = std::max(max_psl, probing_step);
                     total_psl += probing_step;
                     average_psl += static_cast<long double>(probing_step);
                     goto next;
                  }

                  if (bucket.slots[i].key == Sentinel)
                     goto next;
               }

               // Slot is full, choose a new slot index based on probing function
               slot_index = probingfn(orig_slot_index, ++probing_step);
               if (unlikely(slot_index == orig_slot_index))
                  goto next;
            }

         next:
            continue;
         }

         average_psl /= static_cast<long double>(dataset.size());

         return {{"min_psl", min_psl}, {"max_psl", max_psl}, {"average_psl", average_psl}, {"total_psl", total_psl}};
      }

      size_t byte_size() const {
         return sizeof(*this) + buckets.size() * bucket_byte_size();
      }

      static constexpr forceinline size_t bucket_byte_size() {
         return sizeof(Bucket);
      }

      static forceinline std::string name() {
         return ProbingFn::name() + "_robinhood_probing";
      }

      static forceinline std::string hash_name() {
         return HashFn::name();
      }

      static forceinline std::string reducer_name() {
         return ReductionFn::name();
      }

      static constexpr forceinline size_t bucket_size() {
         return BucketSize;
      }

      static constexpr forceinline size_t directory_address_count(const size_t& capacity) {
         return (capacity + BucketSize - 1) / BucketSize;
      }

      /**
       * Clears all keys from the hashtable. Note that payloads are technically
       * still in memory (i.e., might leak if sensitive).
       */
      void clear() {
         for (auto& bucket : buckets) {
            for (auto& slot : bucket.slots) {
               slot.key = Sentinel;
            }
         }
      }

      ~RobinhoodProbing() {
         clear();
         // destroy locks
         for (size_t i=0; i<locks.size(); i++)
            omp_destroy_lock(&(locks[i]));
      }

     protected:
      struct Bucket {
         struct Slot {
            Key key = Sentinel;
            Payload payload;
            size_t psl = 0;
         } packed;

         std::array<Slot, BucketSize> slots /*__attribute((aligned(sizeof(Key) * 8)))*/;
      } packed;

      std::vector<Bucket> buckets;
      std::vector<omp_lock_t> locks;
   };
} // namespace hashtable
