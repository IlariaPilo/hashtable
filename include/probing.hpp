#pragma once

#include <cstdint>
#include <map>
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
           buckets(directory_address_count(capacity)) {}

      Probing(Probing&&) = default;

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
            assert(false); // TODO: this must never happen in practice
            return false;
         }

         // Using template functor should successfully inline actual hash computation
         const auto orig_slot_index = reductionfn(hashfn(key));
         auto slot_index = orig_slot_index;
         size_t probing_step = 0;

         for (;;) {
            auto& bucket = buckets[slot_index];
            for (size_t i = 0; i < BucketSize; i++) {
               if (bucket.slots[i].key == Sentinel) {
                  bucket.slots[i] = {.key = key, .payload = payload};
                  return true;
               } else if (bucket.slots[i].key == key) {
                  // key already exists
                  return false;
               }
            }

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

      std::map<std::string, std::string> lookup_statistics(const std::vector<Key>& dataset) {
         size_t min_psl = 0, max_psl = 0, total_psl = 0;

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

         return {{"min_psl", std::to_string(min_psl)},
                 {"max_psl", std::to_string(max_psl)},
                 {"total_psl", std::to_string(total_psl)}};
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
      typedef Key KeyType;
      typedef Payload PayloadType;

     private:
      const HashFn hashfn;
      const ReductionFn reductionfn;
      const ProbingFn probingfn;
      const size_t capacity;

     public:
      explicit RobinhoodProbing(const size_t& capacity, const HashFn hashfn = HashFn())
         : hashfn(hashfn), reductionfn(ReductionFn(directory_address_count(capacity))),
           probingfn(ProbingFn(directory_address_count(capacity))), capacity(capacity),
           buckets(directory_address_count(capacity)) {}

      RobinhoodProbing(RobinhoodProbing&&) = default;

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
            auto& bucket = buckets[slot_index];
            for (size_t i = 0; i < BucketSize; i++) {
               if (bucket.slots[i].key == Sentinel) {
                  bucket.slots[i] = {.key = key, .psl = probing_step, .payload = payload};
                  return true;
               } else if (bucket.slots[i].key == key) {
                  // key already exists
                  return false;
               } else if (bucket.slots[i].psl < probing_step) {
                  const auto rich_slot = bucket.slots[i];

                  if (unlikely(orig_key == rich_slot.key))
                     throw std::runtime_error("insertion failed, infinite loop detected");

                  bucket.slots[i] = {.key = key, .psl = probing_step, .payload = payload};

                  key = rich_slot.key;
                  payload = rich_slot.payload;
                  probing_step = rich_slot.psl;

                  // This is important to guarantee lookup success, e.g.,
                  // for quadratic probing.
                  orig_slot_index = reductionfn(hashfn(key));
               }
            }

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

      std::map<std::string, std::string> lookup_statistics(const std::vector<Key>& dataset) {
         size_t min_psl = 0, max_psl = 0, total_psl = 0;

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

         return {{"min_psl", std::to_string(min_psl)},
                 {"max_psl", std::to_string(max_psl)},
                 {"total_psl", std::to_string(total_psl)}};
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
      }

     protected:
      struct Bucket {
         struct Slot {
            Key key = Sentinel;
            size_t psl;
            Payload payload;
         } packed;

         std::array<Slot, BucketSize> slots /*__attribute((aligned(sizeof(Key) * 8)))*/;
      } packed;

      std::vector<Bucket> buckets;
   };
} // namespace hashtable
