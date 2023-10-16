#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <omp.h>

// Order important
#include "convenience/builtins.hpp"

namespace hashtable {
   template<class Key, class Payload, size_t BucketSize, class HashFn, class ReductionFn,
            Key Sentinel = std::numeric_limits<Key>::max()>
   struct Chained {
     public:
      using KeyType = Key;
      using PayloadType = Payload;

     private:
      const HashFn hashfn;
      const ReductionFn reductionfn;
      const size_t capacity;

     public:
      explicit Chained(const size_t& capacity, const HashFn hashfn = HashFn())
         : hashfn(hashfn), reductionfn(ReductionFn(directory_address_count(capacity))), capacity(capacity),
           slots(directory_address_count(capacity)), locks(directory_address_count(capacity)) {
         // initialize locks
         for (size_t i=0; i<directory_address_count(capacity); i++)
            omp_init_lock(&(locks[i]));
      }

      Chained(Chained&&) noexcept = default;

      /**
       * Inserts a key, value/payload pair into the hashtable
       *
       * @param key
       * @param payload
       * @return whether or not the key, payload pair was inserted. Insertion will fail
       *    iff the same key already exists or if key == Sentinel value
       */
      bool insert(const Key& key, const Payload& payload) {
         if (unlikely(key == Sentinel)) {
            assert(false); // TODO(unknown): this must never happen in practice
            return false;
         }

         // Using template functor should successfully inline actual hash computation
         const auto idx = reductionfn(hashfn(key));

         // ***** Begin critical section ***** //
         omp_set_lock(&(locks[idx]));
         FirstLevelSlot& slot = slots[idx];

         // Store directly in slot if possible
         if (slot.key == Sentinel) {
            slot.key = key;
            slot.payload = payload;
            omp_unset_lock(&(locks[idx]));
            return true;
         }

         // Initialize bucket chain if empty
         Bucket* bucket = slot.buckets;
         if (bucket == nullptr) {
            auto b = new Bucket();
            b->slots[0] = {.key = key, .payload = payload};
            slot.buckets = b;
            omp_unset_lock(&(locks[idx]));
            return true;
         }

         // Go through existing buckets and try to insert there if possible
         for (;;) {
            // Find suitable empty entry place. Note that deletions with holes will require
            // searching entire bucket to deal with duplicate keys!
            for (size_t i = 0; i < BucketSize; i++) {
               if (bucket->slots[i].key == Sentinel) {
                  bucket->slots[i] = {.key = key, .payload = payload};
                  omp_unset_lock(&(locks[idx]));
                  return true;
               } else if (bucket->slots[i].key == key) {
                  // key already exists
                  omp_unset_lock(&(locks[idx]));
                  return false;
               }
            }

            if (bucket->next == nullptr)
               break;
            bucket = bucket->next;
         }

         // Append a new bucket to the chain and add element there
         auto b = new Bucket();
         b->slots[0] = {.key = key, .payload = payload};
         bucket->next = b;

         omp_unset_lock(&(locks[idx]));
         return true;
      }

      /**
       * Retrieves the associated payload/value for a given key.
       *
       * @param key
       * @return the payload or std::nullopt if key was not found in the Hashtable
       */
      std::optional<Payload> lookup(const Key& key) const {
         if (unlikely(key == Sentinel)) {
            assert(false); // TODO(unknown): this must never happen in practice
            return std::nullopt;
         }

         // Using template functor should successfully inline actual hash computation
         const FirstLevelSlot& slot = slots[reductionfn(hashfn(key))];

         if (slot.key == key) {
            return std::make_optional(slot.payload);
         }

         Bucket* bucket = slot.buckets;
         while (bucket != nullptr) {
            for (size_t i = 0; i < BucketSize; i++) {
               if (bucket->slots[i].key == key) {
                  Payload payload = bucket->slots[i].payload;
                  return std::make_optional(payload);
               }

               if (bucket->slots[i].key == Sentinel)
                  return std::nullopt;
            }
            bucket = bucket->next;
         }

         return std::nullopt;
      }

      /**
       * Retrieves the payloads associated with keys within [min, max].
       *
       * NOTE: this function will only return sensible results if the employed
       * hash function is monotone
       *
       * @param min minimum key value (inclusive)
       * @param max maximum key value (inclusive)
       * @return a vector of payloads associated with keys within the range, if
       *   any exist. Otherwise empty.
       */
      std::vector<Payload> lookup_range(const Key& min, const Key& max) {
         if (unlikely(min == Sentinel || max == Sentinel)) {
            assert(false); // TODO(dominik): this must never happen in practice
            return {};
         }

         // min will be in this slot or a subsequent slot
         const auto lower_bound_index = reductionfn(hashfn(min));

         // edge case: we've got an exact inline match
         std::vector<Payload> result;

         // loop variables
         auto i = lower_bound_index;
         bool continue_until_next_slot = true;
         for (; i<slots.size() && continue_until_next_slot; i++) {
            // get the current slot
            auto& current_slot = slots[i];
            // start from the current slot
            if (current_slot.key >= min && current_slot.key <= max)
               result.push_back(current_slot.payload);
            if (current_slot.key >= max && current_slot.key != Sentinel)
               continue_until_next_slot = false;
            // now, proceed with the buckets
            Bucket* bucket = current_slot.buckets;
            while (bucket != nullptr) {
               for (size_t i = 0; i < BucketSize; i++) {
                  Key k = bucket->slots[i].key;
                  if (k >= min && k <= max) {
                     // add payload to result
                     result.push_back(bucket->slots[i].payload);
                  }
                  // if we encounter something >= max in the bucket chain, we don't need to continue
                  if (k >= max && k != Sentinel)
                     continue_until_next_slot = false;
                  // empty slot -> no futher bucket
                  if (k == Sentinel)
                     break;
               }
               bucket = bucket->next;
            }
         }
         return result;
      }

      std::map<std::string, double> lookup_statistics(const std::vector<Key>& dataset) {
         UNUSED(dataset);

         size_t max_chain_length = 0;
         size_t min_chain_length = std::numeric_limits<size_t>::max();
         size_t empty_buckets = 0;
         size_t additional_buckets = 0;
         size_t empty_additional_slots = 0;

         for (const auto& slot : slots) {
            if (slot.key == Sentinel) {
               empty_buckets++;
               continue;
            }

            size_t chain_length = 0;
            Bucket* b = slot.buckets;
            while (b != nullptr) {
               chain_length++;
               additional_buckets++;

               for (const auto& s : b->slots)
                  empty_additional_slots += s.key == Sentinel ? 1 : 0;

               b = b->next;
            }

            min_chain_length = std::min(min_chain_length, chain_length);
            max_chain_length = std::max(max_chain_length, chain_length);
         }

         return {{"empty_buckets", empty_buckets},
                 {"min_chain_length", min_chain_length},
                 {"max_chain_length", max_chain_length},
                 {"additional_buckets", additional_buckets},
                 {"empty_additional_slots", empty_additional_slots}};
      }

      size_t byte_size() const {
         size_t size = sizeof(decltype(*this)) + slots.size() * slot_byte_size();
         for (const auto& slot : slots)
            size += slot.buckets == nullptr ? 0 : slot.buckets->byte_size();

         return size;
      }

      static constexpr forceinline size_t bucket_byte_size() {
         return sizeof(Bucket);
      }

      static constexpr forceinline size_t slot_byte_size() {
         return sizeof(FirstLevelSlot);
      }

      static forceinline std::string name() {
         return "chained_" + hash_name() + "_" + reducer_name() + "_" + std::to_string(bucket_size());
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
         return capacity;
      }

      /**
       * Clears all keys from the hashtable. Note that payloads are technically
       * still in memory (i.e., might leak if sensitive).
       */
      void clear() {
         for (auto& slot : slots) {
            slot.key = Sentinel;

            auto bucket = slot.buckets;
            slot.buckets = nullptr;

            while (bucket != nullptr) {
               auto next = bucket->next;
               delete bucket;
               bucket = next;
            }
         }
      }

      ~Chained() {
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
         } packit;

         std::array<Slot, BucketSize> slots /*__attribute((aligned(sizeof(Key) * 8)))*/;
         Bucket* next = nullptr;

         size_t byte_size() const {
            return sizeof(decltype(*this)) + (next == nullptr ? 0 : next->byte_size());
         }
      } packit;

      struct FirstLevelSlot {
         Key key = Sentinel;
         Payload payload;
         Bucket* buckets = nullptr;
      } packit;

      // First bucket is always inline in the slot
      std::vector<FirstLevelSlot> slots;
      std::vector<omp_lock_t> locks;
   };
} // namespace hashtable
