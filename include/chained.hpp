#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <map>
#include <memory>
#include <optional>
#include <vector>

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
           slots(directory_address_count(capacity)){};

      Chained(Chained&&) = default;

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
            assert(false); // TODO: this must never happen in practice
            return false;
         }

         // Using template functor should successfully inline actual hash computation
         FirstLevelSlot& slot = slots[reductionfn(hashfn(key))];

         // Store directly in slot if possible
         if (slot.key == Sentinel) {
            slot.key = key;
            slot.payload = payload;
            return true;
         }

         // Initialize bucket chain if empty
         Bucket* bucket = slot.buckets;
         if (bucket == nullptr) {
            auto b = new Bucket();
            b->slots[0] = {.key = key, .payload = payload};
            slot.buckets = b;
            return true;
         }

         // Go through existing buckets and try to insert there if possible
         for (;;) {
            // Find suitable empty entry place. Note that deletions with holes will require
            // searching entire bucket to deal with duplicate keys!
            for (size_t i = 0; i < BucketSize; i++) {
               if (bucket->slots[i].key == Sentinel) {
                  bucket->slots[i] = {.key = key, .payload = payload};
                  return true;
               } else if (bucket->slots[i].key == key) {
                  // key already exists
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
            assert(false); // TODO: this must never happen in practice
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

      std::map<std::string, std::string> lookup_statistics(const std::vector<Key>& dataset) {
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

         return {{"empty_buckets", std::to_string(empty_buckets)},
                 {"min_chain_length", std::to_string(min_chain_length)},
                 {"max_chain_length", std::to_string(max_chain_length)},
                 {"additional_buckets", std::to_string(additional_buckets)},
                 {"empty_additional_slots", std::to_string(empty_additional_slots)}};
      }

      static constexpr forceinline size_t bucket_byte_size() {
         return sizeof(Bucket);
      }

      static constexpr forceinline size_t slot_byte_size() {
         return sizeof(FirstLevelSlot);
      }

      static forceinline std::string name() {
         return "chained";
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
      }

     protected:
      struct Bucket {
         struct Slot {
            Key key = Sentinel;
            Payload payload;
         } packed;

         std::array<Slot, BucketSize> slots /*__attribute((aligned(sizeof(Key) * 8)))*/;
         Bucket* next = nullptr;
      } packed;

      struct FirstLevelSlot {
         Key key = Sentinel;
         Payload payload;
         Bucket* buckets = nullptr;
      } packed;

      // First bucket is always inline in the slot
      std::vector<FirstLevelSlot> slots;
   };
} // namespace hashtable
