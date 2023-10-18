// this spinlock implementation can be found here https://rigtorp.se/spinlock/
# pragma once

#if defined(__x86_64__)
#define X86_ARCHITECTURE
#elif defined(__arm__)
#define ARM_ARCHITECTURE
#else
#define UNKNOWN_ARCHITECTURE
#endif

#include <atomic>

struct spinlock {
  std::atomic<bool> lock_ = {0};

  void lock() noexcept {
    for (;;) {
      // Optimistically assume the lock is free on the first try
      if (!lock_.exchange(true, std::memory_order_acquire)) {
        return;
      }
      // Wait for lock to be released without generating cache misses
      while (lock_.load(std::memory_order_relaxed)) {
        // Issue X86 PAUSE or ARM YIELD instruction to reduce contention between
        // hyper-threads TODO
        #ifdef X86_ARCHITECTURE
        // Use x86-specific code
        __builtin_ia32_pause();
        #elif defined(ARM_ARCHITECTURE)
        // Use ARM-specific code
        __builtin_arm_yield();
        #endif
      }
    }
  }

  bool try_lock() noexcept {
    // First do a relaxed load to check if lock is free in order to prevent
    // unnecessary cache misses if someone does while(!try_lock())
    return !lock_.load(std::memory_order_relaxed) &&
           !lock_.exchange(true, std::memory_order_acquire);
  }

  void unlock() noexcept {
    lock_.store(false, std::memory_order_release);
  }
};
