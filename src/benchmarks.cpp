#include <cstdint>
#include <vector>

#include <exotic_hashing.hpp>

int main() {
   const std::vector<std::uint64_t> vec{1, 2, 3, 4};
   exotic_hashing::MWHC<std::uint64_t> mwhc(vec.begin(), vec.end());

   return mwhc(1);
}
