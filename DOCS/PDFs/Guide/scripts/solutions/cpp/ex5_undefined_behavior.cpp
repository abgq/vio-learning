#include <array>
#include <climits>
#include <cstdio>
#include <cstdlib>

int read_past_end(int i) {
  std::array<int, 4> a{1, 2, 3, 4};
  return a[static_cast<std::size_t>(i)];  // i == 4: one past the end
}

int add_one(int x) { return x + 1; }      // x == INT_MAX overflows

int main(int argc, char** argv) {
  const char* which = argc > 1 ? argv[1] : "array";
  if (which[0] == 'a') {
    std::printf("a[4] = %d\n", read_past_end(3 + argc / 2 + (argc > 1 ? 0 : 1)));
  } else {
    std::printf("INT_MAX + 1 = %d\n", add_one(INT_MAX - 1 + (argc > 1 ? 1 : 0)));
  }
  return 0;
}
