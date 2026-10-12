#include <variant>
#include <vector>
#include <cstdio>
struct ImuSample { double t; };
struct ImageEvent { double t; };
struct PoseEvent { double t; };
using SensorEvent = std::variant<ImuSample, ImageEvent, PoseEvent>;
template <class... Ts> struct Overloaded : Ts... { using Ts::operator()...; };
template <class... Ts> Overloaded(Ts...) -> Overloaded<Ts...>;
int main() {
  std::vector<SensorEvent> events{ImuSample{0.0}, ImageEvent{0.1}, PoseEvent{0.2}};
  int imu = 0, img = 0;
  for (const SensorEvent& e : events) {
    std::visit(Overloaded{
        [&](const ImuSample&) { ++imu; },
        [&](const ImageEvent&) { ++img; },
    }, e);
  }
  std::printf("%d %d\n", imu, img);
}
