#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "bm/encoder.hpp"
#include "bm/sdr.hpp"
#include "bm/spatial_pooler.hpp"
#include "bm/temporal_memory.hpp"

using namespace bm;

static void usage() {
  std::printf(
      "brain_cli <cmd>\n"
      "  encode <text>            print SDR for a symbol\n"
      "  seq                    train TM on a random sequence and report accuracy\n"
      "  anomaly                report anomaly score on unseen input\n");
}

static int cmd_encode(const std::string& text) {
  Encoder enc(2048, 40, 5);
  SDR s = enc.symbol(text);
  std::printf("%s\n", s.to_string().c_str());
  return 0;
}

static SDR col_code(size_t n, size_t k, size_t i) {
  Rng rng(0xABCD + i * 7919);
  return SDR::random(n, k, rng.next());
}

static int cmd_seq() {
  SpatialPoolerParams sp_p;
  sp_p.input_size = 1024;
  sp_p.column_count = 4096;
  sp_p.synapses_per_column = 128;
  sp_p.active_columns = 20;
  SpatialPooler sp(sp_p);

  TemporalMemoryParams tm_p;
  tm_p.column_count = 4096;
  tm_p.cells_per_column = 16;
  tm_p.activation_threshold = 2;
  tm_p.permanence_threshold = 10;
  TemporalMemory tm(tm_p);

  std::vector<SDR> seq;
  for (size_t i = 0; i < 32; ++i) seq.push_back(col_code(1024, 20, i));

  for (int epoch = 0; epoch < 30; ++epoch) {
    for (const SDR& s : seq) {
      SDR cols = sp.activate(s);
      tm.step(cols);
    }
  }

  size_t correct = 0;
  size_t bursts = 0;
  for (const SDR& s : seq) {
    SDR cols = sp.activate(s);
    TMStep st = tm.step(cols);
    if (st.bursting_columns == 0) ++correct;
    bursts += st.bursting_columns;
  }
  std::printf("trained=%zu correct=%zu bursts=%zu accuracy=%.3f\n", seq.size(), correct,
              bursts, static_cast<double>(correct) / static_cast<double>(seq.size()));

  SDR unseen = col_code(1024, 20, 999);
  TMStep st = tm.step(sp.activate(unseen));
  std::printf("anomaly(unseen)=%.3f bursting=%zu\n", tm.anomaly_score(), st.bursting_columns);
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    usage();
    return 2;
  }
  std::string cmd = argv[1];
  if (cmd == "encode" && argc > 2) return cmd_encode(argv[2]);
  if (cmd == "seq") return cmd_seq();
  usage();
  return 2;
}