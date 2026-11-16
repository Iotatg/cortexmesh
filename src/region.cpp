#include "bm/region.hpp"

namespace bm {

SpatialPoolerParams Region::MakePoolerParams(const RegionParams& p) {
  SpatialPoolerParams sp;
  sp.input_size = p.input_size;
  sp.column_count = p.column_count;
  sp.synapses_per_column = p.synapses_per_column;
  sp.active_columns = p.active_columns;
  sp.seed = p.seed;
  sp.feedback_gain = p.feedback_gain;
  return sp;
}

TemporalMemoryParams Region::MakeTmParams(const RegionParams& p) {
  TemporalMemoryParams tm;
  tm.column_count = p.column_count;
  tm.cells_per_column = p.cells_per_column;
  tm.seed = p.seed ^ 0x5DEECE66Dull;
  tm.use_predictive_in_prev = p.use_predictive_in_prev;
  tm.activation_threshold = p.activation_threshold;
  tm.new_segment_min = p.new_segment_min;
  return tm;
}

Region::Region(const RegionParams& p)
    : p_(p),
      sp_(MakePoolerParams(p)),
      tm_(MakeTmParams(p)),
      stable_(p_.column_count, 0) {}

RegionStep Region::step(const SDR& input) {
  RegionStep out;
  out.input = input;
  out.columns = sp_.activate(input);
  TMStep st = tm_.step(out.columns);
  out.active = st.active;
  out.predictive = st.predictive;
  out.bursting_columns = st.bursting_columns;
  out.predicted_active = st.predicted_active;
  out.anomaly = tm_.anomaly_score();

  stable_rate_ = stable_rate_ * 0.95 + (st.bursting_columns > 0 ? 0.0 : 1.0);
  if (st.bursting_columns == 0) stable_ = out.columns;
  return out;
}

RegionStep Region::step(const SDR& input, const SDR& feedback) {
  RegionStep out;
  out.input = input;
  out.columns = sp_.activate(input, feedback);
  TMStep st = tm_.step(out.columns);
  out.active = st.active;
  out.predictive = st.predictive;
  out.bursting_columns = st.bursting_columns;
  out.predicted_active = st.predicted_active;
  out.anomaly = tm_.anomaly_score();
  stable_rate_ = stable_rate_ * 0.95 + (st.bursting_columns > 0 ? 0.0 : 1.0);
  if (st.bursting_columns == 0) stable_ = out.columns;
  return out;
}

SDR Region::predicted_columns() const {
  SDR out(p_.column_count, 0);
  for (uint32_t cell : tm_.predictive().indices()) {
    out.set(static_cast<uint32_t>(tm_.column_of(cell)));
  }
  return out;
}

}  // namespace bm