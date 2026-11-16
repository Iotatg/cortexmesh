#include "bm/dual_controller.hpp"

#include <algorithm>
#include <cmath>

namespace bm {

DualController::DualController(const DualControllerParams& p)
    : p_(p),
      enc_(p_.code_size, p_.code_k, p_.seed),
      outcome_value_(p_.code_size, 0.05f, 0.85f),
      hp_(p_.state_size + p_.action_size, p_.dentate_size, p_.dentate_k, 16, p_.seed ^ 0xABCDull) {}

uint64_t DualController::key_of(const SDR& state, const SDR& action) const {
  SDR conj = enc_.conjunction({state, action});
  return conj.hash() ^ (static_cast<uint64_t>(state.count()) << 32) ^ action.count();
}

float DualController::model_free_value(const SDR& state, const SDR& action) const {
  auto it = action_values_.find(action.hash());
  if (it == action_values_.end()) return 0.0f;
  return it->second.predict(state);
}

float DualController::model_based_value(const SDR& state, const SDR& action) const {
  auto it = outcomes_.find(key_of(state, action));
  if (it == outcomes_.end()) return 0.0f;
  return outcome_value_.predict(it->second);
}

void DualController::learn_model_free(const SDR& state, const SDR& action, float reward) {
  Valuation& v = action_values_[action.hash()];
  v.present(state);
  v.reward(reward);
  hp_.store(enc_.conjunction({state, action}));
}

void DualController::learn_model_based(const SDR& state, const SDR& action, const SDR& outcome,
                                       float outcome_reward) {
  outcomes_[key_of(state, action)] = outcome;
  outcome_value_.present(outcome);
  outcome_value_.reward(outcome_reward);
  hp_.store(enc_.conjunction({state, outcome}));
}

void DualController::tick() {
  for (auto& kv : action_values_) kv.second.tick();
  outcome_value_.tick();
}

Decision DualController::decide(const SDR& state, const std::vector<SDR>& actions, float uncertainty) {
  Decision best;
  bool have = false;
  float second = -1e9f;
  for (const SDR& action : actions) {
    float q_mf = model_free_value(state, action) - p_.cost;
    float q_mb = model_based_value(state, action) - p_.cost;
    bool use_mb = uncertainty > p_.uncertainty_threshold && q_mb > 0.0f;
    float q = use_mb ? q_mb : q_mf;
    if (!have || q > best.value) {
      second = have ? best.value : second;
      best.value = q;
      best.model_free_value = q_mf;
      best.model_based_value = q_mb;
      best.model_based = use_mb;
      best.action = action;
      have = true;
    } else if (q > second) {
      second = q;
    }
  }
  best.confidence = have ? std::fabs(best.value - second) : 0.0f;
  return best;
}

}  // namespace bm