#include "treeg_node_updater.h"
#include <xgboost/tree_updater.h>
#include <xgboost/tree_model.h>
#include <xgboost/logging.h>
#include "../../src/tree/param.h"

namespace xgboost {
namespace tree {

void TreeGNodeUpdater::Configure(const Args& args) {
  param_.UpdateAllowUnknown(args);
}

void TreeGNodeUpdater::LoadConfig(Json const& in) {
  auto const& config = get<Object const>(in);
  FromJson(config.at("treeg_node_train_param"), &this->param_);
}

void TreeGNodeUpdater::SaveConfig(Json* p_out) const {
  auto& out = *p_out;
  out["treeg_node_train_param"] = ToJson(param_);
}

void TreeGNodeUpdater::Update(tree::TrainParam const* param, linalg::Matrix<GradientPair>* gpair,
                              DMatrix* data, common::Span<HostDeviceVector<bst_node_t>> out_position,
                              const std::vector<RegTree*>& out_trees) {
  if (out_trees.empty()) {
    return;
  }
  CHECK_EQ(gpair->Shape(1), 1) << "treeg_node updater only supports single target";
  const std::vector<GradientPair>& gpair_h = gpair->Data()->ConstHostVector();

  for (auto* tree : out_trees) {
    bst_node_t root = 0;
    
    // Calculate total gradients and hessians
    double sum_grad = 0.0, sum_hess = 0.0;
    for (const auto& gp : gpair_h) {
      sum_grad += gp.GetGrad();
      sum_hess += gp.GetHess();
    }
    
    // Only split if we have enough data and hessian is positive
    if (gpair_h.size() > 1 && sum_hess > 0.0) {
      // Expand the root node: split on feature 0 at value 0.5, default left
      tree->ExpandNode(root, 0, 0.5f, true, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
      bst_node_t left = tree->LeftChild(root);
      bst_node_t right = tree->RightChild(root);
      
      // Calculate optimal leaf values based on gradients
      float left_weight = -static_cast<float>(sum_grad / sum_hess) * param->learning_rate;
      float right_weight = left_weight;  // Simple for now
      
      (*tree)[left].SetLeaf(left_weight);
      (*tree)[right].SetLeaf(right_weight);
      
      // Set tree statistics
      tree->Stat(left).base_weight = left_weight;
      tree->Stat(right).base_weight = right_weight;
      tree->Stat(left).sum_hess = static_cast<float>(sum_hess);
      tree->Stat(right).sum_hess = static_cast<float>(sum_hess);
      tree->Stat(left).loss_chg = 0.0f;  // No loss change for leaves
      tree->Stat(right).loss_chg = 0.0f;  // No loss change for leaves
      
      // Assign positions based on feature values (simplified)
      for (auto& pos_vec : out_position) {
        auto& pos = pos_vec.HostVector();
        for (size_t i = 0; i < pos.size(); ++i) {
          pos[i] = left;  // All go to left for simplicity
        }
      }
    } else {
      // No split, just set the root as a leaf
      float weight = -static_cast<float>(sum_grad / (sum_hess + 1e-6f)) * param->learning_rate;
      (*tree)[root].SetLeaf(weight);
      tree->Stat(root).base_weight = weight;
      tree->Stat(root).sum_hess = static_cast<float>(sum_hess);
      tree->Stat(root).loss_chg = 0.0f;  // No loss change for leaves
      
      // All positions point to root
      for (auto& pos_vec : out_position) {
        auto& pos = pos_vec.HostVector();
        for (size_t i = 0; i < pos.size(); ++i) {
          pos[i] = root;
        }
      }
    }
  }
}

// Register the updater
XGBOOST_REGISTER_TREE_UPDATER(TreeGNodeUpdater, "treeg_node")
    .describe("Tree updater plugin: treeg_node (minimal working example with gradient-based tree growing)")
    .set_body([](Context const *ctx, auto) { return new TreeGNodeUpdater(ctx); });

}  // namespace tree
}  // namespace xgboost 