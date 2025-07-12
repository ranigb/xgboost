#ifndef XGBOOST_PLUGIN_TREEG_NODE_TREEG_NODE_UPDATER_H_
#define XGBOOST_PLUGIN_TREEG_NODE_TREEG_NODE_UPDATER_H_

#include <xgboost/tree_updater.h>
#include <xgboost/parameter.h>
#include <string>

namespace xgboost {
namespace tree {

struct TreeGNodeTrainParam : XGBoostParameter<TreeGNodeTrainParam> {
  // Simple parameter for demonstration
  float leaf_value;
  
  DMLC_DECLARE_PARAMETER(TreeGNodeTrainParam) {
    DMLC_DECLARE_FIELD(leaf_value)
        .set_default(0.0f)
        .describe("Default leaf value for treeg_node updater");
  }
};

DMLC_REGISTER_PARAMETER(TreeGNodeTrainParam);

class TreeGNodeUpdater : public TreeUpdater {
 public:
  explicit TreeGNodeUpdater(const Context* ctx) : TreeUpdater(ctx) {}
  char const* Name() const override { return "treeg_node"; }
  void Configure(const Args& args) override;
  void Update(tree::TrainParam const* param, linalg::Matrix<GradientPair>* gpair,
              DMatrix* data, common::Span<HostDeviceVector<bst_node_t>> out_position,
              const std::vector<RegTree*>& out_trees) override;
  bool CanModifyTree() const override { return false; }
  bool HasNodePosition() const override { return true; }
  void LoadConfig(Json const&) override;
  void SaveConfig(Json*) const override;

 protected:
  TreeGNodeTrainParam param_;
};

}  // namespace tree
}  // namespace xgboost

#endif  // XGBOOST_PLUGIN_TREEG_NODE_TREEG_NODE_UPDATER_H_ 