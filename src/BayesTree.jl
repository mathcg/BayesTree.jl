module BayesTree

  using
    Distributions,
    StatsBase

  export
   #type
   Decision,
   DecisionTree,
   DecisionNode,
   DecisionBranch,
   RegressionLeaf,
   ClassificationLeaf,
   BartProbability,
   BartOptions,
   BartStateParameters,
   BartTree,
   BartState,
   BartAdditiveTree,
   Bart,
   BartLeaf,
   Tree,
   Node,
   Branch,
   Leaf,

   #methods
   leaves,
   valid_node,
   bart_options,
   depth,
   branches,
   grand_branches,
   not_grand_branches,
   length,
   parent,
   node_nonterminal,
   log_tree_prior,
   node_loglikelihood,
   fit,
   predict,
   node_grow!,
   node_prune!,
   node_grow_prune!,
   change_decision_rule!,
   swap_decision_rule!,
   normalize,
   denormalize,
   initialize_bart_state

   include("tree.jl")
   include("decision_tree.jl")
   include("bart_new.jl")

end # module
