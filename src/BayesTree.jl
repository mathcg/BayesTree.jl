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
   predict

   include("tree.jl")
   include("decision_tree.jl")
   include("bart_new.jl")

end # module
