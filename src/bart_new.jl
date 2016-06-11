type BartProbability #Records the proposal probabilities for moves
  node_grow_prune::Float64
  change_decision_rule::Float64
  swap_decision_rule::Float64

  function BartProbability(n,c,s)
    assert(n+c+s==1)
    new(n,c,s)
  end
end

BartProbability() = BartProbability(0.5,0.4,0.1)

type BartOptions
  num_trees::Int #number of additive trees
  num_draws::Int #number of posterior draws
  num_burn_in::Int
  num_thinning::Int
  alpha::Float64 #calculating nonterminal probability
  beta::Float64
  k::Int
  proposal_probabilities::BartProbability
end

function bart_options(;num_trees::Int=200,num_draws::Int=1000,
                     num_burn_in::Int=200,num_thinning::Int=10,alpha::Float64=0.95,beta::Float64=2.0,
                     k::Int=2,proposal_probabilities=BartProbability())
   BartOptions(num_trees,num_draws,num_burn_in,num_thinning,alpha,beta,k,proposal_probabilities)
end

type BartStateParameters
  sigma::Float64
  sigma_mu::Float64
  nu::Int64
  lambda::Float64
end

type BartTree
  tree::DecisionTree
end

@make_tree_type(BartTree,DecisionNode)

type BartState
  trees::Vector{BartTree}
  parameters::BartStateParameters
end

type BartAdditiveTree
  trees::Vector{BartTree}
end

type Bart
  bart_additive_trees::Vector{BartAdditiveTree} #store the posterior draw of additive tree
  y_min::Float64
  y_max::Float64
  options::BartOptions
end

type BartLeaf <: DecisionLeaf
  value::Float64 #\mu_{ij}
  residual_mean::Float64
  residual_sigma::Float64
  leaf_data_indices::Vector{Int} #all the training data contained in this leaf

  function BartLeaf(value::Float64,residual::Vector{Float64},leaf_data_indices::Vector{Int})
        #here, we don't allow leaves with no training data
        residual_mean = mean(residual[leaf_data_indices])
        residual_sigma = sqrt(mean((residual[leaf_data_indices].-residual_mean).^2))
        new(value,residual_mean,residual_sigma,leaf_data_indices)
  end
end

function node_nonterminal(depth::Int,alpha::Float64=0.95,beta::Float64=2.0)
  alpha*depth^(-beta)
end

node_nonterminal(depth::Int,bartoptions::BartOptions) = node_nonterminal(depth,bartoptions.alpha,bartoptions.beta)

function log_tree_prior(branch::DecisionBranch,depth::Int,bartoptions::BartOptions)
    prior = log(node_nonterminal(depth,bartoptions))
    #Here, we don't consider the probability of selecting available features
    #We only consider the probability of selecting available values
    #Here, it's possible that length(train_data_indices(branch))==0,then prior = Inf
    prior -= log(length(train_data_indices(branch)))
    prior += log_tree_prior(branch.left,depth+1,bartoptions)
    prior += log_tree_prior(branch.right,depth+1,bartoptions)
    prior
end

log_tree_prior(leaf::BartLeaf,depth::Int,bartoptions::BartOptions) = log(1-node_nonterminal(depth,bartoptions))

node_loglikelihood(branch::DecisionBranch,bart_state::BartState) = node_loglikelihood(branch.left,bart_state)+node_loglikelihood(branch.right,bart_state)

function node_loglikelihood(leaf::BartLeaf,bart_state::BartState)
    n = length(leaf.leaf_data_indices)
    #Here, we don't allow leaf with no train_data_indices
    if n==0
        return Inf
    end
    loglikelihood =
      0.5*(log(bart_state.parameters.sigma^2)-log(bart_state.parameters.sigma^2+n*bart_state.parameters.sigma_mu^2))
    loglikelihood -= 0.5*n*leaf.residual_sigma^2/bart_state.parameters.sigma^2
    loglikelihood -= 0.5*n*leaf.residual_mean^2/(n*bart_state.parameters.sigma_mu^2+bart_state.parameters.sigma^2)
    loglikelihood
end

function update_tree!(bart_state::BartState,tree::BartTree,x::Matrix{Float64},residual::Vector{Float64},bartoptions::BartOptions)
    update_probability = rand()
    prob = bartoptions.proposal_probabilities #probabilities for posterior proposal moves

    if update_probability<prob.node_grow_prune
      updated = node_grow_prune!(bart_state,tree,x,residual,bartoptions)
    elseif update_probability<prob.node_grow_prune+prob.change_decision_rule
      updated = change_decision_rule!(bart_state,tree,x,residual,bartoptions)
    else
      updated = swap_decision_rule!(bart_state,tree,x,residual,bartoptions)
    end
    #after we sample T_j, now we sample M_j given T_j
    #no matter we update updated or not, we update M_j(leaf_values)
    if updated
      update_leaf_values!(tree,bart_state.parameters)
    end
    updated
end

function node_grow_prune!(bart_state::BartState,tree::BartTree,x::Matrix{Float64},residual::Vector{Float64},bartoptions::BartOptions)
    #if the tree only has one root,then we must grow
    probability_grow = typeof(tree.tree.root)==BartLeaf?1.0:0.5
    u = rand()
    if u<probability_grow
      updated = node_grow!(bart_state,tree,probability_grow,x,residual,bartoptions)
    else
      probability_prune = 1-probability_grow
      updated = node_prune!(bart_state,tree,probability_prune,x,residual,bartoptions)
    end
    updated
end

function node_grow!(bart_state::BartState,tree::BartTree,probability_grow::Float64,x::Matrix{Float64},residual::Vector{Float64},bartoptions::BartOptions)
    #first we need to sample a leave to grow new leaves
    leaf_nodes = leaves(tree)
    number_leaves = length(leaf_nodes)
    i = rand(1:number_leaves)
    leaf = leaf_nodes[i]
    leaf_indices = leaf.leaf_data_indices

    #it's impossible to split a leaf with no data points
    #it's feasible to split a leaf with one data points
    if length(leaf_indices)==0
      return false
    end

    leaf_depth = depth(tree,leaf)
    leaf_loglikelihood = node_loglikelihood(leaf,bart_state)
    leaf_nonterminal = node_nonterminal(leaf_depth,bartoptions)

    #uniformly sample a splitting feature from the available predictors

    #split_feature = rand(1:size(x,2))
     split_feature = rand(1:size(x,1))
    #uniformly sample a splitting location from the available locations
    #feature = x[leaf_indices,split_feature]
    feature = vec(x[split_feature,leaf_indices])
    split_location = rand(1:length(feature))
    split_value = feature[split_location]

    #It's possible that length(left_indices) =0 or length(right_indices)=0
    left_indices = leaf_indices[feature.<=split_value]
    right_indices = leaf_indices[feature.>split_value]
    if length(left_indices)*length(right_indices)==0
       return false
    end
    left_leaf_nonterminal = length(left_indices)>0?node_nonterminal(leaf_depth+1,bartoptions):0.0
    right_leaf_nonterminal = length(right_indices)>0?node_nonterminal(leaf_depth+1,bartoptions):0.0

    #construct the new branch and new leaf
    left_leaf = BartLeaf(0.0,residual,left_indices)
    right_leaf = BartLeaf(0.0,residual,right_indices)
    branch = DecisionBranch(split_feature,split_value,left_leaf,right_leaf)
    left_leaf_loglikelihood = node_loglikelihood(left_leaf,bart_state)
    right_leaf_loglikelihood = node_loglikelihood(right_leaf,bart_state)

    if left_leaf_loglikelihood==Inf || right_leaf_loglikelihood==Inf
      return false
    end

    #calculate the number of internal nodes with two terminal nodes of the original tree, which is the
    #not_grand_branches
    not_grand_branch_nodes = not_grand_branches(tree)
    number_not_grand_branch_nodes = length(not_grand_branch_nodes)

    parent_leaf = parent(tree,leaf) #it's posssible that there is no parent
    if parent_leaf!=nothing
      if typeof(parent_leaf.left)==BartLeaf && typeof(parent_leaf.right)==BartLeaf
         number_not_grand_branch_nodes = number_not_grand_branch_nodes
      else
      number_not_grand_branch_nodes += 1
      end
    else
      number_not_grand_branch_nodes += 1
    end

    #the probability of pruning a node from the proposal tree(after grow)
    probability_prune = 0.5

    alpha = (1-left_leaf_nonterminal)*(1-right_leaf_nonterminal)*leaf_nonterminal*number_leaves*probability_prune
    alpha /= (1-leaf_nonterminal)*number_not_grand_branch_nodes*probability_grow
    alpha *= exp(left_leaf_loglikelihood+right_leaf_loglikelihood-leaf_loglikelihood)

    #decide whether or not to grow two leaves
    if rand()<alpha
       if parent_leaf==nothing
         tree.tree.root = branch
       elseif leaf==parent_leaf.left
         parent_leaf.left = branch
       else
         parent_leaf.right = branch
       end
       updated = true
    else
       updated = false
    end
    updated
end


function node_prune!(bart_state::BartState,tree::BartTree,probability_prune::Float64,x::Matrix{Float64},residual::Vector{Float64},bartoptions::BartOptions)
  if typeof(tree.tree.root.left)==BartLeaf && typeof(tree.tree.root.right)==BartLeaf
     probability_grow = 1.0
  else
     probability_grow = 0.5
  end

  not_grand_branch_nodes = not_grand_branches(tree)
  number_not_grand_branch_nodes = length(not_grand_branch_nodes)
  i = rand(1:number_not_grand_branch_nodes)
  not_grand_branch = not_grand_branch_nodes[i]

  leaf_nodes = leaves(tree)
  number_leaves = length(leaf_nodes)

  not_grand_branch_depth = depth(not_grand_branch)
  not_grand_branch_nonterminal = node_nonterminal(not_grand_branch_depth,bartoptions)
  left_leaf_nonterminal = length(not_grand_branch.left.leaf_data_indices)>0?node_nonterminal(not_grand_branch_depth+1,bartoptions):0.0
  right_leaf_nonterminal = length(not_grand_branch.right.leaf_data_indices)>0?node_nonterminal(not_grand_branch_depth+1,bartoptions):0.0

  not_grand_branch_loglikelihood = node_loglikelihood(not_grand_branch,bart_state)
  #here, I construct a new leaf by combining the information in the left and right leaves of the not_grand_branch
  not_grand_branch_data_indices = vcat(not_grand_branch.left.leaf_data_indices,not_grand_branch.right.leaf_data_indices)
  new_leaf = BartLeaf(0.0,residual,not_grand_branch_data_indices)
  proposal_new_leaf_loglikelihood= node_loglikelihood(new_leaf,bart_state)

  alpha = (1-not_grand_branch_nonterminal)*number_not_grand_branch_nodes*probability_grow
  alpha /= (1-left_leaf_nonterminal)*(1-right_leaf_nonterminal)*not_grand_branch_nonterminal*(number_leaves-1)*probability_prune
  alpha *= exp(proposal_new_leaf_loglikelihood-not_grand_branch_loglikelihood)

  parent_not_grand_branch = parent(tree,not_grand_branch)

  if rand() < alpha
    if parent_not_grand_branch == nothing
       tree.tree.root = new_leaf
    elseif not_grand_branch==parent_not_grand_branch.left
          parent_not_grand_branch.left = new_leaf
    else
          parent_not_grand_branch.right = new_leaf
    end
    updated = true
  else
    updated = false
  end
  updated
end


train_data_indices(leaf::BartLeaf) = leaf.leaf_data_indices

function train_data_indices(branch::DecisionBranch)
  function train_data_indices!(branch::DecisionBranch,indices::Vector{Int})
    train_data_indices!(branch.left,indices)
    train_data_indices!(branch.right,indices)
  end

  function train_data_indices!(leaf::BartLeaf,indices::Vector{Int})
    for i in 1:length(leaf.leaf_data_indices)
        push!(indices,leaf.leaf_data_indices[i])
    end
  end

  indices = Int[]
  train_data_indices!(branch,indices)
  indices
end


function tree_adjust!(parent::DecisionBranch,leaf::BartLeaf,x::Matrix{Float64},residual::Vector{Float64},branch_indices::Vector{Int},left::Bool)
   valid_split=true
   if left
     parent.left = BartLeaf(leaf.value,residual,branch_indices)
   else
     parent.right = BartLeaf(leaf.value,residual,branch_indices)
   end
   if length(branch_indices)==0
     valid_split = false
   end
   valid_split
end
#it might be possible that length(branch_indices)==0
function tree_adjust!(parent::DecisionBranch,branch::DecisionBranch,x::Matrix{Float64},residual::Vector{Float64},branch_indices::Vector{Int},left::Bool)
  tree_adjust!(branch,x,residual,branch_indices)
end

function tree_adjust!(branch::DecisionBranch,x::Matrix{Float64},residual::Vector{Float64},branch_indices::Vector{Int64})
    valid_split = true
    if length(branch_indices)==0
        left_data_indices = branch_indices
        right_data_indices = branch_indices
        valid_split = false
    else
        #feature = x[branch_indices,branch.feature]
        feature = vec(x[branch.feature,branch_indices])
        left_data_indices = branch_indices[feature.<=branch.value]
        right_data_indices = branch_indices[feature.>branch.value]
    end

    valid_split = tree_adjust!(branch,branch.left,x,residual,left_data_indices,true)
    valid_split = tree_adjust!(branch,branch.right,x,residual,right_data_indices,false)
    valid_split
end

function change_decision_rule!(bart_state::BartState,tree::BartTree,x::Matrix{Float64},residual::Vector{Float64},bartoptions::BartOptions)
  branch_nodes = branches(tree)

  number_branches = length(branch_nodes)
  if number_branches ==0
     return false
  end

  i = rand(1:number_branches)
  branch = branch_nodes[i]
  branch_depth = depth(branch)

  branch_indices = train_data_indices(branch)
  #because here we allow leaves with no training data, it's possible that
  #branch_indices is empty
  if length(branch_indices)==0
      return false
  end

  #split_feature = rand(1:size(x,2))
  #feature = x[branch_indices,split_feature]
  split_feature = rand(1:size(x,1))
  feature = vec(x[split_feature,branch_indices])

  split_location = rand(1:length(feature))
  split_value = feature[split_location]

  #calculate the prior tree loglikelihood
  branch_loglikelihood = node_loglikelihood(branch,bart_state)
  branch_prior = log_tree_prior(branch,branch_depth,bartoptions)

  #calculate the proposal tree loglikelihood
  old_feature = branch.feature
  old_value = branch.value

  branch.feature = split_feature
  branch.value = split_value
  valid_split = tree_adjust!(branch,x,residual,branch_indices)
  if !valid_split
    branch.feature = old_feature
    branch.value = old_value
    tree_adjust!(branch,x,residual,branch_indices)
    updated = false
    return updated
  end
  proposal_branch_loglikelihood = node_loglikelihood(branch,bart_state)
  #when the following value equals to Inf, it means that there is a branch that has no training data indices
  proposal_branch_prior = log_tree_prior(branch,branch_depth,bartoptions) #value might be Inf

  #calculate alpha
  if proposal_branch_loglikelihood==Inf
    alpha = 0.0
  else
    alpha = proposal_branch_prior==Inf?0.0:exp(proposal_branch_loglikelihood+proposal_branch_prior-branch_loglikelihood-branch_prior)
  end

  if rand()<alpha
    updated = true
  else
    branch.feature = old_feature
    branch.value = old_value
    tree_adjust!(branch,x,residual,branch_indices)
    updated = false #actually I think here should also set updated = true, since now the leaves values are all zero, thus we need to sample again
  end
  updated
end

function swap_decision_rule!(bart_state::BartState,tree::BartTree,x::Matrix{Float64},residual::Vector{Float64},bartoptions::BartOptions)
  left_or_right(branch::DecisionBranch) = typeof(branch.left)==DecisionBranch?(typeof(branch.right)==DecisionBranch?rand(1:2):1):2
  grand_branch_nodes = grand_branches(tree)
  if length(grand_branch_nodes)==0
    return false
  end

  #pick a parent
  grand_branch = grand_branch_nodes[rand(1:length(grand_branch_nodes))]
  grand_branch_depth = depth(grand_branch)
  grand_branch_indices = train_data_indices(grand_branch)
  type_left = typeof(grand_branch.left)
  type_right = typeof(grand_branch.right)
  swap_both = false

  if type_left==DecisionBranch && type_right==DecisionBranch
    if grand_branch.left.feature==grand_branch.right.feature && grand_branch.left.value==grand_branch.right.value
      swap_both = true
    else
      i = rand(1:2)
      swap_branch = i==1?grand_branch.left:grand_branch.right
    end
  else
      swap_branch = left_or_right(grand_branch)==1?grand_branch.left:grand_branch.right
  end

  grand_branch_loglikelihood = node_loglikelihood(grand_branch,bart_state)
  grand_branch_prior = log_tree_prior(grand_branch,grand_branch_depth,bartoptions)
  old_grand_branch_feature = grand_branch.feature
  old_grand_branch_value = grand_branch.value

  if swap_both==true
     old_branch_feature = grand_branch.left.feature
     old_branch_value = grand_branch.left.value

     grand_branch.feature = old_branch_feature
     grand_branch.value = old_branch_value
     grand_branch.left.feature = grand_branch.right.feature = old_grand_branch_feature
     grand_branch.left.value = grand_branch.right.value = old_grand_branch_value
   else
     old_branch_feature = swap_branch.feature
     old_branch_value = swap_branch.value

     grand_branch.feature = old_branch_feature
     grand_branch.value = old_branch_value
     swap_branch.feature = old_grand_branch_feature
     swap_branch.value = old_grand_branch_value
   end

   valid_split = tree_adjust!(grand_branch,x,residual,grand_branch_indices)
   if !valid_split
     grand_branch.feature = old_grand_branch_feature
     grand_branch.value = old_grand_branch_value
     if swap_both
       grand_branch.left.feature = grand_branch.right.feature = old_branch_feature
       grand_branch.left.value = grand_branch.right.value = old_branch_value
     else
       swap_branch.feature = old_branch_feature
       swap_branch.value = old_branch_value
     end
     tree_adjust!(grand_branch,x,residual,grand_branch_indices)
     updated = false
     return updated
   end
   proposal_grand_branch_loglikelihood = node_loglikelihood(grand_branch,bart_state)
   proposal_grand_branch_prior = log_tree_prior(grand_branch,grand_branch_depth,bartoptions)

   #calculate alpha
   if proposal_grand_branch_loglikelihood==Inf
     alpha = 0.0
   else
     alpha = proposal_grand_branch_prior==Inf?0.0:exp(proposal_grand_branch_loglikelihood+proposal_grand_branch_prior-grand_branch_loglikelihood-grand_branch_prior)
   end

   if rand()<alpha
     updated = true
   else
     grand_branch.feature = old_grand_branch_feature
     grand_branch.value = old_grand_branch_value
     if swap_both
       grand_branch.left.feature = grand_branch.right.feature = old_branch_feature
       grand_branch.left.value = grand_branch.right.value = old_branch_value
     else
       swap_branch.feature = old_branch_feature
       swap_branch.value = old_branch_value
     end
     tree_adjust!(grand_branch,x,residual,grand_branch_indices)
     updated = false #actually I think here should also set updated = true, since now the leaves values are all zero, thus we need to sample again
   end
   updated
end

function update_leaf_values!(tree::BartTree,parameters::BartStateParameters)
  for leaf = leaves(tree) #need to write a function to get all the leaves
    update_leaf_values!(leaf,parameters);
  end
end

function update_leaf_values!(leaf::BartLeaf,parameters::BartStateParameters)
  n = length(leaf.leaf_data_indices);
  post_mean = n*leaf.residual_mean*parameters.sigma_mu^2/(n*parameters.sigma_mu^2+parameters.sigma^2);
  post_sigma = sqrt(parameters.sigma^2*parameters.sigma_mu^2/(n*parameters.sigma_mu^2+parameters.sigma^2));
  leaf.value = post_mean + post_sigma*randn();
end

function update_sigma!(bart_state::BartState,residuals::Vector{Float64})
  alpha = (bart_state.parameters.nu+length(residuals))/2;
  beta = (bart_state.parameters.nu*bart_state.parameters.lambda+sum(residuals.^2))/2
  x = Gamma(alpha,1/beta)  #using package Distributions
  bart_state.parameters.sigma = sqrt(1/rand(x,1)[1])
end

function initialize_bart_state(x::Matrix{Float64},y_normalized::Vector{Float64},bartoptions::BartOptions)
  #number_observations = size(x,1)
  number_observations = size(x,2)
  #initialize a bayesian additive regression tree
  trees = BartTree[]
  #here, assume we estimate y_normalized by its mean, thus estimates of each single tree should be
  #mean(y_normalized)/bartoption.num_trees; then residual of this tree should be
  initial_residuals = y_normalized.-(bartoptions.num_trees-1)*mean(y_normalized)/bartoptions.num_trees

  for i in 1:bartoptions.num_trees
     push!(trees,BartTree(DecisionTree(BartLeaf(0.0,initial_residuals,collect(1:number_observations)))))
  end

  sigma = std(y_normalized);
  sigma_mu = 0.5/(sqrt(bartoptions.num_trees)*bartoptions.k);
  nu = 3.0;
  q = 0.90;
  lambda = sigma^2*quantile(Chisq(nu),0.1)/nu; #using package Distributions
  bartparameters= BartStateParameters(sigma,sigma_mu,nu,lambda);
  bart_state = BartState(trees,bartparameters);
  #given all the tree are one single node, update the leaf values
  for tree = bart_state.trees
      update_leaf_values!(tree,bart_state.parameters)
  end
  yhat = predict(bart_state,x)

  for (tree in bart_state.trees)
      y_tree_hat = predict(tree,x)
      residuals = y_normalized-(yhat-y_tree_hat)
      tree.tree.root.residual_mean = mean(residuals)
      tree.tree.root.residual_sigma = sqrt(mean((residuals.-mean(residuals)).^2))
      update_leaf_values!(tree,bart_state.parameters)
      #after we update one tree, we update estimate of y
      yhat+=predict(tree,x)-y_tree_hat
  end
  bart_state
end

function bart_state_to_additivetree(bart_state::BartState)
  trees = [BartTree(DecisionTree(bart_node_regression_node(tree.tree.root))) for tree in bart_state.trees]
  BartAdditiveTree(trees)
end

function bart_node_regression_node(branch::DecisionBranch)
  DecisionBranch(branch.feature,
                 branch.value,
                 bart_node_regression_node(branch.left),
                 bart_node_regression_node(branch.right))
end

function bart_node_regression_node(leaf::BartLeaf)
  RegressionLeaf(leaf.value)
end

function normalize(response::Vector{Float64},y_min::Float64,y_max::Float64)
   k = 1/(y_max-y_min) #whatif y_max = y_min
   b = -0.5*(y_max+y_min)/(y_max-y_min)
   y_normalized = k*response+b;
   y_normalized
end

function denormalize(predict::Vector{Float64},y_min::Float64,y_max::Float64)
  y_denormalized = predict.*(y_max-y_min)+0.5*(y_max+y_min)
  y_denormalized
end

function StatsBase.fit(x::Vector{Float64},y::Vector{Float64},bartoptions::BartOptions)
    x = reshape(x,length(x),1)
    fit(x,y,bartoptions);
end

function StatsBase.fit(x::Matrix{Float64},y::Vector{Float64},bartoptions::BartOptions)
  println("Running BART with numeric y\n")
  println("Number of trees: ",bartoptions.num_trees)
  println("Prior:")
  println("     k: ",bartoptions.k)
  y_min = minimum(y)
  y_max = maximum(y)
  y_normalized = normalize(y,y_min,y_max)
  number_observations = length(y);
  number_predictors = size(x,2);
  x = x'

  bart_state = initialize_bart_state(x,y_normalized,bartoptions)
  println("     degrees of freedom in sigma prior: ",bart_state.parameters.nu)
  println("     quantile in sigma prior: 0.9")
  println("     power and base for tree prior: ",bartoptions.alpha," ",bartoptions.beta)
  println("     use quantile for rule cut points ",0)
  println("data: ")
  println("     number of training observations: ",number_observations)
  println("     number of explanatory variables: ",number_predictors)

  bart_additive_trees = Array(BartAdditiveTree,0)
  y_hat = predict(bart_state,x)
  println("\n")
  println("Running mcmc loop:")

  @time for i in range(1,bartoptions.num_draws+bartoptions.num_burn_in)
           if i % 100 ==0
            println("iteration: ",i," (of ",bartoptions.num_draws+bartoptions.num_burn_in,")")
           end
           updates = 0
           for  j = 1:bartoptions.num_trees
              y_tree_hat = predict(bart_state.trees[j],x)
              residual= y_normalized - (y_hat-y_tree_hat)
              updated = update_tree!(bart_state,bart_state.trees[j],x,residual,bartoptions)
              updates+=updated?1:0
              y_hat += predict(bart_state.trees[j],x)-y_tree_hat
           end
           update_sigma!(bart_state,y_normalized-y_hat)
           #println("there is",updates, "in this iteration")
           if i>bartoptions.num_burn_in
              if i % bartoptions.num_thinning==0
                bart_additive_tree = bart_state_to_additivetree(bart_state)
                push!(bart_additive_trees,bart_additive_tree)
              end
           end
        end
   Bart(bart_additive_trees,y_min,y_max,bartoptions)
end

function StatsBase.predict(tree::BartTree,sample::Vector{Float64}) #import package StatsBase
  node = tree.tree.root
  while typeof(node)==DecisionBranch
    if sample[node.feature]<=node.value
      node = node.left
    else
      node = node.right
    end
  end
  node.value
end

function StatsBase.predict(bart_state::BartState,sample::Vector{Float64})
  Float64(sum([predict(tree,sample) for tree in bart_state.trees]))
end

function StatsBase.predict(bart_state::BartState,x::Matrix{Float64})
  [Float64(predict(bart_state,vec(x[:,i]))) for i in range(1,size(x,2))]
end

function StatsBase.predict(tree::BartTree,x::Matrix{Float64})
   [Float64(predict(tree,vec(x[:,i]))) for i in range(1,size(x,2))]
end

function StatsBase.predict(bart_additive_tree::BartAdditiveTree,x::Matrix{Float64})
  y_predict = zeros(size(x,2))
  for i in range(1,size(x,2))
    y_predict[i] = Float64(sum([predict(tree,vec(x[:,i])) for tree in bart_additive_tree.trees]))
  end
  y_predict
end

function StatsBase.predict(bart::Bart,x::Vector{Float64},confidence_interval::Bool)
   x = reshape(x,length(x),1)
   StatsBase.predict(bart,x,confidence_interval)
end

function StatsBase.predict(bart::Bart,x::Matrix{Float64},confidence_interval::Bool)
   x = x'
   y_predict = zeros(size(x,2),length(bart.bart_additive_trees))
   if confidence_interval
      y_CI = zeros(size(x,2),2)
      for i in range(1,length(bart.bart_additive_trees))
      y_predict[:,i] = predict(bart.bart_additive_trees[i],x)
      y_predict[:,i] = denormalize(y_predict[:,i],bart.y_min,bart.y_max)
      end

      for i in 1:size(x,2)
        y_CI[i,:] = quantile(vec(y_predict[i,:]),[0.025,0.975])
      end
      #point estimate
      y_hat = vec(mean(y_predict,2))
      y_hat,y_CI
   else
      for i in range(1,length(bart.bart_additive_trees))
        y_predict[:,i] = predict(bart.bart_additive_trees[i],x)
        y_predict[:,i] = denormalize(y_predict[:,i],bart.y_min,bart.y_max)
      end
      #point estimate
      y_hat = vec(mean(y_predict,2))
      y_hat
   end
end
