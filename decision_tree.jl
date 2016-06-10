#concrete type of trees
abstract Decision
typealias DecisionNode Node{Decision}
typealias DecisionLeaf Leaf{Decision}

type DecisionTree <: Tree{Decision}
   root::DecisionNode
end

type DecisionBranch <: Branch{Decision}
    feature::Int
    value::Float64
    left::DecisionNode
    right::DecisionNode
end

type RegressionLeaf <: DecisionLeaf
  value::Float64
end

type ClassificationLeaf <: DecisionLeaf
  prob::Vector{Float64}
end
