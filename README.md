# BayesTree

[![Build Status](https://travis-ci.org/mathcg/BayesTree.jl.svg?branch=master)](https://travis-ci.org/mathcg/BayesTree.jl)
=======
Code For Bayesian Additive Regression Trees

Prediction Example
================
```julia   
#generate the data
x = rand(100,5); f = zeros(100);
for i in 1:size(x,1)
   f[i] = 10*sin(pi*x[i,1]*x[i,2])+20*(x[i,3]-0.5)^2+10*x[i,4]+5*x[i,5]
end
y = f+randn(100);
bartoptions = bart_options();#set up the parameters for bart
bart_1 = fit(x,y,bartoptions); #fit the bart model
#evaluate the prediction performance
y_predict = predict(bart_1,x);
println("The MSE is ",mean((y_predict.-f).^2))
println("The correlation is ",cor(y_predict,f))
```
   
Model Selection Example
======================
```julia
x = rand(1000,10); f = zeros(1000);
for i in 1:size(x,1)
  f[i] = 10*sin(pi*x[i,1]*x[i,2])+20*(x[i,3]-0.5)^2+10*x[i,4]+5*x[i,5]
end
y = f+randn(1000);
bartoptions = bart_options();
2
#Here, 5 means that we set m, the number of additive trees being 5.
count_variables = model_selection(x,y,bartoptions,5);
count_variables
```
