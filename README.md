# BayesTree
Code For Bayesian Additive Regression Trees

API Introduction
================
   x = rand(100)
   f = sin(pi*x)  
   y = f+randn(100)
   bartoptions = bart_options()
   bart_1 = fit(x,y,bartoptions)
   test_data = randn(100);
   test_data_true = sin(pi*test_data);
   test_data_predict = predict(bart_1,test_data)
   println("The Mse is ",mean((test_data_true.-test_data_predict).^2))
   

