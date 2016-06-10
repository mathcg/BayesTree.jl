# BayesTree
Code For Bayesian Additive Regression Trees

API Introduction
================
```julia   
   #generate the data
   x = rand(100)
   f = sin(pi*x)  
   y = f+randn(100)
   #set up the parameters for bart
   bartoptions = bart_options()
   #fit the bart model
   bart_1 = fit(x,y,bartoptions)
   #generate a test model and see the prediction performance
   test_data = rand(100);
   test_data_true = sin(pi*test_data);
   test_data_predict = predict(bart_1,test_data)
   println("The MSE is ",mean((test_data_true.-test_data_predict).^2))
```
   

