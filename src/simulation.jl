x = rand(200,5); f = zeros(200);
for i in 1:size(x,1)
  f[i] = 10*sin(pi*x[i,1]*x[i,2])+20*(x[i,3]-0.5)^2+10*x[i,4]+5*x[i,5]
end
y = f+randn(200);
bartoptions = bart_options()
@time bart_1 = fit(x,y,bartoptions)
@time y_predict = predict(bart_1,x);
mean((y_predict-f).^2)
cor(y_predict,f)


y_min = minimum(y)
y_max = maximum(y)
y_normalized = normalize(y,y_min,y_max)
number_observations = length(y);
number_predictors = size(x,2);
x = x'

bart_state = initialize_bart_state(x,y_normalized,bartoptions)

udpates = 0
for i = 1:300
@time for  j = 1:bartoptions.num_trees
   y_tree_hat = predict(bart_state.trees[j],x)
   residual= y_normalized - (y_hat-y_tree_hat)
   updated = update_tree!(bart_state,bart_state.trees[j],x,residual,bartoptions)
   updates+=updated?1:0
   y_hat += predict(bart_state.trees[j],x)-y_tree_hat
end
end
