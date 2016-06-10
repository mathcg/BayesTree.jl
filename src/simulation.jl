x = rand(200,5); f = zeros(200);
for i in 1:size(x,1)
  f[i] = sin(pi*x[i,1]*x[i,2])+20*(x[i,3]-0.5)^2+10*x[i,4]+5*x[i,5]
end
y = f+randn(200);
bartoptions = bart_options()
@time bart_1 = fit(x,y,bartoptions)
@time y_predict = predict(bart_1,x);
mean((y_predict-f).^2)
cor(y_predict,f)
