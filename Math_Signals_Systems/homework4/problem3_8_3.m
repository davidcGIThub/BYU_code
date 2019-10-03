%Problem 3.8-3
x = [2,2.5,3,5,9];
y = [-4.2,-5,2,1,24.3];

scatter(x,y,'r');
hold on;

%determine the best least squares line
A = [x' , [1,1,1,1,1]']
c = (A'*A)\(A')*(y')
a = c(1);
b= c(2);
y_ls = x*a + b;
plot(x,y_ls,'b');
%create weighted least square
W = diag([10,1,1,1,10]);
c_w = (A'*W*A)\(A')*W*(y'); 
a_w = c_w(1);
b_w = c_w(2);
y_w = x*a_w + b_w;
plot(x,y_w);
legend("data points", "Least-Squares", "Weighted-Least-Squares");