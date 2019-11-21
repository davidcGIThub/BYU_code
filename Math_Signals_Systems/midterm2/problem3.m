clear;
clc;

t = -10:.1:10;
x = exp(0.1.*t).*sin(t);
n = -10:10;
len = size(n,2);
Zn = -10:10;
for i = 1:len
    Zn(i) = integral(n(i))
end

plot(t,x,n,Zn);
legend('x(t)','Zn')

function value = integral(N)
    dx = 0.01;
    time = N-0.5:dx:N+0.5;
    f = exp(0.1.*time).*sin(time);
    value = sum(f*dx);
end