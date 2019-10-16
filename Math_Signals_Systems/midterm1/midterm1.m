% % Midterm 1 
% clear;
% clc;
% 
% %problem 2 a
% A = [1 1; 2 0; -1 1];
% temp = A'*A;
% temp = temp\(A');
% P = A*temp;
% c = [-1 ; 1 ; 1];
% x = P*c;
% clear;
% clc
% 
% %problem 2 d
% x = [1 2; -1 1];
% y = [-2 1 ; 2 2];
% temp = x*y'
% clear;
% clc;
% 
% %problem 3
% % a
% data = load('mid1_p3_data.mat').data
% t = data(:,1);
% y = data(:,2);
% id = data(:,3);
% temp = zeros(size(t,1),1) +1;
% A = [temp , t , t.^2 , t.^3];
% a = pinv(A)*y;
% poly_fit = a(1) + a(2)*t + a(3)*t.^2 + a(4)*t.^3;
% scatter(t,y);
% hold on;
% plot(t,poly_fit);
% 
% %part b
% p = A'*y;
% norm = sqrt(y'*y - a'*p);
% 
% % part c
% temp1 = (id-2)*(-1);
% temp2 = (id-1)*10;
% w = temp1 + temp2;
% W = diag(w);
% c = (A'*W*A)\(A')*W*y;
% poly_fit_weight = c(1) + c(2)*t + c(3)*t.^2 + c(4)*t.^3;
% plot(t, poly_fit_weight);
% legend("data points", "polynomial fit", "weighted polynomial fit");
% clear;
% clc;
% 
% %part d
% norm_w = sqrt(y'*y - c'*p)

% %problem 4
% %part a
% B = [1 0 ; 1 1 ; 0 1];
% Pr = B*pinv(B);
% %part b
% P_squared = Pr*Pr;
% %part c
% x = [1; -1; 2];
% range = Pr*x
% nullspace = range - x;
%clear;
%clc

%problem 5
%part a
t = linspace(0,1,1001)';
p1 = t*0 + 1;
p2 = p1;
p2(500:750) = -1;
p3 = p1;
p3(500:1001) = -1;
figure(1);
plot(t,p1,t,p2,t,p3);
legend("p1" , "p2" , "p3");
axis([0 1 -1.5 1.5]);
hold off;
e1 = p1;
dt = t(2) - t(1);
q1 = e1 ./ sqrt(sum(e1.*e1*dt));
e2 = p2 - sum(p2.*q1*dt)*q1;
q2 = e2 ./ sqrt(sum(e2.*e2*dt));
e3 = p3 - sum(p3.*q1*dt)*q1 - sum(p3.*q2*dt)*q2;
q3 = e3 ./ sqrt(sum(e3.*e3*dt));

%part b
figure(2);
subplot(3,1,1);
plot(t,q1);
legend("q1");
subplot(3,1,2);
plot(t,q2);
legend("q2");
subplot(3,1,3);
plot(t,q3);
legend("q3");
axis([0 1 -2 1.5]);

%part c
x = 2*t;
A = [q1 q2 q3];
P = A*pinv(A);
x_hat = P*x;
figure(3);
plot(t,x_hat,t,x);
legend("x_hat","x");

%part d

