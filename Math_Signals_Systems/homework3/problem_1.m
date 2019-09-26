%problem 1
t = linspace(-1,1,10000);
p = zeros(10000, 5);
q = p;
e = p;
dt = t(2) - t(1);
for kk = 1:5
    p(:,kk) = t.^(kk-1);
end
e(:,1) = p(:,1);
q(:,1) = e(:,1) ./ sqrt(sum(e(:,1).*e(:,1).*dt));

for ii = 2:5
    e(:,ii) = p(:,ii);
    for jj = 1:ii-1
        e(:,ii) = e(:,ii) - sum(p(:,ii).*q(:,jj).*dt) * q(:,jj);
    end
    q(:,ii) = e(:,ii) ./ sqrt(sum(e(:,ii).*e(:,ii).*dt));
end
figure(1);
plot(t,q(:,1));
hold on;
plot(t,q(:,2));
plot(t,q(:,3));
plot(t,q(:,4));
plot(t,q(:,5));
legend('q0','q1','q2','q3','q4');
hold off;

% problem 2
f = transpose(exp(-t));

R = zeros(5,5);
for ii = 1:5
    for jj = 1:5
        R(ii,jj) = sum(q(:,ii) .* q(:,jj) * dt);
    end
end

P = zeros(5,1);
for ii = 1:5
    P(ii,1) = sum(f .* q(:,ii) * dt);
end

x_hat = q*inv(R)*P;

figure(2)
plot(t,x_hat);
hold on;
plot(t,f);
legend('approximation', 'exp(-t)');
