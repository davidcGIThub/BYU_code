%problem 3.9-12

%covariance method

A = [1,1;
    2,1;
    3,2;
    5,3;
    8,5;
    13,8;];

R = A'*A;

% autocorrelation method

A = [1,0;
     1,1;
     2,1;
     3,2;
     5,3;
     8,5;
     13,8;
     0,13];
 
 R_a = A'*A;
 
 %part b
 %covariance
 
A = [1,1;
    2,1;
    3,2;
    5,3;
    8,5]
 
 d = [2 ; 3 ; 5 ; 8 ; 13 ];
 
 h = (A'*A)\A'*d
 
 e = d - A*h
 
%autocorrelation
 
A = [1,0;
     1,1;
     2,1;
     3,2;
     5,3;
     8,5;
     13,8;
     0,13];
 
  d = [1;  2 ; 3 ; 5 ; 8 ; 13 ; 0 ; 0];
  
   
 h = (A'*A)\A'*d;
 
 e = d - A*h