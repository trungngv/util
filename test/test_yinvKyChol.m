function test_yinvKyChol()
%TEST_YINVKYCHOL test_yinvKyChol()
% See also
%   yinvKyChol

N = 100;
A = rand(N, N);
K = A'*A;
y = rand(N, 1);
expected = y'*(K\y);
result = yinvKyChol(y, jit_chol(K));
assert(abs(expected - result) <= 1e-10, ...
    ['test yinvKyChol() failed with diff = ' num2str(abs(expected - result))]);
disp('test yinvKyChol() passed')


