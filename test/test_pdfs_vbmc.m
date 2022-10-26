function test_pdfs_vbmc()
%TEST_PDFS_VBMC Test pdfs introduced in the VBMC package.

lb = [-1.1,-4.1];
ub = [3.2,-2.8];
a = [-1,-4];
b = [3,-3];
n = 1e6;

tolerr = 1e-3;      % Error tolerance on normalization constant
tolrmse = 0.05;     % Error tolerance on histogram vs pdf

%% Test multivariate uniform box distribution
pdf1 = @(x) munifboxpdf(x,a(1),b(1));
pdf2 = @(x) munifboxpdf(x,a,b);
pdf1log = @(x) exp(munifboxlogpdf(x,a(1),b(1)));
pdf2log = @(x) exp(munifboxlogpdf(x,a,b));
pdfrnd = @(n) munifboxrnd(a,b,n);
name = 'munifbox';

test_pdf1_normalization(pdf1,lb,ub,tolerr,name);
test_pdf2_normalization(pdf2,lb,ub,tolerr,name);
test_pdf1_normalization(pdf1log,lb,ub,tolerr,name);
test_pdf2_normalization(pdf2log,lb,ub,tolerr,name);
test_rnd(pdfrnd,pdf1,a,b,n,tolrmse,name);

%% Test multivariate trapezoidal distribution
u = [-0.5,-3.8];
v = [1.5,-3.4];
pdf1 = @(x) mtrapezpdf(x,a(1),u(1),v(1),b(1));
pdf2 = @(x) mtrapezpdf(x,a,u,v,b);
pdf1log = @(x) exp(mtrapezlogpdf(x,a(1),u(1),v(1),b(1)));
pdf2log = @(x) exp(mtrapezlogpdf(x,a,u,v,b));
pdfrnd = @(n) mtrapezrnd(a,u,v,b,n);
name = 'mtrapez';
test_pdf1_normalization(pdf1,lb,ub,tolerr,name);
test_pdf2_normalization(pdf2,lb,ub,tolerr,name);
test_pdf1_normalization(pdf1log,lb,ub,tolerr,name);
test_pdf2_normalization(pdf2log,lb,ub,tolerr,name);
test_rnd(pdfrnd,pdf1,a,b,n,tolrmse,name);

%% Test multivariate spline trapezoidal distribution
pdf1 = @(x) msplinetrapezpdf(x,a(1),u(1),v(1),b(1));
pdf2 = @(x) msplinetrapezpdf(x,a,u,v,b);
pdf1log = @(x) exp(msplinetrapezlogpdf(x,a(1),u(1),v(1),b(1)));
pdf2log = @(x) exp(msplinetrapezlogpdf(x,a,u,v,b));
pdfrnd = @(n) msplinetrapezrnd(a,u,v,b,n);
name = 'msplinetrapez';
test_pdf1_normalization(pdf1,lb,ub,tolerr,name);
test_pdf2_normalization(pdf2,lb,ub,tolerr,name);
test_pdf1_normalization(pdf1log,lb,ub,tolerr,name);
test_pdf2_normalization(pdf2log,lb,ub,tolerr,name);
test_rnd(pdfrnd,pdf1,a,b,n,tolrmse,name);

%% Test multivariate smoothbox distribution
sigma = [0.7,0.45];
pdf1 = @(x) msmoothboxpdf(x,a(1),b(1),sigma(1));
pdf2 = @(x) msmoothboxpdf(x,a,b,sigma);
pdf1log = @(x) exp(msmoothboxlogpdf(x,a(1),b(1),sigma(1)));
pdf2log = @(x) exp(msmoothboxlogpdf(x,a,b,sigma));
pdfrnd = @(n) msmoothboxrnd(a,b,sigma,n);
name = 'msmoothbox';
lb = [-5,-7];
ub = [5,0];
test_pdf1_normalization(pdf1,lb,ub,tolerr,name);
test_pdf2_normalization(pdf2,lb,ub,tolerr,name);
test_pdf1_normalization(pdf1log,lb,ub,tolerr,name);
test_pdf2_normalization(pdf2log,lb,ub,tolerr,name);
test_rnd(pdfrnd,pdf1,a,b,n,tolrmse,name);

%close all;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function test_pdf1_normalization(pdf1,lb,ub,tol,name)
%TEST_PDF1_NORMALIZATION Test normalization of univariate pdf.

% Check 1D integral
y = integral(@(x) pdf1(x), lb(1), ub(1), 'ArrayValued', true);
fprintf('%s: 1D integral: %.6f\n', name, y);
assert(abs(y - 1) < tol, ['Test error: univariate ' name ' does not integrate to 1.']);

end

function test_pdf2_normalization(pdf2,lb,ub,tol,name)
%TEST_PDF2_NORMALIZATION Test normalization of bivariate pdf.

% Check 2D integral
y = integral2(@(x1,x2) reshape(pdf2([x1(:),x2(:)]),size(x1)), ...
    lb(1), ub(1), lb(2), ub(2));
fprintf('%s: 2D integral: %.6f\n', name, y);
assert(abs(y - 1) < tol, ['Test error: bivariate ' name ' does not integrate to 1.']);

end

function test_rnd(pdfrnd,pdf1,a,b,n,tol,name)
%TEST_RND Test random sample generation (histogram vs pdf).

r = pdfrnd(n);
h = histogram(r(:,1),100,'BinLimits',[a(1),b(1)],'Normalization','pdf');
x = 0.5*(h.BinEdges(1:end-1) + h.BinEdges(2:end))';
y = pdf1(x)';
rmse = sqrt(sum(((y - h.Values)).^2*h.BinWidth));
fprintf('%s: histogram rmse: %.6f\n', name, rmse);
assert(rmse < tol, ['Test error: generated histogram does not match ' name ' pdf.']);

end