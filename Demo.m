% Demo for RPCA_tl1-2
%% syntic data
% randn('state',2009);
% rand('state',2009);
clear;clc;
addpath PROPACK
pr=0.1;
ps=0.1;%pe
m=400;
r=round(pr*m);              %Rank of the groundtruth matrix
EL0=round(m*m*ps);          %Number of missing values

U=normrnd(0,1,m,r);V=normrnd(0,1,m,r);
A0=U*V';
E=zeros(m,m);
Ind = randperm(m*m);
t = max(abs(A0(:)));
E(Ind(1:EL0))=2*5*rand(1,EL0)-5 ;
D=A0+E;
fprintf('RPCA \n\n')
starttime = tic;
[A_L E_L ]=RPCA(D);
time_L=toc(starttime);
RelativeError_L=(sum(sum((A_L-A0).^2))).^0.5/(sum(sum(A0.^2))).^0.5;
rank_L=rank(A_L);
fprintf( 'Relative Error: %e \nRank of estimated matrix: %f \nRunning Time: %f \n', RelativeError_L, rank_L, time_L );
%% Background subtraction
X3D = load('Data\WaterSurface.mat');
[m,n,k] = size(X3D);
X = X3D.X;
[A_L E_L ]=RPCA(X);
rank(A_L);
figure;

for k =1:size(X,2)
   subplot(221); imshow(reshape(X(1:end,k),[128,160]),[]);title('input');
   subplot(222); imshow(reshape(A_L(1:end,k),[128,160]),[]);
   subplot(223); imshow(abs(reshape(E_L(1:end,k),[128,160])),[]);
   pause(.1);
end


