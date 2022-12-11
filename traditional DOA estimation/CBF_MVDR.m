clc;
clear all;
close all;
M=10;              % elemeent number
theta0=-30;           % income azimuth
theta1=100;          
lamda=2.0;        % wavelength
d=lamda/2;        % elemeent interval
c=1500;          % speed
f0=c/lamda;       % frequency
snap=500;       % snapshot

% ====== make signal ====== %
p=2;                  % source number 
fs=4e+3;         % sampling rate
a=[0:1:M-1]';       % array vector
t=[0:snap-1]/fs;   % time

snr1=10;snr2=10;    % SNR
s0=10^(snr1/20)*sin(2*pi*f0*t+randn(1,snap));
s1=10^(snr2/20)*sin(2*pi*f0*t+randn(1,snap));
% ====== arrar vector
a_theta0=exp(j*pi*2*d/lamda*a*sin(theta0*pi/180));
a_theta1=exp(j*pi*2*d/lamda*a*sin(theta1*pi/180));

X=a_theta0*s0+a_theta1(:,1)*s1+(randn(M,length(t))+j*randn(M,length(t)))/sqrt(2);
Rx=X*X'/length(t);
R=inv(Rx);%inverse matrix
% ===== CBF
tic
theta=-90:1:90;
for ii=1:length(theta)
    a_theta=exp(j*pi*2*d/lamda*a*sin(theta(ii)*pi/180));
    Pdbf(ii)=sum((abs(a_theta'*X).^2));
end
Pdbf=Pdbf/max(Pdbf);

%  ====MVDR
pos=d*(0:M-1)';
wavenum=2*pi*f0/c;
theta=-90:1:90;
Y=zeros(1,length(theta));
for k=1:length(theta)
    A=exp(j*wavenum*pos*sin(theta(k)*pi/180));
    Y(k)=1/(A'*R*A);
end
Y=abs(Y)/max(abs(Y));

figure;plot(theta,Pdbf,'k-');grid on;hold on;%xlabel('方位角^o');ylabel('归一化谱值');title('CBF');
       plot(theta,Y,'k-');grid on;
       xlabel('Azimuth^o');ylabel('Amplitude');legend('CBF','MVDR');
