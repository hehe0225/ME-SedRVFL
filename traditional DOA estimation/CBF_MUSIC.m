
clear;close all;clc; 

num=16;% elemeent number
f1=500;% frequency 1
f2=500;% frequency 2
a1=-45;% azimuth 1 
a2=40;% azimuth 2 
c=1500;% speed 
lmda=c/f1;% wavelength 
d=lmda/2;% elemeent interval
thita=-90:1:90; 
SNR1=-10;% SNR 
SNR2=-10;% SNR
fs=400; 
Ts=[0:1/fs:1]; 
snapshot=length(Ts); 
s1=10^(SNR1/20)*sin(2*pi*f1*Ts);
s2=10^(SNR2/20)*sin(2*pi*f2*Ts);
athita1=exp(j*2*pi*d/lmda*sin(a1*pi/180)*[0:num-1]).';
athita2=exp(j*2*pi*d/lmda*sin(a2*pi/180)*[0:num-1]).';
x=athita1*s1+athita2*s2+(randn(num,length(Ts))+j*randn(num,length(Ts)))/sqrt(2);

%==============MUSIC===============% 
R=x*x'/snapshot;
[V,D]=eig(R); 
test = V;
P=V(:,1:14)*V(:,1:14)'; 
for i=1:length(thita) 
    A=exp(j*2*pi*d/lmda*sin(thita(i)*pi/180)*[0:num-1]).';
    PF(i)=sum(abs(A'*x).^2);
    test = sum(abs(A'*x).^2);
    BF(i)=A'*R*A;
    S(i)=1/(A'*P*A); 
end 

figure; 
plot(thita,10*log10(abs(S)/(max(abs(S))))); 
grid on;hold on; 
plot(thita,10*log10(abs(PF)/(max(abs(PF)))),'r--');
title('MUSIC') 



