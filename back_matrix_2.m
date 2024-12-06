function [profile1,profile2]=back_matrix_2(ifft_matrix1,ifft_matrix2,Rtaur,Rs,delta_r)
%%%%改进的像拼接算法：逆推舍弃法
%输入：ifft_matrix，IFFT后的矩阵
% Rtaur：单脉冲距离分辨率 Rtaur=c*tao/2;
% Rs:采样距离单元 Rs=c*Ts/2
% delta_r:合成后的距离分辨单元 delta_r=c/(2N*delta_f)
[N,M]=size(ifft_matrix1);
p=zeros(1,M);  %每列抽取的起始点
q=zeros(1,M);  %每列抽取的结束点
w=zeros(1,M);  %每列抽取的距离单元数
start=zeros(1,M);  %本方法每列开始的起始点
profile_temp1=zeros(1,2*N);
sum_2 = sqrt(abs(ifft_matrix1).^2+abs(ifft_matrix2).^2);
[ay,ax]=find(max(max(abs(sum_2)))==abs(sum_2));
ay=ay(round((1+length(ay))/2));
ax=ax(round((1+length(ax))/2));
num_Rtaur=fix(Rtaur/Rs);
num_Rs=fix(Rs/delta_r);
sum_max=zeros(num_Rs,num_Rtaur);
matrix=repmat(sum_2,2,1);
if (ay<num_Rs)
    ay=ay+N;
end
for p=0:num_Rtaur-1
    for q=0:num_Rs-1
        sum_max(q+1,p+1)=sum(sum(abs(matrix(ay-q:ay-q+num_Rs-1,ax-p:ax-p+num_Rtaur-1))));   
    end
end
[by,bx]=find(max(max(abs(sum_max)))==abs(sum_max));
x=ax-(bx-1);              %目标区域左下角的横坐标
y=mod(ay-(by-1),N);       %目标区域左下角的纵坐标
for k=1:M
        p(k)=fix((k-1)*(Rs/delta_r))+1;
        q(k)=fix(k*(Rs/delta_r));
        w(k)=q(k)-p(k)+1;         %q=p+w-1   w=q-p+1舍弃法选点
end
start_y = mod((y - sum(w(1:(x-1)))),N); % 起始点
start_x = ax - (x-1);
for  k=1:M-start_x+1
        start(k)=mod(start_y+p(k)-1,N)+1;
        
        profile_temp1(1:N)=ifft_matrix1(:,k+start_x-1);
        profile_temp1(N+1:2*N)=profile_temp1(1:N);   %周期延拓     
        profile1(p(k):q(k))=profile_temp1(start(k):start(k)+w(k)-1);
        
        profile_temp2(1:N)=ifft_matrix2(:,k+start_x-1);
        profile_temp2(N+1:2*N)=profile_temp2(1:N);   %周期延拓     
        profile2(p(k):q(k))=profile_temp2(start(k):start(k)+w(k)-1);
        
end
