clc;
clear all
close all
% addpath('G:\华阴数据处理\华阴实验 资料\新数据处理1105\20200730huayin');
addpath('D:\');
filename2='24_14.dat';
% addpath('H:\烟台部分数据');
% addpath("G:\烟台实测渔船数据\echos\"); % Gzk
% filename2='72_73.dat';

prfnum = 128; % 一个CPI内的脉冲数
cpi_ks = 2000+(0+0+0+0)*5;   % 从第几个开始处理737+180
deltaf_mode = 0;  %  0123分别是6M、12M、20M、40M SPW: delta_f
M = fix(1600); % 2+0*5;      % 几个CPI，CPI总数
% M = 10;

fftnum=256;
fix_hb_num=8028; % 快时间采样点数
% sum1:两个通道各一帧数据
sum1 = fix_hb_num*2*prfnum+4; % *2是因为以前两个通道都是8028，现在揉成一个通道，所以*2; 
cpidata2 = zeros(prfnum, fix_hb_num);

c = 3e8;
fs = 48e6; % 采样频率 
Tr = 80e-6; % 脉冲重复间隔,周期，PRI
rs = c/(2*fs); % 采样单元
Rs = c/2/fs; % 3.125m

fid2 = fopen(filename2,'r'); % 打开文件filename以便以二进制读取形式进行访问，并返回等于或大于3的整数文件标识符
status = fseek(fid2, (2*sum1*(cpi_ks-1)), 'bof'); % 状态
% fseek(fid, offset, origin)
% fid是文件标识符，offset是偏移量，origin是偏移量的起始位置。
% 偏移量可以是正数或负数，表示从当前位置向前或向后移动多少个字节。
% 起始位置可以是以下三种之一：
% - 'bof'：从文件开头开始计算偏移量。
% - 'cof'：从当前位置开始计算偏移量。
% - 'eof'：从文件末尾开始计算偏移量。

ps = ftell(fid2); 
% 返回指定文件中位置指针的当前位置;
% 如果查询成功，则 position 是从 0 开始的整数，指示从文件开头到当前位置的字节数。
% 如果查询不成功，则 position 为 -1。
% if status == -1;
%         disp('ERROR: The end of the file!');
%         break;
% end
    
nn=1; count=0;
for mm = 1:M; % cpi_ks: cpi_ks+M-1
    mm
    % status = fseek(fid2,2*sum1*2*50,'cof');
    status = fseek(fid2, 2*sum1*2*5, 'cof'); % offset是偏移量 2*sum1*2*50
    ps0(mm) = ftell(fid2);   % ps0:每次处理完一帧完整的数据后，指针当前的位置
    y2 = fread(fid2, 2*sum1,'int16','s'); % 2*sum1:读取size()
     
    if(length(y2) < 2*sum1)
        disp('ERROR: The data length is not enough!!');
        break;
    end
     
    ps1(mm) = ftell(fid2);   % ps1:读取1帧数据后，指针所在位置，应该比ps0后移一帧数据*2，因为int16读取
    for kk=1:1:sum1*2
        if ((y2(kk)==32767) && (y2(kk+3)==32767) && (y2(kk+4)==4095) && (y2(kk+7)==4095)) % 这个for motivation?
            offset2(mm)=kk-1; % offset2 作用？
            break;
        end
    end
    status = fseek(fid2,-2*(2*sum1-offset2(mm)),'cof');
    ps2(mm) = ftell(fid2);    % ps2:找到帧头，指针向前移动至帧头位置
    y22=fread(fid2,sum1,'int16','s'); % 读出当前帧的数据

    ps3(mm) = ftell(fid2);     % ps3:读取当前帧完整数据后，指针位置，应该比ps2后移一帧数据*2
    status3 = fseek(fid2,-200,'cof');
    
    if status3 == -1;
        disp('ERROR: The end of the file!');
        break;
    end
    if(length(y22)<sum1)
        disp('ERROR: The data length is not enough!');
        break;
    end
    
    y51 = y22(6:2:end); % 1027584
    y52 = y22(5:2:end);

    %     if mm>(cpi_ks-1)
    if 1 % (mod(mm,5)==1)%mm>(M-1)%-cpi_ks1%
        yI_hb = reshape(y51, fix_hb_num, prfnum);
        yI_hb = yI_hb.'; % transpose
        yQ_hb = reshape(y52, fix_hb_num, prfnum);
        yQ_hb = yQ_hb.';

        y_control(nn,:) = yI_hb(1,4:2:30);
        % aa = y_control(nn,:);
        % disp(aa(7));
        % disp(aa(12));
        % transmit_mode = 1; % y_control(nn,7); % transmit_mode: 1R发；2L发
        
        transmit_mode = y_control(nn,7);
        disp("transmit_mode: ");
        disp(transmit_mode);
        
    % 1：指示距离；2:弹目径向速度；3：工作模式，123分别是搜索截获跟踪；4:数采起点；5：数采类型，0和和1和差；
    % 6：信号时宽，0-》PD，14us，1-》PD,5us，3-》PD，130ns，8-》SF，42us，9-》SF，24us，10-》SF,14us，11-》SF，5us，12-》SF，130ns
    % 7：发射形式: 0 1 2 5，分别是:不发;发L;发R;交替发；8：两和通道的补偿相位；9：带宽，1->24M，4->192M
    %10：方位和差补偿相位；11：俯仰和差补偿相位；12：方位角；13：俯仰角；14：跟踪距离
        
        y_RI = yI_hb(:,31:2:end);
        y_RQ = yQ_hb(:,31:2:end);
        y_LI = yI_hb(:,32:2:end);
        y_LQ = yQ_hb(:,32:2:end);
        
        LorR(mm) = y_RI(6,7);
        
        R = y_RI + j*y_RQ;
        L = y_LI + 1i*y_LQ;
%         R(:,1:15)=0;L(:,1:15)=0;
        
        %% 参数读取
        tao_mode = y_control(nn,6); %11
        switch tao_mode
            case 8
                tao = 42e-6;disp('42us');
            case 9
                tao = 24e-6;disp('24us');
            case 10
                tao = 14e-6;disp('14us');
            case 11
                tao = 5e-6;disp('5us'); % 脉宽
            case 12
                tao = 130e-9;disp('130ns');
        end
%         deltaf_mode = 1;%y_control(nn,8);
        switch deltaf_mode
            case 0
                deltaf = 6e6;disp('6M');  % delta-f:6M
            case 1
                deltaf = 12e6;disp('10M');
            case 2
                deltaf = 20e6;disp('20M');
            case 3
                deltaf = 40e6;disp('40M');
        end
        prfnum_mode = y_control(nn,10);
        switch prfnum_mode
            case 0
                prfnum = 128;disp('128点');
            case 1
                prfnum = 64;disp('64点');
            case 2
                prfnum = 32;disp('32点');
            case 3
                prfnum = 256;disp('256点');
        end
        B_mode = 1; % y_control(nn,9);
        switch B_mode
            case 1
                B = 24e6;disp('SF'); % 24M
            case 4
                B = 192e6;disp('PD');
        end
        
        delta_r = c/(2*prfnum*deltaf); % 距离分辨率
        
        %%%%%  ddc系数  %%%%%
        fid5=fopen('fir_ddci.coe','r');%
        datalen=20;
        width=16;
        fir_ddcQ = fdbin2dec(fid5, datalen, width);

        fid6=fopen('fir_ddcq.coe','r');%
        datalen=20;
        width=16;
        fir_ddcI = fdbin2dec(fid6, datalen, width);
        
        fir1 = fir_ddcI + fir_ddcQ;
%         figure;freqz(fir1)
        load('ddc_coe_xin'); % ddc数字下变频,coe_i1,coe_q1
        fir_ddcI = 2^14*coe_i1;
        fir_ddcQ = 2^14*coe_q1;
%         figure;freqz(fir2)
%                 figure
%                 freqz(fir1)
        
        %=============NCO====
        n=[1:length(y_RI)]; % 相当于range()
        nco=(-1).^n;
        
        for i = 1:prfnum
            mixRI(i,:)=y_RI(i,:).*nco;
            mixRQ(i,:)=y_RQ(i,:).*nco;
            
            ddcRi(i,:) = conv(mixRQ(i,:),fir_ddcI);
            ddcRq(i,:) = conv(mixRI(i,:),fir_ddcQ);
            
            mixLI(i,:) = y_LI(i,:).*nco;
            mixLQ(i,:) = y_LQ(i,:).*nco;
            
            ddcLi(i,:) = conv(mixLQ(i,:),fir_ddcI);
            ddcLq(i,:) = conv(mixLI(i,:),fir_ddcQ);
        end
        
        QQ=5;
        for i=1:1:prfnum
            ddcRI(i,:) = downsample(ddcRi(i,:), QQ); %抽取
            ddcRQ(i,:) = downsample(ddcRq(i,:), QQ);
            ddcLI(i,:) = downsample(ddcLi(i,:), QQ); %抽取
            ddcLQ(i,:) = downsample(ddcLq(i,:), QQ);
        end
        
        R_ddcout = ddcRI + j*ddcRQ; % R路接收
        L_ddcout = ddcLI + j*ddcLQ; % L路接收
%         figure;plot(abs(R_ddcout).','r');   hold on;plot(abs(L_ddcout).','b')
        
        
        %             fd=12.5e3/128*8;
%               figure;subplot(211),plot(real(R_ddcout.'))
%               subplot(212),plot(imag(R_ddcout.'))
%                 figure;subplot(211),plot(real(L_ddcout.'));
%                 subplot(212),plot(imag(L_ddcout.'))
        
%       接收幅度相位补偿
%       amp_bc = 1;%1.1885;
        coe = exp(j*y_control(nn,8)/57.3);
        L_ddcout = L_ddcout*coe;

        k = B/tao; % 调频斜率
        tp = (-tao/2 : 1/fs : tao/2-1/fs); % +tao/2 采样
        sig = exp(-j*pi*k .* tp.^2); % 线性调频脉冲信号
        scoe_l = fliplr((sig)) .* (hamming(length(tp))).'; % fliplr 航向量翻转，列向量不变 h(n)
        fcoe_l = fft(2^11 * fliplr((sig)) .* (hamming(length(sig))).', 4096); % h(f)
        scoe_r = fliplr((sig)) .* (hamming(length(tp))).'; % h(n)
        fcoe_r = fft(2^11 * fliplr(conj(sig)) .* (hamming(length(sig))).', 4096); % h(f)
        % 时域卷积--频域相乘，脉压：h(f).*s(f)
        
        % 给一个CPI中的每一个脉冲进行脉压
        for i = 1:prfnum 
%             R_pc_fft(i,:) = fft(R_ddcout(i,:),4096).*fcoe_r;     % R
%             R_pcoutq(i,:) = ifft(R_pc_fft(i,:),4096);
%             R_pcout_t(i,:) = conv(R_ddcout(i,:),scoe_r);
            
            R_pc_fft1(i,:) = fft (R_ddcout(i,:), 4096) .* fcoe_l; % RL--R收L发
            R_pcout1q(i,:) = ifft(R_pc_fft1(i,:), 4096);
%             R_pcout_t1(i,:) = conv(R_ddcout(i,:),scoe_l);
            
%             L_pc_fft(i,:) = fft(L_ddcout(i,:),4096).*fcoe_r;
%             L_pcoutq(i,:) = ifft(L_pc_fft(i,:),4096);
%             L_pcout_t(i,:) = conv(L_ddcout(i,:),scoe_r);
            
            L_pc_fft1(i,:) =  fft(L_ddcout(i,:) , 4096).*fcoe_l;
            L_pcout1q(i,:) = ifft(L_pc_fft1(i,:), 4096);
%             L_pcout_t1(i,:) = conv(L_ddcout(i,:),scoe_l);
            
        end
       
%         nn = nn+1;
        R_pcout1 = L_pcout1q(:,:); % 为啥L赋值给R？
        L_pcout1 = R_pcout1q(:,:);
        
        echo_R_temp(:,mm) = R_pcout1(1, 1:1500); % 脉压后一维距离像序列（R通道接收）
        echo_L_temp(:,mm) = L_pcout1(1, 1:1500); % 脉压后一维距离像序列（L通道接收）
%      figure(1);plot(abs(R_pcout1).','r');   hold on;plot(abs(L_pcout1).','b')
%     
%      figure;plot(rs*(1:size(L_pcout1,2)),abs(R_pcout1).');figure;plot(rs*(1:size(L_pcout1,2)),abs(L_pcout1.'));   
%         noise_L_power = L_pcout1(:,200:500);
% 幅度补偿
%         noise_L = abs(L_pcout1(:,200:500));
%         noise_R = abs(R_pcout1(:,200:500));
%         power_L_n = std(noise_L(:));
%         power_R_n = std(noise_R(:));
%         amp_bc = 1.5313;%power_L_n/power_R_n;
%         R_pcout1 = amp_bc*R_pcout1;
   %     figure;plot(abs(R_pcout1q).');   figure;plot(abs(L_pcout1).')
    %% 像拼接
%         
%         figure;imagesc(abs(R_pcout1));
%         figure;imagesc(abs(L_pcout1));
%         figure;plot(abs(L_pcout1(1,:)),'r');hold on;
%         plot(abs(R_pcout1(:,1)),'b');legend('LL','RL');xlabel('距离分辨单元'),ylabel('归一化幅度');
% axis tight;title('');

        % 如果 X 是向量，则 fft(X) 返回该向量的傅里叶变换。
        % 如果 X 是矩阵，则 fft(X) 将 X 的各列视为向量，并返回每列的傅里叶变换。
        R_pc_m = ifft(R_pcout1); % 对128*4096的每个列128进行FFT
        L_pc_m = ifft(L_pcout1);
        [N,samplenumber] = size(R_pc_m); % 脉冲数 采样数
%         figure;imagesc(abs(R_pc_m));figure;imagesc(abs(L_pc_m));
%         figure;plot(abs(R_pc_m));figure;plot(abs(L_pc_m));
%         figure;
%        for i=210:290
%            plot(abs(R_pc_m(:,i)));pause(0.3);
%        end
%         sum_2 = abs(R_pc_m)+abs(L_pc_m);
%         aa=max(max(sum_2));
%         [rows,cols] = find(abs(sum_2)==aa);
% %         cols=603;
%         cols_vec(mm)=cols;
%         freq_echo_L(:,mm) = L_pcout1(:,cols);
%         freq_echo_R(:,mm) = R_pcout1(:,cols);
%                 figure;
%         plot((abs(R_pc_m(:,cols))),'linewidth',1.5);axis tight;
%         hold on;
%         plot((abs(L_pc_m(:,cols))),'linewidth',1.5);
%         title('最大元素所在的一列');
%         legend('R','L');axis tight;
        Rtaur = c/2/B; % 步进频合成之前距离分辨率
        wtd = 0.3;
        thd = 0.5;
%         [R_profile,L_profile]=back_matrix_2(R_pc_m,L_pc_m,Rtaur,rs,delta_r);
%         R_pc_m = flipud(R_pc_m);
%         L_pc_m = flipud(L_pc_m);
        % [R_profile,L_profile]=big_abondon_2(R_pc_m,L_pc_m,rs,delta_r); % delta_r是步进频距离分辨率
        % [R_profile,L_profile]=abondon_2(R_pc_m,L_pc_m,Rs,delta_r,samplenumber,N); 
        [R_profile,L_profile]=back_matrix_2(R_pc_m,L_pc_m,Rtaur,Rs,delta_r); 
        % [R_profile, L_profile, row, col] = chouqu_2channel(R_pc_m, L_pc_m, fs, deltaf, wtd, thd);
        if y_control(mm,7) == 1
           title_name = 'L发';
        elseif y_control(mm,7) == 2
                 title_name = 'R发';
            elseif  y_control(mm,7) == 3
                 title_name = '45线发';
            end
%         figure;line1=plot(delta_r*(1:length(R_profile)),db(abs(R_profile)),'r');hold on;line2=plot(delta_r*(1:length(R_profile)),db(abs(L_profile)),'b');
%         xlim([1 2000]);
%         title(title_name);
%         legend('右旋接收','左旋接收');
%         figure;plot(delta_r*(1:length(R_profile))-2.375,(abs(R_profile)),'r');hold on;plot(delta_r*(1:length(R_profile))-2.375,(abs(L_profile)),'b');
%         figure;plot((1:length(R_profile)),(abs(R_profile)),'r');hold on;plot((1:length(R_profile)),(abs(L_profile)),'b');axis tight;
%         xlim([1 2000]);
%         title(title_name);
%         xlabel('距离/m'); ylabel('幅度');
%         legend('右旋接收','左旋接收');
%         [~,max_ind] = max(abs(R_profile)+abs(L_profile));
%         x_cut = max_ind-599:max_ind+2000;
%         x_cut = max_ind-511:max_ind+1536;
%         max_ind =7749;
%         x_cut = max_ind-63:max_ind+64;
% %         x_cut = max_ind-31:max_ind+32;
%       
%       截取的
        echo_R(:,mm) = R_profile(1:65520);%(x_cut); % 高分辨一维距离像序列（R通道接收） 
        echo_L(:,mm) = L_profile(1:65520);%(x_cut); % 高分辨一维距离像序列（L通道接收）
        
        
%         figure;plot((1:length(x_cut)),abs(echo_R(:,mm)),'r');hold on;plot((1:length(x_cut)),abs(echo_L(:,mm)),'b');
%         xlabel('距离分辨单元');ylabel('幅度');title('目标一维距离像');legend('RL极化','LL极化');
        nn = nn+1;
        temp(mm) = yI_hb(6,7);
    end
end

% figure;plot(abs(echo_R));
% figure;imagesc(abs(echo_R));
% xlabel('样本个数');ylabel('距离分辨单元');title('高分辨一维距离像序列（R通道接收）');
% figure;imagesc(abs(echo_L));
% xlabel('样本个数');ylabel('距离分辨单元');title('高分辨一维距离像序列（L通道接收）');
% % echo_R = echo_R(:,2:2:end);
% % echo_L = echo_L(:,2:2:end);  
% 
% figure;plot(abs(echo_R_temp));
% figure;imagesc(abs(echo_R_temp));
% xlabel('样本个数');ylabel('距离分辨单元');title('脉压后一维距离像序列（R通道接收）');
% figure;imagesc(abs(echo_L_temp));
% xlabel('样本个数');ylabel('距离分辨单元');title('脉压后一维距离像序列（L通道接收）');
% 
% kkkk=3;
% figure;hold on;
% plot(abs(echo_R(11100:11355, kkkk)),'r'); % 红色是右旋接收
% plot(abs(echo_L(11100:11355, kkkk)),'b'); % 蓝色是左旋接收
% legend('右通道接收','左通道接收','location','northeast');
% xlabel('高分辨一维距离像');ylabel('幅值');title('单幅高分辨一维距离像');
% hold off;

% figure;hold on;
R_max = max(abs(echo_R));
L_max = max(abs(echo_L));
echo_R_mo = abs(echo_R);
echo_L_mo = abs(echo_L);
% [y_max_R, x_max_R] = find(echo_R_mo(:,kkkk) == R_max(kkkk));
% [y_max_L, x_max_L] = find(echo_L_mo(:,kkkk) == L_max(kkkk));
% plot(abs(echo_R(y_max_R-255:y_max_R+256, kkkk)),'r'); % 红色是右旋接收
% plot(abs(echo_L(y_max_L-255:y_max_L+256, kkkk)),'b'); % 蓝色是左旋接收
% legend('右通道接收','左通道接收','location','northeast');
% xlabel('高分辨一维距离像');ylabel('幅值');title('单幅高分辨一维距离像');
% hold off;

% 合成的距离像可视化
% for i = 1:kkkk
%     figure;hold on;
%     R_max = max(abs(echo_R));
%     L_max = max(abs(echo_L));
%     echo_R_mo = abs(echo_R);
%     echo_L_mo = abs(echo_L);
%     [y_max_R, x_max_R] = find(echo_R_mo(:,i) == R_max(i));
%     [y_max_L, x_max_L] = find(echo_L_mo(:,i) == L_max(i));
%     plot(abs(echo_R(y_max_R-255:y_max_R+256, i)),'r'); % 红色是右旋接收
%     plot(abs(echo_L(y_max_L-255:y_max_L+256, i)),'b'); % 蓝色是左旋接收
%     legend('右通道接收','左通道接收','location','northeast');
%     xlabel('高分辨一维距离像');ylabel('幅值');title('单幅高分辨一维距离像');
%     hold off;
% end

echo = zeros(2, M, 512);
for i = 1:M
    [y_max_R, x_max_R] = find(echo_R_mo(:,i) == R_max(i));
    [y_max_L, x_max_L] = find(echo_L_mo(:,i) == L_max(i));
    echo(1, i, :) = echo_R(y_max_R-255:y_max_R+256, i);
    echo(2, i, :) = echo_L(y_max_L-255:y_max_L+256, i);
end

savepath = 'H:\ZekunGuo\CODE_MAT\HRRP_烟台数据处理\Data\';
savename = ['24_14.mat'];
save([savepath, savename], 'echo');

% figure;plot(abs(echo_L(:,4)),'r');%显示其中第一帧的拼接后的距离像
% xlabel('高分辨距离单元');ylabel('幅值');title('高分辨一维距离像');
% figure;plot(abs(echo_L(:,1)));
% xlabel('高分辨距离单元');ylabel('幅值');title('高分辨一维距离像（L接收）');
% xx = 1406;
% d=y_control(1,4)*0.625-83+xx*3.125-length(sig)*3.125
% xx1=1640;
% d=y_control(1,4)*0.625-83+xx1-length(sig)*3.125

% figure;imagesc(abs(echo_R));figure;imagesc(abs(echo_L));
% % xlabel('样本个数');ylabel('距离分辨单元');title('高分辨一维距离像序列（RL极化）');
% figure;imagesc(abs(echo_L));
% % xlabel('样本个数');ylabel('距离分辨单元');title('高分辨一维距离像序列（LL极化）');
% save echo_corner_115405 echo_R echo_L;
% save echo_target_120042 echo_R echo_L;
% load('echo_target_174436');
% figure;plot(abs(echo_R(:,1)),'r');hold on;plot(abs(echo_L(:,1)),'b');
% save echo_target_174436 echo_R echo_L;
% save echo_tp_180241 echo_R echo_L;   %电线杆

