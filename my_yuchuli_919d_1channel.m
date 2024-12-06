clc;
clear all
close all
% addpath('G:\�������ݴ���\����ʵ�� ����\�����ݴ���1105\20200730huayin');
addpath('D:\');
filename2='24_14.dat';
% addpath('H:\��̨��������');
% addpath("G:\��̨ʵ���洬����\echos\"); % Gzk
% filename2='72_73.dat';

prfnum = 128; % һ��CPI�ڵ�������
cpi_ks = 2000+(0+0+0+0)*5;   % �ӵڼ�����ʼ����737+180
deltaf_mode = 0;  %  0123�ֱ���6M��12M��20M��40M SPW: delta_f
M = fix(1600); % 2+0*5;      % ����CPI��CPI����
% M = 10;

fftnum=256;
fix_hb_num=8028; % ��ʱ���������
% sum1:����ͨ����һ֡����
sum1 = fix_hb_num*2*prfnum+4; % *2����Ϊ��ǰ����ͨ������8028���������һ��ͨ��������*2; 
cpidata2 = zeros(prfnum, fix_hb_num);

c = 3e8;
fs = 48e6; % ����Ƶ�� 
Tr = 80e-6; % �����ظ����,���ڣ�PRI
rs = c/(2*fs); % ������Ԫ
Rs = c/2/fs; % 3.125m

fid2 = fopen(filename2,'r'); % ���ļ�filename�Ա��Զ����ƶ�ȡ��ʽ���з��ʣ������ص��ڻ����3�������ļ���ʶ��
status = fseek(fid2, (2*sum1*(cpi_ks-1)), 'bof'); % ״̬
% fseek(fid, offset, origin)
% fid���ļ���ʶ����offset��ƫ������origin��ƫ��������ʼλ�á�
% ƫ����������������������ʾ�ӵ�ǰλ����ǰ������ƶ����ٸ��ֽڡ�
% ��ʼλ�ÿ�������������֮һ��
% - 'bof'�����ļ���ͷ��ʼ����ƫ������
% - 'cof'���ӵ�ǰλ�ÿ�ʼ����ƫ������
% - 'eof'�����ļ�ĩβ��ʼ����ƫ������

ps = ftell(fid2); 
% ����ָ���ļ���λ��ָ��ĵ�ǰλ��;
% �����ѯ�ɹ����� position �Ǵ� 0 ��ʼ��������ָʾ���ļ���ͷ����ǰλ�õ��ֽ�����
% �����ѯ���ɹ����� position Ϊ -1��
% if status == -1;
%         disp('ERROR: The end of the file!');
%         break;
% end
    
nn=1; count=0;
for mm = 1:M; % cpi_ks: cpi_ks+M-1
    mm
    % status = fseek(fid2,2*sum1*2*50,'cof');
    status = fseek(fid2, 2*sum1*2*5, 'cof'); % offset��ƫ���� 2*sum1*2*50
    ps0(mm) = ftell(fid2);   % ps0:ÿ�δ�����һ֡���������ݺ�ָ�뵱ǰ��λ��
    y2 = fread(fid2, 2*sum1,'int16','s'); % 2*sum1:��ȡsize()
     
    if(length(y2) < 2*sum1)
        disp('ERROR: The data length is not enough!!');
        break;
    end
     
    ps1(mm) = ftell(fid2);   % ps1:��ȡ1֡���ݺ�ָ������λ�ã�Ӧ�ñ�ps0����һ֡����*2����Ϊint16��ȡ
    for kk=1:1:sum1*2
        if ((y2(kk)==32767) && (y2(kk+3)==32767) && (y2(kk+4)==4095) && (y2(kk+7)==4095)) % ���for motivation?
            offset2(mm)=kk-1; % offset2 ���ã�
            break;
        end
    end
    status = fseek(fid2,-2*(2*sum1-offset2(mm)),'cof');
    ps2(mm) = ftell(fid2);    % ps2:�ҵ�֡ͷ��ָ����ǰ�ƶ���֡ͷλ��
    y22=fread(fid2,sum1,'int16','s'); % ������ǰ֡������

    ps3(mm) = ftell(fid2);     % ps3:��ȡ��ǰ֡�������ݺ�ָ��λ�ã�Ӧ�ñ�ps2����һ֡����*2
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
        % transmit_mode = 1; % y_control(nn,7); % transmit_mode: 1R����2L��
        
        transmit_mode = y_control(nn,7);
        disp("transmit_mode: ");
        disp(transmit_mode);
        
    % 1��ָʾ���룻2:��Ŀ�����ٶȣ�3������ģʽ��123�ֱ��������ػ���٣�4:������㣻5���������ͣ�0�ͺ�1�Ͳ
    % 6���ź�ʱ��0-��PD��14us��1-��PD,5us��3-��PD��130ns��8-��SF��42us��9-��SF��24us��10-��SF,14us��11-��SF��5us��12-��SF��130ns
    % 7��������ʽ: 0 1 2 5���ֱ���:����;��L;��R;���淢��8������ͨ���Ĳ�����λ��9������1->24M��4->192M
    %10����λ�Ͳ����λ��11�������Ͳ����λ��12����λ�ǣ�13�������ǣ�14�����پ���
        
        y_RI = yI_hb(:,31:2:end);
        y_RQ = yQ_hb(:,31:2:end);
        y_LI = yI_hb(:,32:2:end);
        y_LQ = yQ_hb(:,32:2:end);
        
        LorR(mm) = y_RI(6,7);
        
        R = y_RI + j*y_RQ;
        L = y_LI + 1i*y_LQ;
%         R(:,1:15)=0;L(:,1:15)=0;
        
        %% ������ȡ
        tao_mode = y_control(nn,6); %11
        switch tao_mode
            case 8
                tao = 42e-6;disp('42us');
            case 9
                tao = 24e-6;disp('24us');
            case 10
                tao = 14e-6;disp('14us');
            case 11
                tao = 5e-6;disp('5us'); % ����
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
                prfnum = 128;disp('128��');
            case 1
                prfnum = 64;disp('64��');
            case 2
                prfnum = 32;disp('32��');
            case 3
                prfnum = 256;disp('256��');
        end
        B_mode = 1; % y_control(nn,9);
        switch B_mode
            case 1
                B = 24e6;disp('SF'); % 24M
            case 4
                B = 192e6;disp('PD');
        end
        
        delta_r = c/(2*prfnum*deltaf); % ����ֱ���
        
        %%%%%  ddcϵ��  %%%%%
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
        load('ddc_coe_xin'); % ddc�����±�Ƶ,coe_i1,coe_q1
        fir_ddcI = 2^14*coe_i1;
        fir_ddcQ = 2^14*coe_q1;
%         figure;freqz(fir2)
%                 figure
%                 freqz(fir1)
        
        %=============NCO====
        n=[1:length(y_RI)]; % �൱��range()
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
            ddcRI(i,:) = downsample(ddcRi(i,:), QQ); %��ȡ
            ddcRQ(i,:) = downsample(ddcRq(i,:), QQ);
            ddcLI(i,:) = downsample(ddcLi(i,:), QQ); %��ȡ
            ddcLQ(i,:) = downsample(ddcLq(i,:), QQ);
        end
        
        R_ddcout = ddcRI + j*ddcRQ; % R·����
        L_ddcout = ddcLI + j*ddcLQ; % L·����
%         figure;plot(abs(R_ddcout).','r');   hold on;plot(abs(L_ddcout).','b')
        
        
        %             fd=12.5e3/128*8;
%               figure;subplot(211),plot(real(R_ddcout.'))
%               subplot(212),plot(imag(R_ddcout.'))
%                 figure;subplot(211),plot(real(L_ddcout.'));
%                 subplot(212),plot(imag(L_ddcout.'))
        
%       ���շ�����λ����
%       amp_bc = 1;%1.1885;
        coe = exp(j*y_control(nn,8)/57.3);
        L_ddcout = L_ddcout*coe;

        k = B/tao; % ��Ƶб��
        tp = (-tao/2 : 1/fs : tao/2-1/fs); % +tao/2 ����
        sig = exp(-j*pi*k .* tp.^2); % ���Ե�Ƶ�����ź�
        scoe_l = fliplr((sig)) .* (hamming(length(tp))).'; % fliplr ��������ת������������ h(n)
        fcoe_l = fft(2^11 * fliplr((sig)) .* (hamming(length(sig))).', 4096); % h(f)
        scoe_r = fliplr((sig)) .* (hamming(length(tp))).'; % h(n)
        fcoe_r = fft(2^11 * fliplr(conj(sig)) .* (hamming(length(sig))).', 4096); % h(f)
        % ʱ����--Ƶ����ˣ���ѹ��h(f).*s(f)
        
        % ��һ��CPI�е�ÿһ�����������ѹ
        for i = 1:prfnum 
%             R_pc_fft(i,:) = fft(R_ddcout(i,:),4096).*fcoe_r;     % R
%             R_pcoutq(i,:) = ifft(R_pc_fft(i,:),4096);
%             R_pcout_t(i,:) = conv(R_ddcout(i,:),scoe_r);
            
            R_pc_fft1(i,:) = fft (R_ddcout(i,:), 4096) .* fcoe_l; % RL--R��L��
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
        R_pcout1 = L_pcout1q(:,:); % ΪɶL��ֵ��R��
        L_pcout1 = R_pcout1q(:,:);
        
        echo_R_temp(:,mm) = R_pcout1(1, 1:1500); % ��ѹ��һά���������У�Rͨ�����գ�
        echo_L_temp(:,mm) = L_pcout1(1, 1:1500); % ��ѹ��һά���������У�Lͨ�����գ�
%      figure(1);plot(abs(R_pcout1).','r');   hold on;plot(abs(L_pcout1).','b')
%     
%      figure;plot(rs*(1:size(L_pcout1,2)),abs(R_pcout1).');figure;plot(rs*(1:size(L_pcout1,2)),abs(L_pcout1.'));   
%         noise_L_power = L_pcout1(:,200:500);
% ���Ȳ���
%         noise_L = abs(L_pcout1(:,200:500));
%         noise_R = abs(R_pcout1(:,200:500));
%         power_L_n = std(noise_L(:));
%         power_R_n = std(noise_R(:));
%         amp_bc = 1.5313;%power_L_n/power_R_n;
%         R_pcout1 = amp_bc*R_pcout1;
   %     figure;plot(abs(R_pcout1q).');   figure;plot(abs(L_pcout1).')
    %% ��ƴ��
%         
%         figure;imagesc(abs(R_pcout1));
%         figure;imagesc(abs(L_pcout1));
%         figure;plot(abs(L_pcout1(1,:)),'r');hold on;
%         plot(abs(R_pcout1(:,1)),'b');legend('LL','RL');xlabel('����ֱ浥Ԫ'),ylabel('��һ������');
% axis tight;title('');

        % ��� X ���������� fft(X) ���ظ������ĸ���Ҷ�任��
        % ��� X �Ǿ����� fft(X) �� X �ĸ�����Ϊ������������ÿ�еĸ���Ҷ�任��
        R_pc_m = ifft(R_pcout1); % ��128*4096��ÿ����128����FFT
        L_pc_m = ifft(L_pcout1);
        [N,samplenumber] = size(R_pc_m); % ������ ������
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
%         title('���Ԫ�����ڵ�һ��');
%         legend('R','L');axis tight;
        Rtaur = c/2/B; % ����Ƶ�ϳ�֮ǰ����ֱ���
        wtd = 0.3;
        thd = 0.5;
%         [R_profile,L_profile]=back_matrix_2(R_pc_m,L_pc_m,Rtaur,rs,delta_r);
%         R_pc_m = flipud(R_pc_m);
%         L_pc_m = flipud(L_pc_m);
        % [R_profile,L_profile]=big_abondon_2(R_pc_m,L_pc_m,rs,delta_r); % delta_r�ǲ���Ƶ����ֱ���
        % [R_profile,L_profile]=abondon_2(R_pc_m,L_pc_m,Rs,delta_r,samplenumber,N); 
        [R_profile,L_profile]=back_matrix_2(R_pc_m,L_pc_m,Rtaur,Rs,delta_r); 
        % [R_profile, L_profile, row, col] = chouqu_2channel(R_pc_m, L_pc_m, fs, deltaf, wtd, thd);
        if y_control(mm,7) == 1
           title_name = 'L��';
        elseif y_control(mm,7) == 2
                 title_name = 'R��';
            elseif  y_control(mm,7) == 3
                 title_name = '45�߷�';
            end
%         figure;line1=plot(delta_r*(1:length(R_profile)),db(abs(R_profile)),'r');hold on;line2=plot(delta_r*(1:length(R_profile)),db(abs(L_profile)),'b');
%         xlim([1 2000]);
%         title(title_name);
%         legend('��������','��������');
%         figure;plot(delta_r*(1:length(R_profile))-2.375,(abs(R_profile)),'r');hold on;plot(delta_r*(1:length(R_profile))-2.375,(abs(L_profile)),'b');
%         figure;plot((1:length(R_profile)),(abs(R_profile)),'r');hold on;plot((1:length(R_profile)),(abs(L_profile)),'b');axis tight;
%         xlim([1 2000]);
%         title(title_name);
%         xlabel('����/m'); ylabel('����');
%         legend('��������','��������');
%         [~,max_ind] = max(abs(R_profile)+abs(L_profile));
%         x_cut = max_ind-599:max_ind+2000;
%         x_cut = max_ind-511:max_ind+1536;
%         max_ind =7749;
%         x_cut = max_ind-63:max_ind+64;
% %         x_cut = max_ind-31:max_ind+32;
%       
%       ��ȡ��
        echo_R(:,mm) = R_profile(1:65520);%(x_cut); % �߷ֱ�һά���������У�Rͨ�����գ� 
        echo_L(:,mm) = L_profile(1:65520);%(x_cut); % �߷ֱ�һά���������У�Lͨ�����գ�
        
        
%         figure;plot((1:length(x_cut)),abs(echo_R(:,mm)),'r');hold on;plot((1:length(x_cut)),abs(echo_L(:,mm)),'b');
%         xlabel('����ֱ浥Ԫ');ylabel('����');title('Ŀ��һά������');legend('RL����','LL����');
        nn = nn+1;
        temp(mm) = yI_hb(6,7);
    end
end

% figure;plot(abs(echo_R));
% figure;imagesc(abs(echo_R));
% xlabel('��������');ylabel('����ֱ浥Ԫ');title('�߷ֱ�һά���������У�Rͨ�����գ�');
% figure;imagesc(abs(echo_L));
% xlabel('��������');ylabel('����ֱ浥Ԫ');title('�߷ֱ�һά���������У�Lͨ�����գ�');
% % echo_R = echo_R(:,2:2:end);
% % echo_L = echo_L(:,2:2:end);  
% 
% figure;plot(abs(echo_R_temp));
% figure;imagesc(abs(echo_R_temp));
% xlabel('��������');ylabel('����ֱ浥Ԫ');title('��ѹ��һά���������У�Rͨ�����գ�');
% figure;imagesc(abs(echo_L_temp));
% xlabel('��������');ylabel('����ֱ浥Ԫ');title('��ѹ��һά���������У�Lͨ�����գ�');
% 
% kkkk=3;
% figure;hold on;
% plot(abs(echo_R(11100:11355, kkkk)),'r'); % ��ɫ����������
% plot(abs(echo_L(11100:11355, kkkk)),'b'); % ��ɫ����������
% legend('��ͨ������','��ͨ������','location','northeast');
% xlabel('�߷ֱ�һά������');ylabel('��ֵ');title('�����߷ֱ�һά������');
% hold off;

% figure;hold on;
R_max = max(abs(echo_R));
L_max = max(abs(echo_L));
echo_R_mo = abs(echo_R);
echo_L_mo = abs(echo_L);
% [y_max_R, x_max_R] = find(echo_R_mo(:,kkkk) == R_max(kkkk));
% [y_max_L, x_max_L] = find(echo_L_mo(:,kkkk) == L_max(kkkk));
% plot(abs(echo_R(y_max_R-255:y_max_R+256, kkkk)),'r'); % ��ɫ����������
% plot(abs(echo_L(y_max_L-255:y_max_L+256, kkkk)),'b'); % ��ɫ����������
% legend('��ͨ������','��ͨ������','location','northeast');
% xlabel('�߷ֱ�һά������');ylabel('��ֵ');title('�����߷ֱ�һά������');
% hold off;

% �ϳɵľ�������ӻ�
% for i = 1:kkkk
%     figure;hold on;
%     R_max = max(abs(echo_R));
%     L_max = max(abs(echo_L));
%     echo_R_mo = abs(echo_R);
%     echo_L_mo = abs(echo_L);
%     [y_max_R, x_max_R] = find(echo_R_mo(:,i) == R_max(i));
%     [y_max_L, x_max_L] = find(echo_L_mo(:,i) == L_max(i));
%     plot(abs(echo_R(y_max_R-255:y_max_R+256, i)),'r'); % ��ɫ����������
%     plot(abs(echo_L(y_max_L-255:y_max_L+256, i)),'b'); % ��ɫ����������
%     legend('��ͨ������','��ͨ������','location','northeast');
%     xlabel('�߷ֱ�һά������');ylabel('��ֵ');title('�����߷ֱ�һά������');
%     hold off;
% end

echo = zeros(2, M, 512);
for i = 1:M
    [y_max_R, x_max_R] = find(echo_R_mo(:,i) == R_max(i));
    [y_max_L, x_max_L] = find(echo_L_mo(:,i) == L_max(i));
    echo(1, i, :) = echo_R(y_max_R-255:y_max_R+256, i);
    echo(2, i, :) = echo_L(y_max_L-255:y_max_L+256, i);
end

savepath = 'H:\ZekunGuo\CODE_MAT\HRRP_��̨���ݴ���\Data\';
savename = ['24_14.mat'];
save([savepath, savename], 'echo');

% figure;plot(abs(echo_L(:,4)),'r');%��ʾ���е�һ֡��ƴ�Ӻ�ľ�����
% xlabel('�߷ֱ���뵥Ԫ');ylabel('��ֵ');title('�߷ֱ�һά������');
% figure;plot(abs(echo_L(:,1)));
% xlabel('�߷ֱ���뵥Ԫ');ylabel('��ֵ');title('�߷ֱ�һά������L���գ�');
% xx = 1406;
% d=y_control(1,4)*0.625-83+xx*3.125-length(sig)*3.125
% xx1=1640;
% d=y_control(1,4)*0.625-83+xx1-length(sig)*3.125

% figure;imagesc(abs(echo_R));figure;imagesc(abs(echo_L));
% % xlabel('��������');ylabel('����ֱ浥Ԫ');title('�߷ֱ�һά���������У�RL������');
% figure;imagesc(abs(echo_L));
% % xlabel('��������');ylabel('����ֱ浥Ԫ');title('�߷ֱ�һά���������У�LL������');
% save echo_corner_115405 echo_R echo_L;
% save echo_target_120042 echo_R echo_L;
% load('echo_target_174436');
% figure;plot(abs(echo_R(:,1)),'r');hold on;plot(abs(echo_L(:,1)),'b');
% save echo_target_174436 echo_R echo_L;
% save echo_tp_180241 echo_R echo_L;   %���߸�

