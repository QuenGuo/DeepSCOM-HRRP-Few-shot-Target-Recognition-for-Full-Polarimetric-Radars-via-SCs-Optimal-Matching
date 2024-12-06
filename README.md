DeepSCCOM: HRRP Few-shot Target Recognition for Full Polarimetric Radars via SCs Optimal Matching

这是雷达高分辨一维距离像成像的matlab 代码，TAES文章《HRRP Few-shot Target Recognition for Full Polarimetric Radars via SCs Optimal Matching》所采用的实测数据成像方法与此类似，原理相同。

my_yuchuli_919d_1channel.m是主程序.
back_matrix_2.m是成像环节的像拼接代码，方法为推逆舍弃法。


输入为雷达所采集的".dat"格式文件
输出为一维距离像数据

这个代码处理的是双圆极化HRRP数据，可利用双圆极化HRRP数据进一步转化为全极化HRRP数据。
