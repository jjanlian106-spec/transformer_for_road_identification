import matlab.engine
import os
import csv

#启动MATLAB引擎
my_engine = matlab.engine.start_matlab()
#加载Simulink模型
env_name = "/Users/lzj/github_project/transformer_for_road_identification/compline_with_m/simu_py.slx"
my_engine.load_system(env_name)
#加载模型所在路径
mfile_dir = "/Users/lzj/github_project/transformer_for_road_identification/compline_with_m"
# Add the directory (and subdirectories) to MATLAB path so the .m function is found
my_engine.addpath(my_engine.genpath(mfile_dir), nargout=0)
# my_engine.cd(mfile_dir, nargout=0)
# 运行初始化m文件start_m （m文件的函数名）
my_engine.start_m(nargout=0)
input_value = []
input_prob = []
with open('/Users/lzj/github_project/transformer_for_road_identification/road_info/predict_road_info/predict_road_mu.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        input_value.append(float(row['mu']))
        input_prob.append(float(row['prob']))
results = []
dynamic_results = []
for i in range(len(input_value)):
    [mu_est_by_fusion,mu_est_by_vision,mu_est_by_dynamic,mu_est_by_vision_prob]= my_engine.simu_py_use(input_value[i],input_prob[i],nargout=4)
    print("===================第{}次仿真结果==================".format(i+1)) 
    print(f"mu fusion: {mu_est_by_fusion}")
    print(f"mu vision: {mu_est_by_vision}")
    print(f"mu dynamics: {mu_est_by_dynamic}")
    print(f"mu vision prob: {mu_est_by_vision_prob}")
    results.append(mu_est_by_fusion)
    dynamic_results.append(mu_est_by_dynamic)
#保存结果到csv文件
with open('/Users/lzj/github_project/transformer_for_road_identification/road_info/fusion_road_info/fusion_road_mu.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['index', 'mu'])
    for idx, val in enumerate(results, start=1):
        writer.writerow([idx, val])
with open('/Users/lzj/github_project/transformer_for_road_identification/road_info/dynamic_road_info/dynamic_road_mu.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['index', 'mu'])
    for idx, val in enumerate(dynamic_results, start=1):
        writer.writerow([idx, val])

