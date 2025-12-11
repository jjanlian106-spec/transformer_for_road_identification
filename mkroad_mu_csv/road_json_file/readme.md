1. mkroad_mu_csv/road_json_file/build_road_json.py: 输入全量图像的文件夹路径，输出一个对应地址的json文件
2. mkroad_mu_csv/real_road2csv.py: 读取road_info下面的road.json，参考附着系数字典，生成实际路面附着系数的csv
3. mkroad_mu_csv/predict_road2csv.py:读取road_info下面的road.json，对json文件中的每一个图像进行识别，并将识别结果生成为一个csv
4. mkroad_mu_csv/show.py :读取生成的两个csv，将两个csv对应的结果给可视化出来
5. main_muest.py :主函数，跑一个数据图象时，更改build_road_json.py中输入的图像文件夹即可