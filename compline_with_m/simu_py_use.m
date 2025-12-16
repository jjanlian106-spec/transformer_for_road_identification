%open simulink file（一定要在当前工作区）
%simu_py;
%启动模型
%set_param("simu_py","SimulationCommand","start");
%暂停模型
%set_param("simu_py","SimulationCommand","pause");
%模型前进一个步长
%set_param("simu_py","SimulationCommand","step");
%读取当前的时间戳
%clock = evalin("caller","out.time.data(end)");

%内置simulink执行函数(函数返回clock时间戳，x为输入)
function[mu_est_by_fusion,mu_est_by_vision,mu_est_by_dynamics,mu_est_by_vision_prob]= simu_py_use(mu,prob)
set_param("simu_py","SimulationCommand","pause");
%转化成字符串类型，否则没办法输入
mu = num2str(mu);%输入视觉估计出来的mu
prob = num2str(prob);%输入视觉估计出来的准确率
%暂停模型
%set_param("simu_py","SimulationCommand","pause");
%更改simulink模块中的值
set_param("simu_py/input_value","Value",mu);
set_param("simu_py/input_prob","Value",prob);
%模型前进一个步长
set_param("simu_py","SimulationCommand","step");
%获取FFRLS估计得到的路面峰值附着系数
%融合结果接口
mu_est_by_fusion = evalin("caller","out.mu_est_by_fusion(end)");%by scope
%视觉结果接口
mu_est_by_vision = evalin("caller","out.mu_est_by_vision(end)");%by to_workspace
%视觉概率接口
mu_est_by_vision_prob = evalin("caller","out.mu_est_by_vision_prob(end)");%byto_workspace
%动力学结果接口
mu_est_by_dynamics = evalin("caller","out.mu_est_by_dynamic(end)");%by to_workspace
end
