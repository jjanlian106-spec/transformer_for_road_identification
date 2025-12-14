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
function[mu_est_by_dynamic,mu_est_by_vision]= simu_py_use(x)
set_param("simu_py","SimulationCommand","pause");
%转化成字符串类型，否则没办法输入
x = num2str(x);
%暂停模型
%set_param("simu_py","SimulationCommand","pause");
%更改simulink模块中的值
set_param("simu_py/input_value","Value",x);
%模型前进一个步长
set_param("simu_py","SimulationCommand","step");
%获取FFRLS估计得到的路面峰值附着系数
mu_est_by_dynamic = evalin("caller","out.mu_est_by_dynamic(end,2)");%by scope
mu_est_by_vision = evalin("caller","out.mu_est_by_vision(end)");%by to_workspace
end
