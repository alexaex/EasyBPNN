#include<iostream>
#include <vector>
#include <DNN/module/module.h>
#include <DNN/optimizer_utils/optimizer_utils.h>
#include <Eigen/Dense>



/***********************************
 ***********************************
 **********TEST EXAMPLE*************
 ***********************************
 ***********************************/





int main(){

    // A normal MLP
    std::vector<std::shared_ptr<dnn::module>> model;
    model.push_back(std::make_shared<dnn::Linear>(1,128));
    model.push_back(std::make_shared<dnn::Sigmoid>());
    model.push_back(std::make_shared<dnn::Linear>(128,64));
    model.push_back(std::make_shared<dnn::Sigmoid>());
    model.push_back(std::make_shared<dnn::Linear>(64,32));
    model.push_back(std::make_shared<dnn::Sigmoid>());
    model.push_back(std::make_shared<dnn::Linear>(32,1));
    dnn::MSE loss_fn;


    // OUTPUT = SIN(X)
    Eigen::MatrixXd input = Eigen::MatrixXd::Zero(628,1);
    for(auto i = 0;i<input.rows();i++){
        input(i,0) = (i+1) * 0.01;
    }
    auto output = input.array().sin();

    std::cout<<"===========================================================\n";




    std::pair<double,double> metric(0.1,0.2);
    optim::SGD optimizer(8e-2);


    for(auto i = 1;i<=5000;i++){

        if(metric.first == metric.second){
            std::cout<<"early stop @ epochs:"<<i<<", loss@"<<metric.first<<std::endl;
            break;
        }

        optimizer.zero_grad(model);
        metric.second = metric.first;
        loss_fn.forward(model,input,output);
        std::cout<<"=========================================================================\n";
        metric.first = loss_fn.metric();
        loss_fn.backward(model);
        optimizer.step(model); // update parameters with gradient


    }




    Eigen::MatrixXd valid = Eigen::MatrixXd::Zero(4,1);
    valid(0,0) = 0.0;
    valid(1,0) = 1.57;
    valid(2,0) = 3.14;
    valid(3,0) = 4.71;
    std::cout<<valid<<std::endl;

    for(auto& layer:model){
        valid = layer->forward(valid);
    }
    std::cout<<valid<<std::endl;

};