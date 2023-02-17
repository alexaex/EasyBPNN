#include <iostream>
#include <DNN/module/module.h>
#include <DNN/optimizer_utils/optimizer_utils.h>
#include <Eigen/Dense>


optim::SGD::SGD( double lr) {
    this->learning_rate = lr;
}

void optim::SGD::zero_grad(std::vector<std::shared_ptr<dnn::module>>& model) {
    for(auto& layer:model){
        layer->zero_grad();
    }
}

void optim::SGD::step(std::vector<std::shared_ptr<dnn::module>>& model) {
    for(auto& layer:model){
        layer->step(this->learning_rate);
    }
}
