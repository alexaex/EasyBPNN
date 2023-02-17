#include <DNN/module/module.h>
#include <Eigen/Dense>
#include <vector>

dnn::Linear::Linear(int in_features, int out_features):
        weight(Eigen::MatrixXd::Random(out_features,in_features)),
        bias(Eigen::MatrixXd::Zero(out_features,1)),
        weight_grad(Eigen::MatrixXd::Zero(out_features,in_features)),
        bias_grad(Eigen::MatrixXd::Zero(out_features,1))
{}


Eigen::MatrixXd dnn::Linear::forward(const Eigen::MatrixXd &input) {
    this->layer_input = input;
    auto output = input * weight.transpose() + bias.transpose().replicate(input.rows(),1);
    return output;
}



Eigen::MatrixXd
dnn::Linear::backward(const Eigen::MatrixXd& grad_input) {



    this->weight_grad = grad_input * layer_input;

    this->bias_grad = grad_input.rowwise().sum();
    auto grad =weight.transpose() * grad_input;
    return grad;

}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> dnn::Linear::parameters() const {
    return std::make_pair(this->weight,this->bias);
}


std::pair<Eigen::MatrixXd, Eigen::MatrixXd> dnn::Linear::grad() const {
    return std::make_pair(this->weight_grad,this->bias_grad);
}

void dnn::Linear::zero_grad() {
    this->weight_grad.setZero();
    this->bias_grad.setZero();
}

void dnn::Linear::step(double lr) {
    this->weight -= lr * this->weight_grad;
    this->bias -= lr * this->bias_grad;
}


Eigen::MatrixXd dnn::MSE::forward(const Eigen::MatrixXd &input) {
    return input;
}

void dnn::MSE::forward(std::vector<std::shared_ptr<dnn::module>>& model,const Eigen::MatrixXd &input, const Eigen::MatrixXd &output) {
    auto x = input;



    for(auto& layer : model)
        x = layer->forward(x);

    this->diff = x - output;
    this->loss = (diff.array()*diff.array()).sum() / (int)output.rows();
    std::cout<<"loss@"<<this->loss<<std::endl;

}

Eigen::MatrixXd
dnn::MSE::backward(const Eigen::MatrixXd& grad_input) {
    return grad_input;
}



void dnn::MSE::backward(std::vector<std::shared_ptr<dnn::module>>& model) {

    Eigen::MatrixXd grad= 2 * diff.transpose() / diff.rows();
    for(auto itr = model.rbegin();itr!=model.rend();itr++)
    {
        grad = (*itr)->backward(grad);
    }
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> dnn::MSE::parameters() const {
    return std::make_pair(diff,diff);
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> dnn::MSE::grad() const {
    return std::make_pair(diff,diff);
}


dnn::ReLU::ReLU(){/* pass */}


/*
Eigen::MatrixXd dnn::ReLU::forward(const Eigen::MatrixXd &input) {
    Eigen::MatrixXd output;
    output.setZero(input.rows(),input.cols());
    for(int i  = 0;i<input.rows();i++){
        for(int j = 0;j<input.cols();j++){
            if(input(i,j) > 0)
                output(i,j) = input(i,j);
        }
    }
    return output;
}
*/

Eigen::MatrixXd dnn::ReLU::backward(const Eigen::MatrixXd &grad_input) {
    Eigen::MatrixXd cur_grad = grad_input;
    for(auto i = 0;i<cur_grad.rows();i++){
        for(auto j = 0;j<cur_grad.cols();j++){
            if(cur_grad(i,j) > 0)
                cur_grad(i,j) = 1;
            else
                cur_grad(i,j) = 0;
        }
    }
    return cur_grad;
}



Eigen::MatrixXd dnn::ReLU::forward(const Eigen::MatrixXd &input) {
    return input.cwiseMax(0);
}

dnn::Sigmoid::Sigmoid(){/* pass */}

Eigen::MatrixXd dnn::Sigmoid::backward(const Eigen::MatrixXd &grad_input) {
    Eigen::MatrixXd cur_grad = grad_input.transpose();
    cur_grad = cur_grad.array() * output_.array() * (1 - output_.array());
    return cur_grad.transpose();
}

Eigen::MatrixXd dnn::Sigmoid::forward(const Eigen::MatrixXd &input) {
    output_ = 1.0 / (1.0 + (-input.array()).exp());
    return output_;
}
