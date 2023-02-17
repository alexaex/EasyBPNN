#ifndef OPTIMIZER_UTILS_H
#define OPTIMIZER_UTILS_H

/*********************************
 * *********include headers*******
 * *******************************
 *********************************/
#include <iostream>
#include "Eigen/Dense"
#include <vector>
#include <DNN/module/module.h>



namespace optim{
    class optimizer{
        virtual void zero_grad(std::vector<std::shared_ptr<dnn::module>>& model) = 0;
        virtual void step(std::vector<std::shared_ptr<dnn::module>>& model) = 0;
    };



    class SGD:public optimizer{
    public:
        explicit SGD(double lr);
        void zero_grad(std::vector<std::shared_ptr<dnn::module>>& model)override;
        void step(std::vector<std::shared_ptr<dnn::module>>& model)override;
    private:
        double learning_rate;

    };

}

















#endif
