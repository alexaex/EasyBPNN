#ifndef MODULE_H
#define MODULE_H

/*********************************
 * *********include headers*******
 * *******************************
 *********************************/
#include <iostream>
#include <Eigen/Dense>
#include <vector>

namespace dnn {

    class module {
    public:

        virtual Eigen::MatrixXd forward(const Eigen::MatrixXd &input) = 0;

        virtual void forward(std::vector<std::shared_ptr<dnn::module>>& model,const Eigen::MatrixXd &input, const Eigen::MatrixXd &output) = 0;

        virtual Eigen::MatrixXd
        backward(const Eigen::MatrixXd& grad_input) = 0;

        virtual void backward(std::vector<std::shared_ptr<dnn::module>>&) = 0;

        virtual std::pair<Eigen::MatrixXd, Eigen::MatrixXd> parameters() const = 0;

        virtual std::pair<Eigen::MatrixXd, Eigen::MatrixXd> grad() const = 0;

        virtual void zero_grad() = 0;

        virtual void step(double) = 0;
    };

    class Linear : public dnn::module {
    public:
        Linear(int in_features, int out_features);

        Eigen::MatrixXd forward(const Eigen::MatrixXd &input) override;


        Eigen::MatrixXd
        backward(const Eigen::MatrixXd& grad_input) override;

        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> parameters() const override;

        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> grad() const override;

        void zero_grad() override;

        void step(double lr) override;

    private:
        // placeholder
        void forward(std::vector<std::shared_ptr<dnn::module>>& model,const Eigen::MatrixXd &input, const Eigen::MatrixXd &output) override {/*pass */
            }
        void backward(std::vector<std::shared_ptr<dnn::module>>& model) override {/*pass */};
        //end
        Eigen::MatrixXd weight;
        Eigen::MatrixXd bias;
        Eigen::MatrixXd weight_grad;
        Eigen::MatrixXd bias_grad;
        Eigen::MatrixXd layer_input;

    };

    class ReLU : public dnn::module{
    public:
        ReLU();

        Eigen::MatrixXd forward(const Eigen::MatrixXd &input) override;


        Eigen::MatrixXd
        backward(const Eigen::MatrixXd& grad_input) override;




        //do nothing
        void zero_grad() override{};

        void step(double lr) override{};

    private:
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> parameters() const override{
            return std::make_pair(Eigen::MatrixXd::Random(),Eigen::MatrixXd::Random());
        }
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> grad() const override{
            return std::make_pair(Eigen::MatrixXd::Random(),Eigen::MatrixXd::Random());
        }



        // placeholder
        void forward(std::vector<std::shared_ptr<dnn::module>>& model,const Eigen::MatrixXd &input, const Eigen::MatrixXd &output) override {
            /*pass */
        }

        void backward(std::vector<std::shared_ptr<dnn::module>>& model) override {
            /*pass */
        };
        //end
    };

    class Sigmoid : public dnn::module{
    public:
        Sigmoid();

        Eigen::MatrixXd forward(const Eigen::MatrixXd &input) override;

        Eigen::MatrixXd backward(const Eigen::MatrixXd &grad_input) override;

        //do nothing
        void zero_grad() override{};

        void step(double lr) override{};

    private:
        Eigen::MatrixXd output_;
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> parameters() const override{
            return std::make_pair(Eigen::MatrixXd::Random(),Eigen::MatrixXd::Random());
        }
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> grad() const override{
            return std::make_pair(Eigen::MatrixXd::Random(),Eigen::MatrixXd::Random());
        }

        // placeholder
        void forward(std::vector<std::shared_ptr<dnn::module>>& model,const Eigen::MatrixXd &input, const Eigen::MatrixXd &output) override {
            /*pass */
        }

        void backward(std::vector<std::shared_ptr<dnn::module>>& model) override {
            /*pass */
        };
        //end
    };

    class MSE : public dnn::module {
    public:

        void forward(std::vector<std::shared_ptr<dnn::module>>& model,const Eigen::MatrixXd &input, const Eigen::MatrixXd &output)override;

        void backward(std::vector<std::shared_ptr<dnn::module>>& model)override;

        double metric()const{
            return loss;
        }

    private:
        double loss{0.0};

        Eigen::MatrixXd diff;

        Eigen::MatrixXd forward(const Eigen::MatrixXd &input) override;

        Eigen::MatrixXd
        backward(const Eigen::MatrixXd& grad_input) override;

        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> parameters() const override;
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> grad() const override;

        void zero_grad() override{
            /*pass */
        }

        void step(double lr) override{
            /*pass */
        }
    };


}







#endif
