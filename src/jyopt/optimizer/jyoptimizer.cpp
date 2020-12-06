#include <assert.h>
#include <jyopt/common/constants.h>
#include <jyopt/optimizer/jyoptimizer.h>
#include <iostream>

namespace JYOPT
{
bool_t Jyoptimizer::MatrixInit()
{
    x_ = MatrixX::Zero(1, 4);
    x_ << 1, 5, 5, 1;

    xl_ = MatrixX::Zero(1, 4);
    xl_ << 1, 1, 1, 1;

    xu_ = MatrixX::Zero(1, 4);
    xu_ << 5, 5, 5, 5;

    bl_ = MatrixX::Zero(1, 2);
    bl_ << 25, 40;

    bu_ = MatrixX::Zero(1, 2);
    bu_ << MAX_FLOAT64_NUM, 40;

    s_ = MatrixX::Zero(1, 1);
    s_ << 0;

    sl_ = MatrixX::Zero(1, 1);
    sl_ << 25;

    su_ = MatrixX::Zero(1, 1);
    su_ << MAX_FLOAT64_NUM;

    residual_ = MatrixX::Zero(1, 2);
    residual_ = CalcRes();

    CheckXSFeasible();

    zl_ = MatrixX::Zero(1, x_.cols() + s_.cols());

    zu_ = MatrixX::Zero(1, x_.cols() + s_.cols());

    UpdateZLU();

    // init
    g_ = MatrixX::Zero(1, 5);
    g_         = CalcObjgrad();

    // init
    j_ = MatrixX::Zero(2, 5);
    j_         = CalcJac();

    return true;
}

MatrixX Jyoptimizer::CalcJacStub()
{
    MatrixX res = MatrixX::Zero(2, 4);

    float64_t x1 = x_(0, 0);
    float64_t x2 = x_(0, 1);
    float64_t x3 = x_(0, 2);
    float64_t x4 = x_(0, 3);

    res(0, 0) = x2 * x3 * x4;
    res(0, 1) = x1 * x3 * x4;
    res(0, 2) = x1 * x2 * x4;
    res(0, 3) = x1 * x2 * x3;

    res(1, 0) = 2 * x1;
    res(1, 1) = 2 * x2;
    res(1, 2) = 2 * x3;
    res(1, 3) = 2 * x4;

    return res;
}

MatrixX Jyoptimizer::CalcJac()
{
    MatrixX res_temp = CalcJacStub();

    MatrixX res           = MatrixX(2, 5);
    res.block(0, 0, 2, 4) = res_temp;
    res(0, 4)             = -1;
    res(1, 4)             = 0;

    // std::cout<<res<<std::endl;
    return res;
}

MatrixX Jyoptimizer::CalcObjgrad()
{
    MatrixX res = MatrixX::Zero(1, 5);

    float64_t x1 = x_(0, 0);
    float64_t x2 = x_(0, 1);
    float64_t x3 = x_(0, 2);
    float64_t x4 = x_(0, 3);

    res(0, 0) = x4 * (2 * x1 + x2 + x3);
    res(0, 1) = x1 * x4;
    res(0, 2) = x1 * x4 + 1;
    res(0, 3) = x1 * (x1 + x2 + x3);
    res(0, 4) = 0.0;

    // std::cout<<res<<std::endl;
    return res;
}

void Jyoptimizer::UpdateZLU()
{
    for (uint32_t i = 0; i < 4; i++)
    {
        zl_(0, i) = param_.mu_ / (x_(0, i) - xl_(0, i));
        zu_(0, i) = param_.mu_ / (xu_(0, i) - x_(0, i));
    }
    zl_(0, 4) = param_.mu_ / (s_(0, 0) - sl_(0, 0));
    zu_(0, 4) = param_.mu_ / (su_(0, 0) - s_(0, 0));
}

void Jyoptimizer::CheckXSFeasible()
{
    for (uint32_t i = 0; i < 4; i++)
    {
        if (x_(0, i) <= xl_(0, i))
        {
            x_(0, i) = std::min<float64_t>(xl_(0, i) + ALMOST_ZERO, xu_(0, i));
        }
        if (x_(0, i) >= xu_(0, i))
        {
            x_(0, i) = std::max<float64_t>(xl_(0, i), xu_(0, i) - ALMOST_ZERO);
        }
    }

    if (s_(0, 0) <= sl_(0, 0))
    {
        s_(0, 0) = std::min<float64_t>(sl_(0, 0) + ALMOST_ZERO, su_(0, 0));
    }
    if (s_(0, 0) >= su_(0, 0))
    {
        s_(0, 0) = std::max<float64_t>(sl_(0, 0), su_(0, 0) - ALMOST_ZERO);
    }
}
MatrixX Jyoptimizer::CalcResStub()
{
    float64_t x1 = x_(0, 0);
    float64_t x2 = x_(0, 1);
    float64_t x3 = x_(0, 2);
    float64_t x4 = x_(0, 3);

    MatrixX res = MatrixX::Zero(1, 2);
    // inequality constrain
    res(0, 0) = x1 * x2 * x3 * x4;

    // equality constrain
    res(0, 1) = x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4;

    return res;
}

MatrixX Jyoptimizer::CalcRes()
{
    MatrixX res = CalcResStub();
    assert(res.rows() == 1);
    assert(res.cols() == 2);

    // inequality constrain
    res(0, 0) = res(0, 0) - s_(0, 0);

    // equality  constrain
    res(0, 1) = res(0, 1) - bl_(0, 1);

    return res;
}

bool_t Jyoptimizer::OptCalcFunc()
{
    MatrixInit();

    // float64_t tau = std::min<float64_t>(param_.tau_max_, 100 * param_.mu_);

    ShowResidual();
    ShowX();
    ShowS();
    ShowG();
    ShowJ();

    return true;
}
bool_t Jyoptimizer::ShowX()
{
    std::cout << " - - - X - - - " << std::endl;
    std::cout << x_ << std::endl;
    return true;
}
bool_t Jyoptimizer::ShowS()
{
    std::cout << " - - - s - - - " << std::endl;
    std::cout << s_ << std::endl;
    return true;
}

bool_t Jyoptimizer::ShowG()
{
    std::cout << " - - - g - - - " << std::endl;
    std::cout << g_ << std::endl;
    return true;
}
bool_t Jyoptimizer::ShowJ()
{
    std::cout << " - - - J - - - " << std::endl;
    std::cout << j_ << std::endl;
    return true;
}
bool_t Jyoptimizer::ShowResidual()
{
    std::cout << " - - - Residual - - - " << std::endl;
    std::cout << residual_ << std::endl;
    return true;
}
}  // namespace JYOPT