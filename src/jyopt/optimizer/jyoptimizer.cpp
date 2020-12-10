#include <assert.h>
#include <jyopt/common/constants.h>
#include <jyopt/optimizer/jyoptimizer.h>
#include <iostream>
#include <cmath>

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

    if (param_.slack_init_)
    {
        s_(0, 0) = residual_(0, 0);
    }

    CheckXSFeasible();

    tau_ = std::min<float64_t>(param_.tau_max_, 100 * param_.mu_);

    zl_ = MatrixX::Zero(1, x_.cols() + s_.cols());

    zu_ = MatrixX::Zero(1, x_.cols() + s_.cols());

    UpdateZLU();

    // init Gradient matrix of object function
    g_ = MatrixX::Zero(1, 5);
    g_ = CalcObjgrad();

    // init jacobi matrix of constrains
    j_ = MatrixX::Zero(2, 5);
    j_ = CalcJac();

    // init lambda
    lam_             = MatrixX::Zero(2, 1);
    MatrixX temp_lam = pinv((j_ * j_.transpose())) * j_ * (zl_.transpose() - zu_.transpose() - g_.transpose());
    lam_             = temp_lam.transpose();

    th_ = CalcTheta();
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

float64_t Jyoptimizer::CalcObj()
{
    float64_t x1 = x_(0, 0);
    float64_t x2 = x_(0, 1);
    float64_t x3 = x_(0, 2);
    float64_t x4 = x_(0, 3);

    return x1 * x4 * (x1 + x2 + x3) + x3;
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

MatrixX Jyoptimizer::CalcHess()
{
    float64_t x1 = x_(0, 0);
    float64_t x2 = x_(0, 1);
    float64_t x3 = x_(0, 2);
    float64_t x4 = x_(0, 3);

    float64_t lam1 = lam_(0, 0);
    float64_t lam2 = lam_(0, 1);

    MatrixX   h       = MatrixX::Zero(5, 5);
    float64_t objfact = 1.0;

    h(0, 0) = objfact * 2 * x4 + lam2 * 2;
    h(1, 0) = objfact * x4 + lam1 * x3 * x4;
    h(0, 1) = h(1, 0);
    h(1, 1) = lam2 * 2;
    h(2, 0) = objfact * x4 + lam1 * x2 * x4;
    h(0, 2) = h(2, 0);
    h(2, 1) = lam1 * x1 * x4;
    h(1, 2) = h(2, 1);
    h(2, 2) = lam2 * 2;
    h(3, 0) = objfact * (2 * x1 + x2 + x3) + lam1 * x2 * x3;
    h(0, 3) = h(3, 0);
    h(3, 1) = objfact * x1 + lam1 * x1 * x3;
    h(1, 3) = h(3, 1);
    h(3, 2) = objfact * x1 + lam1 * x1 * x2;
    h(2, 3) = h(3, 2);
    h(3, 3) = lam2 * 2;
    return h;
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

float64_t Jyoptimizer::CalcTheta()
{
    MatrixX temp_res = CalcRes();
    return temp_res.cwiseAbs().sum();
}

float64_t Jyoptimizer::CalcPhi()
{
    float64_t ph = CalcObj();

    for (uint32_t i = 0; i < 4; i++)
    {
        ph = ph - param_.mu_ * (log(x_(0, i) - xl_(0, i)) + log(xu_(0, i) - x_(0, i)));
    }

    ph = ph - param_.mu_ * (log(s_(0, 0) - bl_(0, 0)) + log(bu_(0, 0) - s_(0, 0)));

    return ph;
}

bool_t Jyoptimizer::OptCalcFunc()
{
    MatrixInit();

    // ShowResidual();
    // ShowX();
    // ShowS();
    // ShowG();
    // ShowJ();
    // ShowLambda();
    // ShowZLU();

    float64_t alpha_pr = 1.0;
    float64_t alpha_du = 1.0;
    MatrixX   e        = MatrixX::Ones(5, 1);

    uint32_t iteration_num = 0;
    while(iteration_num<param_.max_iteration_num_)
    {
        residual_ = CalcRes();

        th_ = CalcTheta();

        float64_t phi = CalcPhi();

        // make diagonal matrices
        MatrixX zml = MatrixX::Zero(zl_.cols(), zl_.cols());
        for (uint32_t i = 0; i < zl_.cols(); i++)
        {
            zml(i, i) = zl_(0, i);
        }

        MatrixX zmu = MatrixX::Zero(zu_.cols(), zu_.cols());
        for (uint32_t i = 0; i < zu_.cols(); i++)
        {
            zmu(i, i) = zu_(0, i);
        }

        // sigmas
        MatrixX dml=MatrixX::Zero(5,5);
        for(uint32_t i=0;i<dml.cols();i++)
        {
            if(i<4)
            {
                dml(i, i) = x_(0, i) - xl_(0, i);
            }
            else
            {
                dml(i, i) = s_(0, 0) - sl_(0, 0);
            }
        }

        MatrixX dmu = MatrixX::Zero(5, 5);
        for (uint32_t i = 0; i < dmu.cols(); i++)
        {
            if (i < 4)
            {
                dmu(i, i) = xu_(0, i) - x_(0, i);
            }
            else
            {
                dmu(i, i) = su_(0, 0) - s_(0, 0);
            }
        }

        MatrixX inv_dml = MatrixX::Zero(5, 5);
        MatrixX inv_dmu = MatrixX::Zero(5, 5);
        MatrixX sigl    = MatrixX::Zero(5, 5);
        MatrixX sigu    = MatrixX::Zero(5, 5);

        for (uint32_t i = 0; i < 5; i++)
        {
            inv_dml(i, i) = 1 / dml(i, i);
            inv_dmu(i, i) = 1 / dmu(i, i);
            sigl(i, i)    = zml(i, i) / dml(i, i);
            sigu(i, i)    = zmu(i, i) / dmu(i, i);
        }

        //  1st and 2nd derivatives
        j_        = CalcJac();
        MatrixX w = CalcHess();
        g_        = CalcObjgrad();

        MatrixX h = MatrixX::Zero(5, 5);

        h = w + sigl + sigu;

        // construct and solve linear system Ax=b
        MatrixX A1 = MatrixX::Zero(7, 7);

        A1.block(0, 0, 5, 5) = h;
        A1.block(5, 0, 2, 5) = j_;
        A1.block(0, 5, 5, 2) = j_.transpose();

        MatrixX b1       = MatrixX::Zero(7, 1);
        MatrixX b1_temp1 = g_.transpose() - zl_.transpose() + zu_.transpose() + j_.transpose() * lam_.transpose() +
                           zl_.transpose() - param_.mu_ * inv_dml * e - zu_.transpose() + param_.mu_ * inv_dmu * e;
        MatrixX b1_temp2     = residual_.transpose();
        b1.block(0, 0, 5, 1) = b1_temp1;
        b1.block(0, 5, 2, 1) = b1_temp2;

        MatrixX d1 = -1.0 * pinv(A1) * b1;

        MatrixX dx1   = d1.block(0, 0, 4, 1);
        MatrixX ds1   = d1.block(4, 0, 1, 1);
        MatrixX dlam1 = d1.block(5, 0, 2, 1);

        // compute search direction for z (explicit solution)

        MatrixX dzl1         = MatrixX::Zero(5, 1);
        MatrixX dzu1         = MatrixX::Zero(5, 1);
        MatrixX temp_dx1_ds1 = MatrixX::Zero(5, 1);

        temp_dx1_ds1.block(0, 0, 4, 1) = dx1;
        temp_dx1_ds1.block(4, 0, 1, 1) = ds1;

        dzl1 = param_.mu_ * inv_dml * e - zl_.transpose() - sigl * temp_dx1_ds1;
        dzu1 = param_.mu_ * inv_dmu * e - zu_.transpose() + sigu * temp_dx1_ds1;

        // // Un -symmetric ,zl/zu implicitly solved
        // // construct A
        
        // MatrixX A2=MatrixX::Zero(17,17);
        // A2.block(0,0,5,5)=w;
        // A2.block(5,0,2,5)=j_;
        // A2.block(0,5,5,2)=j_.transpose();

        iteration_num++;
    }

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
bool_t Jyoptimizer::ShowLambda()
{
    std::cout<<" - - - Lambda - - - "<<std::endl;
    std::cout<<lam_<<std::endl;
    // std::cout<<" + + + + + +"<<std::endl;
    // std::cout<<(j_ * j_.transpose())<<std::endl;
    // std::cout<<" + + + + + +"<<std::endl;
    // std::cout<<pinv((j_ * j_.transpose()))<<std::endl;
    // std::cout<<" + + + + + +"<<std::endl;
    // std::cout<<(zl_.transpose() - zu_.transpose() - g_.transpose())<<std::endl;
    return true;
}
bool_t Jyoptimizer::ShowZLU()
{
    std::cout << " - - - ZL - - - " << std::endl;
    std::cout << zl_ << std::endl;
    std::cout << " - - - ZU - - - " << std::endl;
    std::cout << zu_ << std::endl;
    return true;
}
MatrixX Jyoptimizer::pinv(const MatrixX& a, const float64_t tol )
{
    Eigen::JacobiSVD<MatrixX> svd_holder(a, Eigen::ComputeThinU | Eigen::ComputeThinV);

    MatrixX u = svd_holder.matrixU();
    MatrixX v = svd_holder.matrixV();
    MatrixX d = svd_holder.singularValues();

    MatrixX s = MatrixX::Zero(v.cols(), u.cols());
    for (uint32_t i = 0; i < d.size(); i++)
    {
        if (d(i, 0) > tol)
        {
            s(i, i) = 1 / d(i, 0);
        }
    }
    return v * s * u.transpose();
}

}  // namespace JYOPT