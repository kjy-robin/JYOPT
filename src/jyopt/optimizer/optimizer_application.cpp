/*
 * Copyright (C) KOUJIAYU Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

#include <jyopt/common/constants.h>
#include <jyopt/optimizer/optimizer_application.h>
#include <iostream>

namespace JYOPT
{
bool_t OptimizerApplication::OptimalSolver(std::shared_ptr<QuestionBase> ptr)
{
    // --- according question info init m n
    uint32_t opt_n;
    uint32_t opt_m;
    ptr->Get_Question_Info(opt_n, opt_m);

    // --- according start info init x
    EVector opt_x = EVector::Zero(opt_n);
    ptr->Get_StartPoint_Info(opt_n, opt_x);

    // --- according question bounds info init xl, xu, bl, bu.
    EVector opt_xl = EVector::Zero(opt_n);
    EVector opt_xu = EVector::Zero(opt_n);
    EVector opt_bl = EVector::Zero(opt_m);
    EVector opt_bu = EVector::Zero(opt_m);
    ptr->Get_Bound_Info(opt_n, opt_xl, opt_xu, opt_m, opt_bl, opt_bu);

    // --- according bound info init slack variable s, sl, su
    // calc slack variables num, which is equal to inequality constraints num
    EVector temp_sl = EVector::Zero(opt_m);
    EVector temp_su = EVector::Zero(opt_m);

    uint32_t opt_s_num = 0;
    for (uint32_t i = 0; i < opt_bl.size(); i++)
    {
        if (opt_bl(i) < opt_bu(i))
        {
            temp_sl(opt_s_num) = opt_bl(i);
            temp_su(opt_s_num) = opt_bu(i);
            opt_s_num++;
        }
    }

    EVector opt_s  = EVector::Zero(opt_s_num);
    EVector opt_sl = EVector::Zero(opt_s_num);
    EVector opt_su = EVector::Zero(opt_s_num);

    opt_sl = temp_sl.block(0, 0, opt_s_num, 1);
    opt_su = temp_su.block(0, 0, opt_s_num, 1);

    // --- initial residual
    EVector opt_r;
    Calc_Residual(ptr, opt_x, opt_s, opt_bl, opt_bu, opt_r);

    // --- initial full ones vector
    EVector opt_e = EVector::Ones(opt_n + opt_s_num);

    // --- initial equation slack variables
    uint32_t k = 0;
    for (uint32_t i = 0; i < opt_m; i++)
    {
        if (opt_bl(i) < opt_bu(i))
        {
            if (param_.slack_init_)
            {
                opt_s(k) = opt_r(i);
            }
            else
            {
                opt_s(k) = 0.01;
            }
        }
    }

    // --- check consistent bounds
    if (!Check_Bound_Info(opt_xl, opt_xu, opt_bl, opt_bu))
    {
        return false;
    }

    // --- Move x and slack variable to be feasible
    if (!Move_X_AND_S_Feasible(opt_x, opt_xl, opt_xu, opt_s, opt_sl, opt_su))
    {
        std::cout << "WARNING : move init x to feasible area " << std::endl;
    }

    // --- initial tau,which will used to update alpha

    float64_t opt_tau = std::min<float64_t>(param_.tau_max_, 100 * param_.mu_);

    // --- initial variable constraint multipliers zl / zu
    EVector opt_zl;
    EVector opt_zu;

    UpdateZLU(opt_x, opt_xl, opt_xu, opt_s, opt_sl, opt_su, opt_zl, opt_zu);

    // --- initial objective gradient matrix
    EVector temp_obj_grad;
    ptr->Calc_Objective_Function_Gradient_Matrix(opt_n, opt_x, temp_obj_grad);
    EVector opt_obj_grad               = EVector::Zero(opt_n + opt_s_num);
    opt_obj_grad.block(0, 0, opt_n, 1) = temp_obj_grad;

    // --- initial constraints jacobian matrix
    EMatrix temp_cons_jac;
    ptr->Calc_Constraint_Function_Jacobian_Matrix(opt_n, opt_x, opt_m, temp_cons_jac);
    EMatrix opt_cons_jac;
    if (opt_s_num == 0)
    {
        opt_cons_jac = temp_cons_jac;
    }
    else
    {
        opt_cons_jac                           = EMatrix::Zero(opt_m, opt_n + 1);
        opt_cons_jac.block(0, 0, opt_m, opt_n) = temp_cons_jac;
        for (uint32_t i = 0; i < opt_m; i++)
        {
            if (opt_bl(i) < opt_bu(i))
            {
                opt_cons_jac(i, opt_n) = -1;
            }
        }
    }

    // --- initial lambda
    EMatrix temp_lambda =
        Pinv(opt_cons_jac * opt_cons_jac.transpose()) * opt_cons_jac * (opt_zl - opt_zu - opt_obj_grad);

    EVector opt_lambda = temp_lambda;

    return true;
}
bool_t OptimizerApplication::Calc_Residual(std::shared_ptr<QuestionBase> ptr,
                                           const EVector&                x,
                                           const EVector&                s,
                                           const EVector&                bl,
                                           const EVector&                bu,
                                           EVector&                      residual)
{
    ptr->Calc_Constraint_Function_Value(x.size(), x, bl.size(), residual);

    uint32_t j = 0;
    for (uint32_t i = 0; i < residual.size(); i++)
    {
        if (bu(i) == bl(i))
        {
            residual(i) = residual(i) - bl(i);
        }
        else
        {
            residual(i) = residual(i) - s(j);
            j += 1;
        }
    }

    return true;
}
bool_t
OptimizerApplication::Check_Bound_Info(const EVector& xl, const EVector& xu, const EVector& bl, const EVector& bu)
{
    for (uint32_t i = 0; i < xl.size(); i++)
    {
        if (xl(i) > xu(i))
        {
            std::cout << "ERROR : xl > xu in :\t" << i << std::endl;
            return false;
        }
    }

    for (uint32_t i = 0; i < bl.size(); i++)
    {
        if (bl(i) > bu(i))
        {
            std::cout << "ERROR : bl > bu :\t" << i << std::endl;
            return false;
        }
    }
    return true;
}

bool_t OptimizerApplication::Move_X_AND_S_Feasible(EVector&       x,
                                                   const EVector& xl,
                                                   const EVector& xu,
                                                   EVector&       s,
                                                   const EVector& sl,
                                                   const EVector& su)
{
    bool_t all_variable_is_legal = true;

    for (uint32_t i = 0; i < x.size(); i++)
    {
        if (x(i) <= xl(i))
        {
            x(i)                  = std::min<float64_t>(xl(i) + ALMOST_ZERO, xu(i));
            all_variable_is_legal = false;
        }

        if (x(i) >= xu(i))
        {
            x(i)                  = std::max<float64_t>(xu(i) - ALMOST_ZERO, xl(i));
            all_variable_is_legal = false;
        }
    }

    for (uint32_t i = 0; i < s.size(); i++)
    {
        if (s(i) <= sl(i))
        {
            s(i) = std::min<float64_t>(su(i), sl(i) + ALMOST_ZERO);
        }

        if (s(i) >= su(i))
        {
            s(i) = std::max<float64_t>(sl(i), su(i) - ALMOST_ZERO);
        }
    }
    return all_variable_is_legal;
}

bool_t OptimizerApplication::UpdateZLU(const EVector& x,
                                       const EVector& xl,
                                       const EVector& xu,
                                       const EVector& s,
                                       const EVector& sl,
                                       const EVector& su,
                                       EVector&       zl,
                                       EVector&       zu)
{
    zl = EVector::Zero(x.size() + s.size());
    zu = EVector::Zero(x.size() + s.size());

    for (uint32_t i = 0; i < x.size(); i++)
    {
        zl(i) = param_.mu_ / (x(i) - xl(i));
        zu(i) = param_.mu_ / (xu(i) - x(i));
    }

    for (uint32_t i = 0; i < s.size(); i++)
    {
        zl(i + x.size()) = param_.mu_ / (s(i) - sl(i));
        zu(i + x.size()) = param_.mu_ / (su(i) - s(i));
    }
    return true;
}
EMatrix OptimizerApplication::Pinv(const EMatrix& a, const float64_t tol)
{
    Eigen::JacobiSVD<EMatrix> svd_holder(a, Eigen::ComputeThinU | Eigen::ComputeThinV);

    EMatrix u = svd_holder.matrixU();
    EMatrix v = svd_holder.matrixV();
    EMatrix d = svd_holder.singularValues();

    EMatrix s = EMatrix::Zero(v.cols(), u.cols());

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