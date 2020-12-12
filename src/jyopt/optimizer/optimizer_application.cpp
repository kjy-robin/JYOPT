/*
 * Copyright (C) KOUJIAYU Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

#include <jyopt/common/constants.h>
#include <jyopt/optimizer/optimizer_application.h>
#include <cmath>
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
    EVector opt_obj_grad;
    Calc_Gradient_Augmentation(ptr, opt_x, opt_s, opt_obj_grad);

    // --- initial constraints jacobian matrix
    EMatrix opt_cons_jac;
    Calc_Jacobian_Augmentation(ptr, opt_x, opt_bl, opt_bu, opt_s, opt_cons_jac);

    // --- initial lambda
    EMatrix temp_lambda =
        Pinv(opt_cons_jac * opt_cons_jac.transpose()) * opt_cons_jac * (opt_zl - opt_zu - opt_obj_grad);

    EVector opt_lambda = temp_lambda;

    // --- initial parameters
    float64_t opt_alpha_pr = 1.0;
    float64_t opt_alpha_du = 1.0;

    // --- initial theta
    float64_t opt_th     = Calc_Theta(ptr, opt_x, opt_s, opt_bl, opt_bu);
    float64_t opt_th_max = 1e4 * std::max<float64_t>(1, opt_th);
    float64_t opt_th_min = 1e-4 * std::max<float64_t>(1, opt_th);

    // --- initial iteration count
    uint32_t opt_iter_count = 1;

    // * * * show * * *
    Show_X(opt_x);
    Show_Jacobian(opt_cons_jac);
    Show_Objective_Gradient(opt_obj_grad);
    Show_Lambda(opt_lambda);
    Show_Residual(opt_r);
    Show_Theta(opt_th);
    Show_ZLU(opt_zl, opt_zu);

    // std::cout << " + + + + + + + + + + + + + + + + + + + + " << std::endl;
    while (opt_iter_count <= param_.max_iteration_num_)
    {
        Calc_Residual(ptr, opt_x, opt_s, opt_bl, opt_bu, opt_r);

        opt_th = Calc_Theta(ptr, opt_x, opt_s, opt_bl, opt_bu);

        float64_t opt_phi = Calc_Phi(ptr, opt_x, opt_xl, opt_xu, opt_s, opt_bl, opt_bu);

        if (param_.z_update_ == 2)
        {
            UpdateZLU(opt_x, opt_xl, opt_xu, opt_s, opt_sl, opt_su, opt_zu, opt_zu);
        }

        // --- make diagonal matrices
        EMatrix opt_zml, opt_zmu;
        Make_Diag(opt_zl, opt_zml);
        Make_Diag(opt_zu, opt_zmu);

        // --- sigmas
        EVector opt_dl, opt_du;
        opt_dl = EVector::Zero(opt_n + opt_s_num);
        opt_du = EVector::Zero(opt_n + opt_s_num);

        opt_dl.block(0, 0, opt_n, 1)         = opt_x - opt_xl;
        opt_dl.block(opt_n, 0, opt_s_num, 1) = opt_s - opt_sl;

        opt_du.block(0, 0, opt_n, 1)         = opt_xu - opt_x;
        opt_du.block(opt_n, 0, opt_s_num, 1) = opt_su - opt_s;

        EMatrix opt_dml, opt_dmu;
        Make_Diag(opt_dl, opt_dml);
        Make_Diag(opt_du, opt_dmu);

        EMatrix opt_inv_dml = EMatrix::Zero(opt_n + opt_s_num, opt_n + opt_s_num);
        EMatrix opt_inv_dmu = EMatrix::Zero(opt_n + opt_s_num, opt_n + opt_s_num);
        EMatrix opt_sigl    = EMatrix::Zero(opt_n + opt_s_num, opt_n + opt_s_num);
        EMatrix opt_sigu    = EMatrix::Zero(opt_n + opt_s_num, opt_n + opt_s_num);

        for (uint32_t i = 0; i < opt_n + opt_s_num; i++)
        {
            opt_inv_dml(i, i) = 1 / opt_dml(i, i);
            opt_inv_dmu(i, i) = 1 / opt_dmu(i, i);
            opt_sigl(i, i)    = opt_zml(i, i) / opt_dml(i, i);
            opt_sigu(i, i)    = opt_zmu(i, i) / opt_dmu(i, i);
        }

        // --- calc 1st and 2nd derivatives
        Calc_Jacobian_Augmentation(ptr, opt_x, opt_bl, opt_bu, opt_s, opt_cons_jac);
        EMatrix opt_w;
        Calc_Hessian_Augmentation(ptr, opt_x, opt_n, opt_m, opt_s_num, opt_lambda, opt_w);
        Calc_Gradient_Augmentation(ptr, opt_x, opt_s, opt_obj_grad);
        EMatrix opt_h = opt_w + opt_sigl + opt_sigu;

        // --- construct and solve linear system Ax=b
        /*
            symmetric, zl/zu explicitly solved
        */
        uint32_t temp_h_rows        = opt_h.rows();
        uint32_t temp_h_cols        = opt_h.cols();
        uint32_t temp_cons_jac_rows = opt_cons_jac.rows();
        uint32_t temp_cons_jac_cols = opt_cons_jac.cols();

        assert(temp_h_rows == temp_h_cols);
        assert(temp_h_cols == temp_cons_jac_cols);

        EMatrix opt_a1 = EMatrix::Zero(temp_h_rows + temp_cons_jac_rows, temp_h_cols + temp_cons_jac_rows);

        opt_a1.block(0, 0, temp_h_rows, temp_h_cols)                         = opt_h;
        opt_a1.block(temp_h_rows, 0, temp_cons_jac_rows, temp_cons_jac_cols) = opt_cons_jac;
        opt_a1.block(0, temp_h_cols, temp_cons_jac_cols, temp_cons_jac_rows) = opt_cons_jac.transpose();

        assert((opt_n + opt_s_num + opt_m) == (temp_h_rows + temp_cons_jac_rows));
        EVector opt_b1 = EVector::Zero(opt_n + opt_s_num + opt_m);
        opt_b1.block(0, 0, opt_n + opt_s_num, 1) =
            opt_obj_grad - opt_zl + opt_zu + opt_cons_jac.transpose() * opt_lambda + opt_zl -
            param_.mu_ * opt_inv_dml * opt_e - opt_zu + param_.mu_ * opt_inv_dmu * opt_e;

        opt_b1.block(opt_n + opt_s_num, 0, opt_m, 1) = opt_r;

        // --- calc search direction, solving Ax=b
        EVector opt_d1    = -1.0 * Pinv(opt_a1) * opt_b1;
        EVector opt_dx1   = opt_d1.block(0, 0, opt_n, 1);
        EVector opt_ds1   = opt_d1.block(opt_n, 0, opt_s_num, 1);
        EVector opt_dlam1 = opt_d1.block(opt_n + opt_s_num, 0, opt_m, 1);

        // --- compute search direction for z (explicit solution)
        EVector temp_x_s                       = EVector::Zero(opt_n + opt_s_num);
        temp_x_s.block(0, 0, opt_n, 1)         = opt_dx1;
        temp_x_s.block(opt_n, 0, opt_s_num, 1) = opt_ds1;

        EVector opt_dzl1 = param_.mu_ * opt_inv_dml * opt_e - opt_zl - opt_sigl * temp_x_s;
        EVector opt_dzu1 = param_.mu_ * opt_inv_dmu * opt_e - opt_zu + opt_sigu * temp_x_s;
        // +-+-+-+-+-+-+-+-+-+-+-+-
        // TODO:ADD search1
        // +-+-+-+-+-+-+-+-+-+-+-+-

        /*
            Un-symmetric, zl/zu implicitly solved
            construct A2
        */
        // uint32_t temp_w_rows  = opt_w.rows();
        // uint32_t temp_w_cols  = opt_w.cols();
        // uint32_t temp_a2_rows = temp_w_rows + temp_cons_jac_rows + 2 * (opt_n + opt_s_num);
        // uint32_t temp_a2_cols = temp_a2_rows;
        // EMatrix  opt_a2       = EMatrix::Zero(temp_a2_rows, temp_a2_cols);

        // opt_a2.block(0, 0, temp_w_rows, temp_w_cols)                         = opt_w;
        // opt_a2.block(temp_w_rows, 0, temp_cons_jac_rows, temp_cons_jac_cols) = opt_cons_jac;
        // opt_a2.block(0, temp_w_cols, temp_cons_jac_cols, temp_cons_jac_rows) = opt_cons_jac.transpose();

        // EMatrix temp_eye_n_s_sum = -1.0 * EMatrix::Identity(opt_n + opt_s_num, opt_n + opt_s_num);

        // opt_a2.block(0, temp_w_cols + temp_cons_jac_cols, opt_n + opt_s_num, opt_n + opt_s_num) = temp_eye_n_s_sum;
        // opt_a2.block(0, temp_w_cols + temp_cons_jac_cols + opt_n + opt_s_num, opt_n + opt_s_num, opt_n + opt_s_num) =
        //     temp_eye_n_s_sum;

        // opt_a2.block(temp_w_rows + temp_cons_jac_rows, 0, opt_n + opt_s_num, opt_n + opt_s_num) = opt_zml;
        // opt_a2.block(temp_w_rows + temp_cons_jac_rows, opt_n + opt_s_num + opt_m, opt_n + opt_s_num,
        //              opt_n + opt_s_num)                                                         = opt_dml;
        // opt_a2.block(temp_w_rows + temp_cons_jac_rows + opt_n + opt_s_num, 0, opt_n + opt_s_num, opt_n + opt_s_num) =
        //     -1.0 * opt_zmu;
        // opt_a2.block(temp_w_rows + temp_cons_jac_rows + opt_n + opt_s_num,
        //              temp_w_cols + temp_cons_jac_rows + opt_m + opt_n + opt_s_num, opt_n + opt_s_num,
        //              opt_n + opt_s_num) = opt_dmu;

        // construct B2

        // EVector opt_b2 = EVector::Zero(temp_a2_rows);
        // opt_b2.block(0, 0, opt_n + opt_s_num, 1) =
        //     opt_obj_grad - opt_zl + opt_zu + opt_cons_jac.transpose() * opt_lambda;
        // opt_b2.block(opt_n + opt_m, 0, temp_cons_jac_rows, 1) = opt_r;
        // opt_b2.block(opt_n + opt_m + temp_cons_jac_rows, 0, opt_n + opt_s_num, 1) =
        //     opt_dml * opt_zml * opt_e - param_.mu_ * opt_e;
        // opt_b2.block(2 * (opt_n + opt_s_num) + temp_cons_jac_rows, 0, opt_n + opt_s_num, 1) =
        //     opt_dmu * opt_zmu * opt_e - param_.mu_ * opt_e;

        // EVector opt_d2    = -1.0 * Pinv(opt_a2) * opt_b2;
        // EVector opt_dx2   = opt_d2.block(0, 0, opt_n, 1);
        // EVector opt_ds2   = opt_d2.block(opt_n, 0, opt_s_num, 1);
        // EVector opt_dlam2 = opt_d2.block(opt_n + opt_s_num, 0, opt_m, 1);
        // EVector opt_dzl2  = opt_d2.block(opt_n + opt_s_num + opt_m, 0, opt_n + opt_s_num, 1);
        // EVector opt_dzu2  = opt_d2.block(2 * (opt_n + opt_s_num) + opt_m, 0, opt_n + opt_s_num, 1);
        // +-+-+-+-+-+-+-+-+-+-+-+-
        // TODO:ADD search1
        // +-+-+-+-+-+-+-+-+-+-+-+-

        /*
         +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
         TODO:Test for positive definiteness
         1. A1 , all eigs of A1 is positive? each step det(A..) is positive
         2. A2 , all eigs of A2 is positive? each step det(A..) is positive
         +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
        */

        /*
            reduced or full matrix inversion
        */
        EVector opt_dx, opt_ds, opt_dlam, opt_dzl, opt_dzu;

        if (param_.matrix_ == 1)
        {
            opt_dx   = opt_dx1;
            opt_ds   = opt_ds1;
            opt_dlam = opt_dlam1;
            opt_dzl  = opt_dzl1;
            opt_dzu  = opt_dzu1;
        }
        else
        {
            // opt_dx   = opt_dx2;
            // opt_ds   = opt_ds2;
            // opt_dlam = opt_dlam2;
            // opt_dzl  = opt_dzl2;
            // opt_dzu  = opt_dzu2;
        }

        // compute acceptance point
        EVector opt_xa = opt_x + opt_alpha_pr * opt_dx;
        EVector opt_sa, opt_lama, opt_zla, opt_zua;

        if (opt_s_num > 0)
        {
            opt_sa = opt_s + opt_alpha_pr * opt_ds;
        }
        opt_lama = opt_lambda + opt_alpha_du * opt_dlam;

        switch (param_.z_update_)
        {
            case OptimizerParam::SOLVING_FOR_Z::UPDATE_FROM_DIRECT_SOLVE_APPROACH:
                opt_zla = opt_zl + opt_alpha_du * opt_dzl;
                opt_zua = opt_zu + opt_alpha_du * opt_dzu;
                break;
            case OptimizerParam::SOLVING_FOR_Z::UPDATE_EXPLICITLY_FROM_FUNCTION:

                // +-+-+-+-+-+-+-+-+-+-+-+-
                // TODO: Update explicitly from z = mu / x
                // +-+-+-+-+-+-+-+-+-+-+-+-

                break;
        }
        // --- max alpha is that which brings the search point to within "tau" of constraint
        // tau is 0 to 0.01
        float64_t opt_alpha_pr_max = 1.0;
        float64_t opt_alpha_du_max = 1.0;

        // check for constraint violations
        for (uint32_t i = 0; i < opt_n; i++)
        {
            if (opt_xa(i) < opt_xl(i))
            {
                opt_alpha_pr_max =
                    std::min<float64_t>(opt_alpha_pr_max, (opt_tau - 1) * (opt_x(i) - opt_xl(i)) / opt_dx(i));
            }

            if (opt_xa(i) > opt_xu(i))
            {
                opt_alpha_pr_max =
                    std::min<float64_t>(opt_alpha_pr_max, (1 - opt_tau) * (opt_xu(i) - opt_x(i)) / opt_dx(i));
            }
        }

        for (uint32_t i = 0; i < opt_s_num; i++)
        {
            if (opt_sa(i) < opt_sl(i))
            {
                opt_alpha_pr_max =
                    std::min<float64_t>(opt_alpha_pr_max, (opt_tau - 1) * (opt_s(i) - opt_sl(i)) / opt_ds(i));
            }

            if (opt_sa(i) > opt_su(i))
            {
                opt_alpha_pr_max =
                    std::min<float64_t>(opt_alpha_pr_max, (1 + opt_tau) * (opt_su(i) - opt_s(i)) / opt_ds(i));
            }
        }

        for (uint32_t i = 0; i < opt_n; i++)
        {
            if (param_.z_update_ == 1)
            {
                if (opt_zla(i) < 0)
                {
                    opt_alpha_du_max =
                        std::min<float64_t>(opt_alpha_du_max, (opt_tau * opt_zl(i) - opt_zl(i)) / opt_dzl(i));
                }

                if (opt_zua(i) < 0)
                {
                    opt_alpha_du_max =
                        std::min<float64_t>(opt_alpha_du_max, (opt_tau * opt_zu(i) - opt_zu(i)) / opt_dzu(i));
                }
            }
        }

        // --- line search
        switch (param_.line_search_)
        {
            case OptimizerParam::LINE_SEARCH_CRITERIA::REDUCTION_IN_MERIT_FUNCTION:
                opt_alpha_du = opt_alpha_du_max;
                opt_alpha_pr = std::min<float64_t>(opt_alpha_pr, opt_alpha_pr_max);
                break;
            case OptimizerParam::LINE_SEARCH_CRITERIA::SIMPLE_CLIPPING:
                opt_alpha_pr = 1;
                opt_alpha_du = 1;
                break;
            case OptimizerParam::LINE_SEARCH_CRITERIA::FILTER_METHOD:
                opt_alpha_du = opt_alpha_du_max;
                opt_alpha_pr = std::min<float64_t>(opt_alpha_pr, opt_alpha_pr_max);
                break;
        }

        // --- After Line search step ,recalculate xa sa lama za

        opt_xa = opt_x + opt_alpha_pr * opt_dx;
        if (opt_s_num > 0)
        {
            opt_sa = opt_s + opt_alpha_pr * opt_ds;
        }
        opt_lama = opt_lambda + opt_alpha_pr * opt_dlam;  // This sentence may be wrong  need check opr? or du?

        switch (param_.z_update_)
        {
            case OptimizerParam::SOLVING_FOR_Z::UPDATE_FROM_DIRECT_SOLVE_APPROACH:
                opt_zla = opt_zl + opt_alpha_du * opt_dzl;
                opt_zua = opt_zu + opt_alpha_du * opt_dzu;
                break;

            case OptimizerParam::SOLVING_FOR_Z::UPDATE_EXPLICITLY_FROM_FUNCTION:
                // +-+-+-+-+-+-+-+-+-+-+-+-
                // TODO: Update explicitly from z = mu / x
                // +-+-+-+-+-+-+-+-+-+-+-+-
                break;
        }

        /*
         clipping (this should already be arranged by alpha_max)
         push away from boundary with tau
        */
        for (uint32_t i = 0; i < opt_n; i++)
        {
            if (opt_xa(i) < opt_xl(i))
            {
                opt_xa(i) = opt_xl(i) + opt_tau * (opt_x(i) - opt_xl(i));
            }

            if (opt_xa(i) > opt_xu(i))
            {
                opt_xa(i) = opt_xu(i) - opt_tau * (opt_xu(i) - opt_x(i));
            }

            if (opt_zla(i) < 0)
            {
                opt_zla(i) = opt_tau * opt_zl(i);
            }

            if (opt_zua(i) < 0)
            {
                opt_zua(i) = opt_tau * opt_zu(i);
            }
        }

        for (uint32_t i = 0; i < opt_s_num; i++)
        {
            if (opt_sa(i) < opt_sl(i))
            {
                opt_sa(i) = opt_sl(i) + opt_tau * (opt_s(i) - opt_sl(i));
            }

            if (opt_sa(i) > opt_su(i))
            {
                opt_sa(i) = opt_su(i) - opt_tau * (opt_su(i) - opt_s(i));
            }
        }

        // +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
        // TODO: Predicted reduction in merit function
        // +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-

        bool_t opt_ac = false;
        switch (param_.line_search_)
        {
            case OptimizerParam::LINE_SEARCH_CRITERIA::REDUCTION_IN_MERIT_FUNCTION:
                // TODO ;  merit function
                break;

            case OptimizerParam::LINE_SEARCH_CRITERIA::SIMPLE_CLIPPING:
                opt_ac       = true;
                opt_alpha_pr = 1;
                opt_alpha_du = 1;
                break;
            case OptimizerParam::LINE_SEARCH_CRITERIA::FILTER_METHOD:
                // TODO : check if acceptable point with filter method
                break;
        }
        // +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
        // TODO: testing for acceptance
        // opt_ac =true
        // second order correction
        // +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-

        if (opt_ac)
        {
            // --- accept point
            opt_x      = opt_xa;
            opt_s      = opt_sa;
            opt_lambda = opt_lama;
            opt_zl     = opt_zla;
            opt_zu     = opt_zua;

            // To DO: update filter
        }

        // --- check for convergence
        float64_t opt_s_max = 100;  // >1

        float64_t opt_s_d = std::max<float64_t>(
            opt_s_max, (opt_lambda.cwiseAbs().sum() + opt_zl.cwiseAbs().sum() + opt_zu.cwiseAbs().sum()) /
                           (opt_m + 2 * (opt_n + opt_s_num)));

        float64_t opt_s_c = std::max<float64_t>(opt_s_max, (opt_zl.cwiseAbs().sum() + opt_zu.cwiseAbs().sum()) /
                                                               (2 * (opt_n + opt_s_num)));
        //  part 1
        float64_t opt_e_mu = std::numeric_limits<float64_t>::lowest();
        EVector   temp_obj_grad;
        Calc_Gradient_Augmentation(ptr, opt_x, opt_s, temp_obj_grad);
        EMatrix temp_cons_jac;
        Calc_Jacobian_Augmentation(ptr, opt_x, opt_bl, opt_bu, opt_s, temp_cons_jac);
        EVector        temp_part_1 = temp_obj_grad + temp_cons_jac.transpose() * opt_lambda - opt_zl + opt_zu;
        EVector::Index useless_col, useless_row;
        float64_t      opt_part_1 = temp_part_1.cwiseAbs().maxCoeff(&useless_row, &useless_col) / opt_s_d;
        opt_e_mu                  = std::max<float64_t>(opt_e_mu, opt_part_1);

        //  part 2
        EVector temp_residual;
        Calc_Residual(ptr, opt_x, opt_s, opt_bl, opt_bu, temp_residual);
        float64_t opt_part_2 = temp_residual.cwiseAbs().maxCoeff(&useless_row, &useless_col);
        opt_e_mu             = std::max<float64_t>(opt_e_mu, opt_part_2);

        //  part 3
        EMatrix temp_m_x_s_part_3;
        EVector temp_v_x_s_part_3                       = EVector::Zero(opt_n + opt_s_num);
        temp_v_x_s_part_3.block(0, 0, opt_n, 1)         = opt_x - opt_xl;
        temp_v_x_s_part_3.block(opt_n, 0, opt_s_num, 1) = opt_s - opt_sl;
        Make_Diag(temp_v_x_s_part_3, temp_m_x_s_part_3);

        EMatrix temp_m_zl_part_3;
        Make_Diag(opt_zl, temp_m_zl_part_3);

        EVector   temp_part_3 = temp_m_x_s_part_3 * temp_m_zl_part_3 * opt_e;
        float64_t opt_part_3  = temp_part_3.cwiseAbs().maxCoeff(&useless_row, &useless_col) / opt_s_c;
        opt_e_mu              = std::max<float64_t>(opt_e_mu, opt_part_3);

        //  part 4
        EMatrix temp_m_x_s_part_4;
        EVector temp_v_x_s_part_4                       = EVector::Zero(opt_n + opt_s_num);
        temp_v_x_s_part_4.block(0, 0, opt_n, 1)         = opt_xu - opt_x;
        temp_v_x_s_part_4.block(opt_n, 0, opt_s_num, 1) = opt_su - opt_s;
        Make_Diag(temp_v_x_s_part_4, temp_m_x_s_part_4);

        EMatrix temp_m_zu_part_4;
        Make_Diag(opt_zu, temp_m_zu_part_4);

        EVector   temp_part_4 = temp_m_x_s_part_4 * temp_m_zu_part_4 * opt_e;
        float64_t opt_part_4  = temp_part_4.cwiseAbs().maxCoeff(&useless_row, &useless_col) / opt_s_c;
        opt_e_mu              = std::max<float64_t>(opt_e_mu, opt_part_4);

        // --- Check for termination conditions
        if (opt_e_mu <= param_.e_tol_)
        {
            std::cout << "* * * * * *Successful  Solution* * * * * *" << std::endl;
            std::cout <<opt_x<<std::endl;
            break;
        }

        // --- check for new barrier problem
        float64_t opt_k_mu = 0.2;

        if (param_.mu_update_)
        {
            if (opt_e_mu < opt_k_mu * param_.mu_)
            {
                float64_t opt_th_mu = 1.5;
                // --- update mu
                param_.mu_ = std::max<float64_t>(
                    param_.e_tol_ / 10, std::min<float64_t>(opt_k_mu * param_.mu_, pow(param_.mu_, opt_th_mu)));

                // --- update tau
                opt_tau = std::min<float64_t>(param_.tau_max_, 100 * param_.mu_);
                // TODO : refresh filter
            }
        }

        if (opt_iter_count == param_.max_iteration_num_)
        {
            std::cout << "Failed : max iterations" << std::endl;
            break;
        }
        opt_iter_count++;

        if (opt_ac)
        {
            opt_alpha_pr = 1.0;
            opt_alpha_du = 1.0;
        }
        else
        {
            opt_alpha_pr = opt_alpha_pr / 2.0;
            opt_alpha_du = opt_alpha_du / 2.0;
            if (opt_alpha_pr < 1e-4)
            {
                opt_alpha_pr = 1.0;
                opt_alpha_du = 1.0;
            }
        }
    }

    return true;
}

bool_t OptimizerApplication::Calc_Hessian_Augmentation(std::shared_ptr<QuestionBase> ptr,
                                                       const EVector&                x,
                                                       const uint32_t&               n,
                                                       const uint32_t&               m,
                                                       const uint32_t                s_num,
                                                       const EVector&                lambda,
                                                       EMatrix&                      hess)
{
    EMatrix temp_hess;
    ptr->Calc_Hessian_Matrix(n, x, m, lambda, temp_hess);

    hess                   = EMatrix::Zero(n + s_num, n + s_num);
    hess.block(0, 0, n, n) = temp_hess;
    return true;
}

bool_t OptimizerApplication::Calc_Gradient_Augmentation(std::shared_ptr<QuestionBase> ptr,
                                                        const EVector&                x,
                                                        const EVector&                s,
                                                        EVector&                      grad)
{
    uint32_t temp_n     = x.size();
    uint32_t temp_s_num = s.size();
    EVector  temp_obj_grad;

    ptr->Calc_Objective_Function_Gradient_Matrix(temp_n, x, temp_obj_grad);

    grad                        = EVector::Zero(temp_n + temp_s_num);
    grad.block(0, 0, temp_n, 1) = temp_obj_grad;
    return true;
}

float64_t OptimizerApplication::Calc_Theta(std::shared_ptr<QuestionBase> ptr,
                                           const EVector&                x,
                                           const EVector&                s,
                                           const EVector&                bl,
                                           const EVector&                bu)
{
    EVector temp_r;
    Calc_Residual(ptr, x, s, bl, bu, temp_r);
    return temp_r.cwiseAbs().sum();
}

bool_t OptimizerApplication::Calc_Jacobian_Augmentation(std::shared_ptr<QuestionBase> ptr,
                                                        const EVector&                x,
                                                        const EVector&                bl,
                                                        const EVector&                bu,
                                                        const EVector&                s,
                                                        EMatrix&                      res)
{
    uint32_t temp_n = x.size();
    uint32_t temp_m = bl.size();

    EMatrix temp_cons_jac;
    ptr->Calc_Constraint_Function_Jacobian_Matrix(temp_n, x, temp_m, temp_cons_jac);

    uint32_t temp_s_num = s.size();

    if (temp_s_num == 0)
    {
        res = temp_cons_jac;
    }
    else
    {
        res                             = EMatrix::Zero(temp_m, temp_n + 1);
        res.block(0, 0, temp_m, temp_n) = temp_cons_jac;
        for (uint32_t i = 0; i < temp_m; i++)
        {
            if (bl(i) < bu(i))
            {
                res(i, temp_n) = -1;
            }
        }
    }
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

float64_t OptimizerApplication::Calc_Phi(std::shared_ptr<QuestionBase> ptr,
                                         const EVector&                x,
                                         const EVector&                xl,
                                         const EVector&                xu,
                                         const EVector&                s,
                                         const EVector&                bl,
                                         const EVector&                bu)
{
    float64_t temp_n = x.size();
    float64_t temp_m = bl.size();
    float64_t phi;
    ptr->Calc_Objective_Function_Value(temp_n, x, phi);

    for (uint32_t i = 0; i < temp_n; i++)
    {
        phi -= param_.mu_ * (log(x(i) - xl(i)) + log(xu(i) - x(i)));
    }

    float64_t j = 0;
    for (uint32_t i = 0; i < temp_m; i++)
    {
        if (bu(i) > bl(i))
        {
            phi -= param_.mu_ * (log(s(j) - bl(i)) + log(bu(i) - s(j)));
        }
    }
    return phi;
}

bool_t OptimizerApplication::Make_Diag(const EVector& v, EMatrix& m)
{
    uint32_t size = v.size();
    m             = EMatrix::Zero(size, size);
    for (uint32_t i = 0; i < size; i++)
    {
        m(i, i) = v(i);
    }
    return true;
}

void OptimizerApplication::Show_X(const EVector& x)
{
    std::cout << " - - - X - - - " << std::endl;
    std::cout << x << std::endl;
}
void OptimizerApplication::Show_Residual(const EVector& r)
{
    std::cout << " - - - Residual - - - " << std::endl;
    std::cout << r << std::endl;
}
void OptimizerApplication::Show_Jacobian(const EMatrix& j)
{
    std::cout << " - - - Jacobian- - - " << std::endl;
    std::cout << j << std::endl;
}
void OptimizerApplication::Show_Theta(const float64_t& th)
{
    std::cout << " - - - Theta - - - " << std::endl;
    std::cout << th << std::endl;
}
void OptimizerApplication::Show_Lambda(const EVector& lam)
{
    std::cout << " - - - Lambda - - - " << std::endl;
    std::cout << lam << std::endl;
}
void OptimizerApplication::Show_Objective_Gradient(const EVector& grad)
{
    std::cout << " - - - Obj Grad - - - " << std::endl;
    std::cout << grad << std::endl;
}
void OptimizerApplication::Show_ZLU(const EVector& zl, const EVector& zu)
{
    std::cout << " - - - ZL - - - " << std::endl;
    std::cout << zl << std::endl;
    std::cout << " - - - ZU - - - " << std::endl;
    std::cout << zu << std::endl;
}

}  // namespace JYOPT