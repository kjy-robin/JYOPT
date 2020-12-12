/*
 * Copyright (C) KOUJIAYU Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file optimizer_application.h
 * @brief describe optimizer application
 * @author koujiayu(robink9611@gmail.com)
 * @date 2020-12-11
 */

#ifndef _JYOPT_OPTIMIZER_APPLICATION_H_
#define _JYOPT_OPTIMIZER_APPLICATION_H_

#include <jyopt/common/headers.h>
#include <jyopt/optimizer/optimizer_param.h>
#include <jyopt/question/question_base.h>

namespace JYOPT
{
class OptimizerApplication
{
public:
    OptimizerApplication() = default;

    ~OptimizerApplication() = default;

    bool_t OptimalSolver(std::shared_ptr<QuestionBase> ptr);

public:
    void Show_X(const EVector& x);
    void Show_Residual(const EVector& r);
    void Show_Jacobian(const EMatrix& j);
    void Show_Theta(const float64_t& th);
    void Show_Lambda(const EVector& lam);
    void Show_Objective_Gradient(const EVector& grad);
    void Show_ZLU(const EVector& zl, const EVector& zu);

private:
    bool_t Calc_Residual(std::shared_ptr<QuestionBase> ptr,
                         const EVector&                x,
                         const EVector&                s,
                         const EVector&                bl,
                         const EVector&                bu,
                         EVector&                      residual);

    bool_t Check_Bound_Info(const EVector& xl, const EVector& xu, const EVector& bl, const EVector& bu);

    bool_t Move_X_AND_S_Feasible(EVector&       x,
                                 const EVector& xl,
                                 const EVector& xu,
                                 EVector&       s,
                                 const EVector& sl,
                                 const EVector& su);

    bool_t UpdateZLU(const EVector& x,
                     const EVector& xl,
                     const EVector& xu,
                     const EVector& s,
                     const EVector& sl,
                     const EVector& su,
                     EVector&       zl,
                     EVector&       zu);

    EMatrix Pinv(const EMatrix& a, const float64_t tol = 1e-10);

    bool_t Calc_Jacobian_Augmentation(std::shared_ptr<QuestionBase> ptr,
                                      const EVector&                x,
                                      const EVector&                bl,
                                      const EVector&                bu,
                                      const EVector&                s,
                                      EMatrix&                      res);

    bool_t
    Calc_Gradient_Augmentation(std::shared_ptr<QuestionBase> ptr, const EVector& x, const EVector& s, EVector& grad);

    bool_t Calc_Hessian_Augmentation(std::shared_ptr<QuestionBase> ptr,
                                     const EVector&                x,
                                     const uint32_t&               n,
                                     const uint32_t&               m,
                                     const uint32_t                s_num,
                                     const EVector&                lambda,
                                     EMatrix&                      hess);

    float64_t Calc_Theta(std::shared_ptr<QuestionBase> ptr,
                         const EVector&                x,
                         const EVector&                s,
                         const EVector&                bl,
                         const EVector&                bu);

    float64_t Calc_Phi(std::shared_ptr<QuestionBase> ptr,
                       const EVector&                x,
                       const EVector&                xl,
                       const EVector&                xu,
                       const EVector&                s,
                       const EVector&                bl,
                       const EVector&                bu);

    bool_t Make_Diag(const EVector& v, EMatrix& m);

private:
    OptimizerParam param_;
};
}  // namespace JYOPT
#endif