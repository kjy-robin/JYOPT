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
    OptimizerApplication();

    ~OptimizerApplication();

    bool_t OptimalSolver(std::shared_ptr<QuestionBase> ptr);

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

private:
    OptimizerParam param_;
};
}  // namespace JYOPT
#endif