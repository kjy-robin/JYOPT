/*
 * Copyright (C) KOUJIAYU Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file question_hs071.h
 * @brief  define the hs071 question ,which is a derived class
 * @author koujiayu(robink9611@gmail.com)
 * @date 2020-12-11
 */

#ifndef _JYOPT_QUESTION_HS071_H_
#define _JYOPT_QUESTION_HS071_H_

#include <jyopt/question/question_base.h>

namespace JYOPT
{
class QuestionHs : public QuestionBase
{
public:
    virtual bool_t Get_Question_Info(uint32_t& n, uint32_t& m) override;

    virtual bool_t
    Get_Bound_Info(const uint32_t& n, EVector& xl, EVector& xu, const uint32_t& m, EVector& gl, EVector& gu) override;

    virtual bool_t Get_StartPoint_Info(const uint32_t& n, EVector& x_init) override;

    virtual bool_t Calc_Objective_Function_Value(const uint32_t& n, const EVector& x, float64_t& obj_val) override;

    virtual bool_t Calc_Objective_Function_Gradient_Matrix(const uint32_t& n,
                                                           const EVector&  x,
                                                           EVector&        obj_grad) override;
    virtual bool_t
    Calc_Constraint_Function_Value(const uint32_t& n, const EVector& x, const uint32_t& m, EVector& cons_val) override;

    virtual bool_t Calc_Constraint_Function_Jacobian_Matrix(const uint32_t& n,
                                                            const EVector&  x,
                                                            const uint32_t& m,
                                                            EMatrix&        cons_jac) override;

    virtual bool_t Calc_Hessian_Matrix(const uint32_t& n,
                                       const EVector&  x,
                                       const uint32_t& m,
                                       const EVector&  lambda,
                                       EMatrix&        hess) override;
};
}  // namespace JYOPT
#endif