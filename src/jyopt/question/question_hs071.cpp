/*
 * Copyright (C) KOUJIAYU Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

#include <assert.h>

#include <jyopt/common/macro.h>
#include <jyopt/question/question_hs071.h>

namespace JYOPT
{
bool_t QuestionHs::Get_Question_Info(uint32_t& n, uint32_t& m)
{
    n = 4;
    m = 2;
    return true;
}

bool_t
QuestionHs::Get_Bound_Info(const uint32_t& n, EVector& xl, EVector& xu, const uint32_t& m, EVector& bl, EVector& bu)
{

    xl = EVector::Zero(n);
    xl << 1, 1, 1, 1;

    xu = EVector::Zero(n);
    xu << 5, 5, 5, 5;

    bl = EVector::Zero(m);
    bl << 25, 40;

    bu = EVector::Zero(m);
    bu << 1e10, 40;

    return true;
}

bool_t QuestionHs::Get_StartPoint_Info(const uint32_t& n, EVector& x_init)
{
    UNREFERENCE_PARAM(n);
    x_init = EVector::Zero(4);
    x_init << 2, 1, 1, 2;
    return true;
}

bool_t QuestionHs::Calc_Objective_Function_Value(const uint32_t& n, const EVector& x, float64_t& obj_val)
{
    assert(x.size() == n);

    obj_val = x(0) * x(3) * (x(0) + x(1) + x(2)) + x(2);

    return true;
}

bool_t QuestionHs::Calc_Objective_Function_Gradient_Matrix(const uint32_t& n, const EVector& x, EVector& obj_grad)
{
    assert(x.size() == n);

    obj_grad = EVector::Zero(n);

    obj_grad(0) = x(3) * (2 * x(0) + x(1) + x(2));
    obj_grad(1) = x(0) * x(3);
    obj_grad(2) = x(0) * x(3) + 1;
    obj_grad(3) = x(0) * (x(0) + x(1) + x(2));

    return true;
}
bool_t
QuestionHs::Calc_Constraint_Function_Value(const uint32_t& n, const EVector& x, const uint32_t& m, EVector& cons_val)
{
    assert(x.size() == n);

    cons_val = EVector::Zero(m);

    cons_val(0) = x(0) * x(1) * x(2) * x(3);
    cons_val(1) = x(0) * x(0) + x(1) * x(1) + x(2) * x(2) + x(3) * x(3);
    return true;
}

bool_t QuestionHs::Calc_Constraint_Function_Jacobian_Matrix(const uint32_t& n,
                                                            const EVector&  x,
                                                            const uint32_t& m,
                                                            EMatrix&        cons_jac)
{
    assert(x.size() == n);

    cons_jac = EMatrix::Zero(m, n);

    cons_jac(0, 0) = x(1) * x(2) * x(3);
    cons_jac(0, 1) = x(0) * x(2) * x(3);
    cons_jac(0, 2) = x(0) * x(1) * x(3);
    cons_jac(0, 3) = x(0) * x(1) * x(2);
    cons_jac(1, 0) = 2 * x(0);
    cons_jac(1, 1) = 2 * x(1);
    cons_jac(1, 2) = 2 * x(2);
    cons_jac(1, 3) = 2 * x(3);

    return true;
}

bool_t QuestionHs::Calc_Hessian_Matrix(const uint32_t& n,
                                       const EVector&  x,
                                       const uint32_t& m,
                                       const EVector&  lambda,
                                       EMatrix&        hess)
{
    assert(x.size() == n);
    assert(lambda.size() == m);

    hess = EMatrix::Zero(n, n);

    float64_t objfact = 1.0;

    hess(0, 0) = objfact * 2 * x(3) * lambda(1) * 2;
    hess(1, 0) = objfact * x(3) + lambda(0) * x(2) * x(3);
    hess(0, 1) = hess(1, 0);
    hess(1, 1) = lambda(1) * 2;
    hess(2, 0) = objfact * x(3) + lambda(0) * x(1) * x(3);
    hess(0, 2) = hess(2, 0);
    hess(2, 1) = lambda(0) * x(0) * x(3);
    hess(1, 2) = hess(2, 1);
    hess(2, 2) = lambda(1) * 2;
    hess(3, 0) = objfact * (2 * x(0) + x(1) + x(2)) + lambda(0) * x(1) * x(2);
    hess(0, 3) = hess(3, 0);
    hess(3, 1) = objfact * x(0) + lambda(0) * x(0) * x(2);
    hess(1, 3) = hess(3, 1);
    hess(3, 2) = objfact * x(0) + lambda(0) * x(0) * x(1);
    hess(2, 3) = hess(3, 2);
    hess(3, 3) = lambda(1) * 2;

    return true;
}
}  // namespace JYOPT