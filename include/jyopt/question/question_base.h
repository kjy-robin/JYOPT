/*
 * Copyright (C) KOUJIAYU Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file question_base.h
 * @brief  define the problem interface ProblemBase class ,which is a virtual base class
 * @author koujiayu(robink9611@gmail.com)
 * @date 2020-12-11
 */

#ifndef _JYOPT_QUESTION_BASE_H_
#define _JYOPT_QUESTION_BASE_H_

#include <jyopt/common/headers.h>

namespace JYOPT
{
class QuestionBase
{
public:
    /**
     * @brief      Get the question basic information.
     *
     * @param[out]  n  number of optimization variables
     * @param[out]  m  number of constraints
     *
     * @return     True if optimizer gets question info successful. False otherwise.
     */
    virtual bool_t Get_Question_Info(uint32_t& n, uint32_t& m) = 0;

    /**
     * @brief      Get variables and constraints bounds  information.
     *
     * @param[in]  n  number of optimization variables
     * @param[out] xl storage of variables lower bounds
     * @param[out] xu storage of variables upper bounds
     * @param[in]  m  number of constraints
     * @param[out] gl storage of constraints lower bounds
     * @param[out] gu storage of constraints upper bounds
     *
     * @return     True if optimizer gets bounds info successful. False otherwise.
     */
    virtual bool_t
    Get_Bound_Info(const uint32_t& n, EVector& xl, EVector& xu, const uint32_t& m, EVector& gl, EVector& gu) = 0;

    /**
     * @brief      Get the start point information.
     *
     * @param[in]  n  number of optimization variables
     * @param[out] x_init storage of variables init info
     *
     * @return     True if optimizer gets start point info successful. False otherwise.
     */
    virtual bool_t Get_StartPoint_Info(const uint32_t& n, EVector& x_init) = 0;

    /**
     * @brief Calc the objective function value
     *
     * @param[in]  n         number of optimization variables
     * @param[in]  x         the variable value
     * @param[out] obj_val the value of the objective function
     *
     * @return True or false
     */
    virtual bool_t Calc_Objective_Function_Value(const uint32_t& n, const EVector& x, float64_t& obj_val) = 0;

    /**
     * @brief Calc the objective function gradient matrix
     *
     * @param[in]  n        number of optimization
     * @param[in]  x        the variable value
     * @param[out] obj_grad the gradient matrix of the objective function
     *
     * @return True or false
     */
    virtual bool_t Calc_Objective_Function_Gradient_Matrix(const uint32_t& n, const EVector& x, EVector& obj_grad) = 0;

    /**
     * @brief Calc the constraints function value
     *
     * @param[in]  n        number of optimization
     * @param[in]  x        the variable value
     * @param[in]  m        number of constraints
     * @param[out] cons_val the constraints function value
     *
     * @return True or false
     */
    virtual bool_t
    Calc_Constraint_Function_Value(const uint32_t& n, const EVector& x, const uint32_t& m, EVector& cons_val) = 0;

    /**
     * @brief Calc the constraints jacobian matrix
     *
     * @param[in]  n        number of optimization
     * @param[in]  x        the variable value
     * @param[in]  m        number of constraints
     * @param[out] cons_jac the constraints function value
     *
     * @return True or false
     */
    virtual bool_t Calc_Constraint_Function_Jacobian_Matrix(const uint32_t& n,
                                                            const EVector&  x,
                                                            const uint32_t& m,
                                                            EMatrix&        cons_jac) = 0;

    /**
     * @brief Calc the hessian matrix
     *
     * @param[in]  n        number of optimization
     * @param[in]  x        the variable value
     * @param[in]  m        number of constraints
     * @param[in]  lambda
     * @param[out] hess     the hessain metrix
     *
     * @return True or false
     */
    virtual bool_t Calc_Hessian_Matrix(const uint32_t& n,
                                       const EVector&  x,
                                       const uint32_t& m,
                                       const EVector&  lambda,
                                       EMatrix&        hess) = 0;
};
}  // namespace JYOPT

#endif