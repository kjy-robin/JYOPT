/*
 * Copyright (C) KOUJIAYU Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file optimizer_param.h
 * @brief describe optimizer basic param and default value
 * @author koujiayu(robink9611@gmail.com)
 * @date 2020-12-05
 */

#ifndef _JYOPT_OPTIMIZER_PARAM_H_
#define _JYOPT_OPTIMIZER_PARAM_H_

#include <jyopt/common/headers.h>

namespace JYOPT
{
// This class define the necessary params ,which one will using in JYOPT calc
struct OptimizerParam
{
    /**
     * @brief initial merit function weighting of constraint residuals
     */
    float64_t nu_ = 1000.0;

    /**
     * @brief initial barrier term
     */
    float64_t mu_ = 10;

    /**
     * @brief update mu? - used for testing
     */
    bool_t mu_update_ = true;

    enum LINE_SEARCH_CRITERIA : uint32_t
    {
        REDUCTION_IN_MERIT_FUNCTION = 1,
        SIMPLE_CLIPPING             = 2,
        FILTER_METHOD               = 3,
    };

    LINE_SEARCH_CRITERIA line_search_ = SIMPLE_CLIPPING;

    enum SOLVING_FOR_Z : uint32_t
    {
        UPDATE_FROM_DIRECT_SOLVE_APPROACH = 1,
        UPDATE_EXPLICITLY_FROM_FUNCTION   = 2,
    };
    SOLVING_FOR_Z z_update_ = UPDATE_FROM_DIRECT_SOLVE_APPROACH;

    enum MATRIX_INVERSION : uint32_t
    {
        CONDENSED_SYMMETRIC_MATRIX = 1,
        FULL_UNSYMMETRIC_MATRIX    = 2,
    };
    MATRIX_INVERSION matrix_ = CONDENSED_SYMMETRIC_MATRIX;

    uint32_t max_iteration_num_ = 100;

    /**
     * @brief initialize slack variables to equation residual (true)
     */
    bool_t slack_init_ = true;

    // update tau
    float64_t tau_max_ = 0.01;

    /**
     * @brief solution error tolerance
     */

    float64_t e_tol_ = 1e-6;

    float64_t k_e_ = 10;

    float64_t k_soc_ = 0.99;

    /**
     * @brief check for new barrier problem
     */

    float64_t k_mu_ = 0.2;

    /**
     * @brief not currently used
     */
    float64_t gama_alpha_ = 0.05;
    float64_t s_th_       = 1.1;
    float64_t s_phi_      = 2.3;
    float64_t epsilon_    = 1;
    float64_t eta_phi_    = 1e-4;
};

}  // namespace JYOPT

#endif