/*
 * Copyright (C) KOUJIAYU Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file jyoptimizer.h
 * @brief describe jyoptimizer
 * @author koujiayu(robink9611@gmail.com)
 * @date 2020-12-05
 */

#ifndef _JYOPT_OPT_MAIN_H_
#define _JYOPT_OPT_MAIN_H_

#include <jyopt/optimizer/base_param.h>
#include <Eigen/Dense>

namespace JYOPT
{
using MatrixX = Eigen::MatrixXd;

class Jyoptimizer
{
public:
    bool_t OptCalcFunc();
    bool_t ShowX();
    bool_t ShowS();
    bool_t ShowG();
    bool_t ShowJ();
    bool_t ShowResidual();
    bool_t ShowLambda();
    bool_t ShowZLU();

protected:
    bool_t MatrixInit();

    MatrixX CalcResStub();

    MatrixX CalcRes();

    float64_t CalcObj();

    MatrixX CalcObjgrad();

    MatrixX CalcJacStub();

    MatrixX CalcJac();

    float64_t CalcTheta();

    float64_t CalcPhi();

    MatrixX CalcHess();

    void UpdateZLU();

    void CheckXSFeasible();

    MatrixX pinv(const MatrixX& a, const float64_t tol = 0.0);

private:
    float64_t tau_;

    float64_t th_;

    MatrixX x_;

    MatrixX xl_;

    MatrixX xu_;

    MatrixX bl_;

    MatrixX bu_;

    MatrixX s_;

    MatrixX sl_;

    MatrixX su_;

    MatrixX zl_;

    MatrixX zu_;

    MatrixX residual_;

    MatrixX g_;

    MatrixX j_;

    MatrixX lam_;

    BaseParam param_;
};
}  // namespace JYOPT
#endif