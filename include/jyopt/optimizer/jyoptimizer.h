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

protected:
    bool_t MatrixInit();

    MatrixX CalcResStub();

    MatrixX CalcRes();

    MatrixX CalcObjgrad();

    MatrixX CalcJacStub();

    MatrixX CalcJac();

    void UpdateZLU();

    void CheckXSFeasible();

private:
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

    BaseParam param_;
};
}  // namespace JYOPT
#endif