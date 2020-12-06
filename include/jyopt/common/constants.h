/*
 * Copyright (C) KOUJIAYU Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file constants.h
 * @brief describe constants file
 * @author koujiayu(robink9611@gmail.com)
 * @date 2020-12-06
 */

#ifndef _JYOPT_CONSTANTS_H_
#define _JYOPT_CONSTANTS_H_

#include <jyopt/common/headers.h>
#include <limits>
namespace JYOPT
{
// using to avoid precision when compare two float numbers
// const float64_t ALMOST_ZERO = 10 * std::numeric_limits<float64_t>::epsilon();
const float64_t ALMOST_ZERO     = 1e-2;
const float64_t MAX_FLOAT64_NUM = 1e10;
}  // namespace JYOPT

#endif