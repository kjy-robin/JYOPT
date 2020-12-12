#include <jyopt/common/headers.h>
#include <jyopt/optimizer/optimizer_application.h>
#include <jyopt/question/question_hs071.h>
#include <iostream>

int main()
{
    JYOPT::QuestionHs           hs;
    JYOPT::OptimizerApplication opt;
    opt.OptimalSolver(std::make_shared<JYOPT::QuestionHs>(hs));
    return 0;
}