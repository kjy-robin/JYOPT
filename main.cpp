#include <jyopt/common/headers.h>
#include <jyopt/question/question_hs071.h>
#include <iostream>

int main()
{
    JYOPT::uint32_t   m, n;
    JYOPT::QuestionHs hs;

    hs.Get_Question_Info(n, m);

    std::cout << "n :\t" << n << " m :\t" << m << std::endl;
    return 0;
}