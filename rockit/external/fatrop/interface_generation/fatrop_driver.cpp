#define CASADI_MAX_NUM_THREADS 1

/////////////////////////////////////////////////////////////
#include "ocp/StageOCPApplication.hpp"
#include <limits>
// #include "rockit_generated/casadi_codegen.h"
#include "../casadi_codegen.h"
double inf = std::numeric_limits<double>::infinity();
#include "../problem_information.h"
#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
using namespace fatrop;

fatrop::StageOCPRockit create()
{
#define INSTANTIATE_EVAL_CAS_GEN(FUNCTION_NAME) \
    EvalCasGen(&FUNCTION_NAME##_incref, &FUNCTION_NAME##_decref, &FUNCTION_NAME##_checkout, &FUNCTION_NAME##_release, &FUNCTION_NAME##_n_in, &FUNCTION_NAME##_n_out, &FUNCTION_NAME##_sparsity_in, &FUNCTION_NAME##_sparsity_out, &FUNCTION_NAME##_work, &FUNCTION_NAME)
    return fatrop::StageOCPRockit(
        MACRO_nu,
        MACRO_nx,
        MACRO_ngI,
        MACRO_ng,
        MACRO_ngF,
        MACRO_ng_ineqI,
        MACRO_ng_ineq,
        MACRO_ng_ineqF,
        MACRO_n_stage_params,
        MACRO_n_global_params,
        MACRO_K,
        INSTANTIATE_EVAL_CAS_GEN(BAbt),
        INSTANTIATE_EVAL_CAS_GEN(bk),
        INSTANTIATE_EVAL_CAS_GEN(RSQrqtI),
        INSTANTIATE_EVAL_CAS_GEN(rqI),
        INSTANTIATE_EVAL_CAS_GEN(RSQrqt),
        INSTANTIATE_EVAL_CAS_GEN(rqk),
        INSTANTIATE_EVAL_CAS_GEN(RSQrqtF),
        INSTANTIATE_EVAL_CAS_GEN(rqF),
        INSTANTIATE_EVAL_CAS_GEN(GgtI),
        INSTANTIATE_EVAL_CAS_GEN(gI),
        INSTANTIATE_EVAL_CAS_GEN(Ggt),
        INSTANTIATE_EVAL_CAS_GEN(g),
        INSTANTIATE_EVAL_CAS_GEN(GgtF),
        INSTANTIATE_EVAL_CAS_GEN(gF),
        INSTANTIATE_EVAL_CAS_GEN(GgineqIt),
        INSTANTIATE_EVAL_CAS_GEN(gineqI),
        INSTANTIATE_EVAL_CAS_GEN(Ggineqt),
        INSTANTIATE_EVAL_CAS_GEN(gineq),
        INSTANTIATE_EVAL_CAS_GEN(GgineqFt),
        INSTANTIATE_EVAL_CAS_GEN(gineqF),
        INSTANTIATE_EVAL_CAS_GEN(LI),
        INSTANTIATE_EVAL_CAS_GEN(Lk),
        INSTANTIATE_EVAL_CAS_GEN(LF),
        std::vector<double>(MACRO_bounds_L),
        std::vector<double>(MACRO_bounds_U),
        std::vector<double>(MACRO_stage_params),
        std::vector<double>(MACRO_global_params),
        std::vector<double>(MACRO_initial_u),
        std::vector<double>(MACRO_initial_x));
}

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

#ifndef CASADI_SYMBOL_EXPORT
#if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
#if defined(STATIC_LINKED)
#define CASADI_SYMBOL_EXPORT
#else
#define CASADI_SYMBOL_EXPORT __declspec(dllexport)
#endif
#elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
#define CASADI_SYMBOL_EXPORT __attribute__((visibility("default")))
#else
#define CASADI_SYMBOL_EXPORT
#endif
#endif
#ifdef __cplusplus
extern "C"
{
#endif

#define DENSE_SPARSITY_FMT(n_rows, n_cols)               \
    static casadi_int fmt[3 + n_cols + n_rows * n_cols]; \
    casadi_int count = 0;                                \
    fmt[count++] = n_rows;                               \
    fmt[count++] = n_cols;                               \
    for (casadi_int i = 0; i < n_cols + 1; i++)          \
    {                                                    \
        fmt[count++] = i * n_rows;                       \
    }                                                    \
    for (casadi_int i = 0; i < n_cols; i++)              \
    {                                                    \
        for (casadi_int j = 0; j < n_rows; j++)          \
        {                                                \
            fmt[count++] = j;                            \
        }                                                \
    }

    struct fatrop_driver_memory
    {
    };

    static int fatrop_driver_mem_counter = 0;
    static int fatrop_driver_unused_stack_counter = -1;
    static int fatrop_driver_unused_stack[CASADI_MAX_NUM_THREADS];
    static std::shared_ptr<StageOCPApplication> fatrop_appl_p[CASADI_MAX_NUM_THREADS];
    static std::vector<casadi_real> fatrop_x[CASADI_MAX_NUM_THREADS];
    static std::vector<casadi_real> fatrop_u[CASADI_MAX_NUM_THREADS];
    static std::vector<casadi_real> fatrop_stage_params[CASADI_MAX_NUM_THREADS];
    static std::vector<casadi_real> fatrop_global_params[CASADI_MAX_NUM_THREADS];

    CASADI_SYMBOL_EXPORT int fatrop_driver_init_mem(int mem);
    CASADI_SYMBOL_EXPORT casadi_int fatrop_driver(const casadi_real **arg, casadi_real **res, casadi_int *iw, casadi_real *w, int mem)
    {
        // copy global params
        for (int i = 0; i < MACRO_n_global_params; i++)
        {
            fatrop_global_params[mem][i] = arg[0][i];
        }
        // copy stage params
        for (int i = 0; i < MACRO_n_stage_params * MACRO_K; i++)
        {
            fatrop_stage_params[mem][i] = arg[1][i];
        }
        // set parameters
        // std::cout << "calling set params" << std::endl;
        fatrop_appl_p[mem]->set_params(fatrop_global_params[mem], fatrop_stage_params[mem]);
        // copy x0 to fatrop_x
        for (int i = 0; i < MACRO_K * MACRO_nx; i++)
        {
            fatrop_x[mem][i] = arg[2][i];
        }
        // copy u0 to fatrop_u
        for (int i = 0; i < (MACRO_K - 1) * MACRO_nu; i++)
        {
            fatrop_u[mem][i] = arg[3][i];
        }
        // set initial guess
        fatrop_appl_p[mem]->set_initial_x(fatrop_x[mem]);
        fatrop_appl_p[mem]->set_initial_u(fatrop_u[mem]);
        // call the solver
        fatrop_appl_p[mem]->optimize();
        // get x
        fatrop_appl_p[mem]->last_solution().get_x(fatrop_x[mem]);
        // // get u
        fatrop_appl_p[mem]->last_solution().get_u(fatrop_u[mem]);

        // // copy fatrop_x to res[0]
        if (res[0] != nullptr)
        {
            for (int i = 0; i < fatrop_x[mem].size(); i++)
            {
                res[0][i] = fatrop_x[mem][i];
            }
        }
        // // copy fatrop_u to res[1]
        if (res[1] != nullptr)
        {
            for (int i = 0; i < MACRO_nu * (MACRO_K - 1); i++)
            {
                res[1][i] = fatrop_u[mem][i];
            }
        }
        return 0;
    }

    CASADI_SYMBOL_EXPORT casadi_int fatrop_driver_n_in(void)
    {
        return 4;
    }

    CASADI_SYMBOL_EXPORT casadi_int fatrop_driver_n_out(void)
    {
        return 4;
    }

    CASADI_SYMBOL_EXPORT const casadi_int *fatrop_driver_sparsity_in(casadi_int i)
    {
        switch (i)
        {
        case 0:
        {
            DENSE_SPARSITY_FMT(MACRO_n_global_params, 1);
            return fmt;
        }
        case 1:
        {
            DENSE_SPARSITY_FMT(MACRO_n_stage_params, MACRO_K);
            return fmt;
        }
        case 2:
        {
            DENSE_SPARSITY_FMT(MACRO_nx, MACRO_K);
            return fmt;
        }
        case 3:
        {
            DENSE_SPARSITY_FMT(MACRO_nu, MACRO_K - 1);
            return fmt;
        }
        default:
            return 0;
        }
    }

    CASADI_SYMBOL_EXPORT const casadi_int *fatrop_driver_sparsity_out(casadi_int i)
    {
        switch (i)
        {
        case 0:
        {
            DENSE_SPARSITY_FMT(MACRO_nx, MACRO_K);
            return fmt;
        }
        case 1:
        {
            DENSE_SPARSITY_FMT(MACRO_nu, MACRO_K - 1);
            return fmt;
        }
        case 2:
        {
            DENSE_SPARSITY_FMT(0, 1);
            return fmt;
        }
        case 3:
        {
            DENSE_SPARSITY_FMT(0, 1);
            return fmt;
        }
        default:
            return 0;
        }
    }

    CASADI_SYMBOL_EXPORT const char *fatrop_driver_name_in(casadi_int i)
    {
        switch (i)
        {
        case 0:
            return "p_global";

        case 1:
            return "p_stage";

        case 2:
            return "x0";

        case 3:
            return "u0";

        default:
            return 0;
        }
    }

    CASADI_SYMBOL_EXPORT const char *fatrop_driver_name_out(casadi_int i)
    {
        switch (i)
        {
        case 0:
            return "x_opt";

        case 1:
            return "u_opt";

        case 2:
            return "v_opt";

        case 3:
            return "T_opt";

        default:
            return 0;
        }
    }

    CASADI_SYMBOL_EXPORT int fatrop_driver_work(casadi_int *sz_arg, casadi_int *sz_res, casadi_int *sz_iw, casadi_int *sz_w)
    {

        *sz_arg = 4, *sz_res = 4, *sz_iw = 0, *sz_w = 0;

        return 0;
    }

    // Alloc memory
    CASADI_SYMBOL_EXPORT int fatrop_driver_alloc_mem(void)
    {
        return fatrop_driver_mem_counter++;
    }

    // Clear memory
    CASADI_SYMBOL_EXPORT void fatrop_driver_free_mem(int mem)
    {
        fatrop_appl_p[mem] = nullptr;
        // fatrop_x[mem].resize(MACRO_nx * MACRO_K);
        // fatrop_u[mem].resize(MACRO_nu * (MACRO_K-1));
    }

    CASADI_SYMBOL_EXPORT int fatrop_driver_checkout(void)
    {
        int mid;
        if (fatrop_driver_unused_stack_counter >= 0)
        {
            return fatrop_driver_unused_stack[fatrop_driver_unused_stack_counter--];
        }
        else
        {
            if (fatrop_driver_mem_counter == CASADI_MAX_NUM_THREADS)
                return -1;
            mid = fatrop_driver_alloc_mem();
            if (mid < 0)
                return -1;
            if (fatrop_driver_init_mem(mid))
                return -1;
            return mid;
        }

        return fatrop_driver_unused_stack[fatrop_driver_unused_stack_counter--];
    }

    CASADI_SYMBOL_EXPORT void fatrop_driver_release(int mem)
    {
        fatrop_driver_unused_stack[++fatrop_driver_unused_stack_counter] = mem;
    }

    void set_option(const char *option, const double value)
    {
        if (fatrop_appl_p[0])
            fatrop_appl_p[0]->set_option(std::string(option), value);
        else std::cout << "option " << option << " not set - driver not initialized " << std::endl;
    }

    // Initialize memory
    CASADI_SYMBOL_EXPORT int fatrop_driver_init_mem(int mem)
    {
        fatrop_appl_p[mem] = std::make_shared<StageOCPApplication>(std::make_shared<StageOCPRockit>(create()));
        fatrop_appl_p[mem]->build();
        fatrop_x[mem].resize(MACRO_nx * MACRO_K);
        fatrop_u[mem].resize(MACRO_nu * (MACRO_K - 1));
        fatrop_stage_params[mem].resize(MACRO_n_stage_params * MACRO_K);
        fatrop_global_params[mem].resize(MACRO_n_global_params * 1);
        #include "after_init.c.in"
        return 0;
    }

#ifdef __cplusplus
}
#endif

#undef MACRO_nx
#undef MACRO_nu
#undef MACRO_ngI
#undef MACRO_ng
#undef MACRO_ngF
#undef MACRO_ng_ineqI
#undef MACRO_ng_ineq
#undef MACRO_ng_ineqF
#undef MACRO_n_stage_params
#undef MACRO_n_global_params
#undef MACRO_global_params
#undef MACRO_stage_params
#undef MACRO_K
#undef MACRO_initial_x
#undef MACRO_initial_u
#undef MACRO_bounds_L
#undef MACRO_bounds_U
#undef MACRO_sampler
#undef DENSE_SPARSITY_FMT