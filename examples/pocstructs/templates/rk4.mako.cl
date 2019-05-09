// $${vars_t}
// $${params_t}
// $${equations}
// $${n_systems}
// {step_size}
// functions
// {assign(dst, sign, src)}
// {call_equations(time_var, left_side_mod, right_side_mod)}

void rk4(
    ${vars_t}   y[${n_systems}], //
    ${params_t} p                //
) {
    ${vars_t} k[${n_systems}];
    ${vars_t} r[${n_systems}];

    // for (int i = 0; i < steps; ++i)
    {
        for (int j = 0; j < ${n_systems}; ++j) {
            ${assign("k[j]", "=", "y[j]")};
        }

        // k1
        // fn_System(t, k, p);
        ${call_equations("t", "k", "p")}
        //
        for (int j = 0; j < ${n_systems}; ++j) {
            // k[j] *= h / R(2.0);
            // r[j] = k[j] / R(3.0);
            // k[j] += y[j];
            ${assign("k[j]", "*=", "h / 2")}
            ${assign("r[j]", " =", "k[j]{ct} / 3")}
            ${assign("k[j]", "+=", "y[j]{ct}")}
        }

        t += h / 2;

        // k2
        // fn_System(t, k, p);
        ${call_equations("t", "k", "p")}
        //
        for (int j = 0; j < ${n_systems}; ++j) {
            // k[j] *= h;
            // r[j] += k[j] / R(3.0);
            // k[j] = y[j] + k[j] / R(2.0);
            ${assign("k[j]", "*=", "h")}
            ${assign("r[j]", "+=", "k[j]{ct} / 3")}
            ${assign("k[j]", " =", "y[j]{ct} + k[j]{ct} / 2")}
        }

        // k3
        // fn_System(t, k, p);
        ${call_equations("t", "k", "p")}
        //
        for (int j = 0; j < ${n_systems}; ++j) {
            // k[j] *= h;
            // r[j] += k[j] / R(3.0);
            // k[j] += y[j];
            ${assign("k[j]", "*=", "h")}
            ${assign("r[j]", "+=", "k[j]{ct} / 3")}
            ${assign("k[j]", "+=", "y[j]{ct}")}
        }

        t += h / 2;

        // k4
        // fn_System(t, k, p);
        ${call_equations("t", "k", "p")}
        //
        for (int j = 0; j < ${n_systems}; ++j) {
            // y[j] += r[j] + h * k[j] / R(6.0);
            ${assign("y[j]", "+=", "r[j]{ct} + h * k[j]{ct} / 6")}
        }
    }
}