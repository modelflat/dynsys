import pyopencl as cl
import numpy
from time import perf_counter

from dynsys.LCE import dummyOption
from ifs_fractal import SCRIPT_DIR, IFS_SOURCE

test_kernel = r"""

kernel void solve_cubic_correctness(
    const global real* equations,
    global int* answer_incorrect
) {
    const int id = get_global_id(0);
    if (id + 1 == get_global_size(0)) {
        return;
    }
    
    equations += id;
    answer_incorrect += id;
    
    real tol = 1e-12;

    real2 a = vload2(0, equations);
    real2 c = vload2(1, equations);
    
    real2 roots[3];
    int err = solve_cubic(a, (real2)(0, 0), c, 1e-8, -1, roots);
    
    *answer_incorrect = 0;
    for (int i = 0; i < 3; ++i) {
        real2 root_opt[3];
        int err_opt = solve_cubic_optimized(a, (real2)(0, 0), c, 1e-8, i, root_opt);
        int root_idx = i;
        
        if (err_opt != err 
            || fabs(root_opt[root_idx].x - roots[i].x) > tol 
            || fabs(root_opt[root_idx].y - roots[i].y) > tol) {
            *answer_incorrect = 1;
            break;
        }
    }
}

kernel void solve_cubic_bench_seq_v1(
    const int n_iter,
    const global real* equation,
    global real* result
) {
    real2 c = vload2(1, equation);
    
    real2 dummy_acc;
    real2 root_opt[3];
    for (int i = 0; i < n_iter; ++i) {
        solve_cubic(root_opt[i % 3].yx, (real2)(0, 0), c, 1e-8, i % 3, root_opt);
        dummy_acc += root_opt[i % 3];
    }
    *result = dummy_acc.x;
}

kernel void solve_cubic_bench_seq_v2(
    const int n_iter,
    const global real* equation,
    global real* result
) {
    real2 a = vload2(0, equation);
    real2 c = vload2(1, equation);
    
    real2 dummy_acc;
    real2 root_opt[3];
    for (int i = 0; i < n_iter; ++i) {
        solve_cubic_optimized(root_opt[i % 3].yx, (real2)(0, 0), c, 1e-8, i % 3, root_opt);
        dummy_acc += root_opt[i % 3];
    }
    *result = dummy_acc.x;
}


"""

dev = cl.get_platforms()[0].get_devices()[0]
ctx = cl.Context(devices=[dev])
queue = cl.CommandQueue(ctx)
prg = cl.Program(ctx, IFS_SOURCE + test_kernel).build(options=["-I", SCRIPT_DIR + "/include", dummyOption()])


def run_bench(n, ver):
    inputs = ((numpy.random.random(size=(2,)) - 0.5) * 20).astype(numpy.float64)
    inputs_dev = cl.Buffer(ctx, cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_WRITE, hostbuf=inputs)
    outputs = numpy.empty((2,), dtype=numpy.float64)
    outputs_dev = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=outputs.nbytes)

    t = perf_counter()
    if ver == 1:
        prg.solve_cubic_bench_seq_v1(
            queue, (1,), None,
            numpy.int32(n),
            inputs_dev,
            outputs_dev
        )
        cl.enqueue_copy(queue, outputs, outputs_dev)
    elif ver == 2:
        prg.solve_cubic_bench_seq_v2(
            queue, (1,), None,
            numpy.int32(n),
            inputs_dev,
            outputs_dev
        )
        cl.enqueue_copy(queue, outputs, outputs_dev)
    t = perf_counter() - t
    return t


def run_bench_ver(ver):
    times = []

    n_runs = 200
    n_iter = 100000

    for i in range(10):
        # heatin' up!
        run_bench(n_iter // 10, ver=ver)

    for i in range(n_runs):
        times.append(run_bench(n_iter, ver=ver))

    avt = numpy.mean(times)
    std = numpy.std(times)
    print("Average time for {}: {:3f} s, std: {:3f}".format(ver, avt, std))
    print("\t(~{} equations per second)".format(int(n_iter / avt)))
    return avt, std


def run_correctness():
    n = 1000000
    print("Running correctness...")
    inputs = ((numpy.random.random(size=(n,)) - 0.5) * 20).astype(numpy.float64)
    inputs_dev = cl.Buffer(ctx, cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_WRITE, hostbuf=inputs)
    outputs = numpy.empty((n - 1,), dtype=numpy.int32)
    outputs_dev = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=outputs.nbytes)

    prg.solve_cubic_correctness(
        queue, (n,), None,
        inputs_dev, outputs_dev
    )

    cl.enqueue_copy(queue, outputs, outputs_dev)

    if outputs.any():
        a = inputs[numpy.array(outputs.nonzero())]
        c = inputs[numpy.array(outputs.nonzero()) + 1]

        ac = numpy.hstack((a.T, c.T))

        print(ac)

    assert not outputs.any(), "Incorrect results!"

    print("Everything is fine!")


with open("bench_results", "a") as f: pass # just in case this file is not present

with open("bench_results") as f:
    #print(f.read())
    pass

with open("bench_results", "a") as bench:
    run_correctness()
    avt1, std1 = run_bench_ver(1)
    avt2, std2 = run_bench_ver(2)
    bench.write("{} {} {} {}\n".format(avt1, std1, avt2, std2))

