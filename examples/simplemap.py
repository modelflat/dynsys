



ctx = ...
arr1 = ...
arr2 = ...

out1 = cl_map(ctx, fn, (arr1, arr2))

user_fn = """


void user_fn() {

    ...
}


"""


fn = """

kernel a

{
    user_fn();
}

"""



src = user_fn + fn
prg = cl.Program(ctx, src).build()

prg.a(
    asdfsaf
)



