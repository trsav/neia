

from utils import *
import os
from jax import vmap
import copy
from jaxopt import ScipyBoundedMinimize as bounded_solver
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
import uuid
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.optimize import minimize as minimize_mo

def bo(
    f,
    f_aq,
    problem_data,
):
    path = problem_data["file_name"]



    try:
        os.mkdir(path)
    except FileExistsError:
        shutil.rmtree(path)
        os.mkdir(path)
    
    data_path = path + "/res.json"

    sample_initial = problem_data["sample_initial"]
    gp_ms = problem_data["gp_ms"]

    x_bounds = f.bounds
    try:
        det_init = problem_data["deterministic_initial"]
    except:
        det_init = 'false'
    if det_init == 'true':
        lhs_key = 0 # key for deterministic initial sample for expectation over functions
        jnp_samples = lhs(jnp.array(x_bounds), sample_initial,lhs_key)
        samples = []
        for i in range(sample_initial):
            samples.append(list([float(s) for s in jnp_samples[i]]))
    elif det_init == 'false':
        samples = numpy_lhs(jnp.array(x_bounds), sample_initial)

    problem_data['deterministic_initial'] = str(det_init)

    data = {"data": []}

    for sample in samples:
        res = f(sample)
        run_info = {
            "id": str(uuid.uuid4()),
            "inputs": list(sample),
            "objective": res + np.random.normal(0,problem_data['noise'])
        }

        data["data"].append(run_info)


    save_json(data, data_path)


    data = read_json(data_path)
    for i in range(len(data['data'])):
        data['data'][i]['regret'] = (f.f_opt - jnp.max(jnp.array([data['data'][j]['objective'] for j in range(i+1)]))).item()

    # print(data)

    problem_data['f_opt'] = (f.f_opt)
    data["problem_data"] = problem_data
    save_json(data, data_path)

    iteration = len(data["data"]) - 1


    while len(data['data']) < problem_data['max_iterations']:
        
            
        start_time = time.time()
        data = read_json(data_path)
        inputs, outputs, cost = format_data(data)
        mean_outputs = np.mean(outputs)
        std_outputs = np.std(outputs)
        if std_outputs != 0:
            outputs = (outputs - mean_outputs) / std_outputs

        mean_inputs = np.mean(inputs, axis=0)
        std_inputs = np.std(inputs, axis=0)
        inputs = (inputs - mean_inputs) / std_inputs

        bounds = []
        for i in range(len(x_bounds)):
            lb = float((x_bounds[i][0] - mean_inputs[i]) / std_inputs[i])
            ub = float((x_bounds[i][1] - mean_inputs[i]) / std_inputs[i])

            bounds.append([lb,ub])

        d = len(inputs[0])
        f_best = np.max(outputs)
        gp = build_gp_dict(*train_gp(inputs, outputs, gp_ms,its=10000,noise=problem_data['noisy']))


        util_args = (gp, f_best)
        if problem_data['noisy'] == True:
            n_gps = problem_data['letham_gps']
            gp_list = []
            f_best_list = []
            key = jax.random.PRNGKey(0)
            for i in range(n_gps):
                mean,std = inference(gp,inputs)
                det_outputs = []
                for i in range(len(mean)):
                    normal = tfd.Normal(loc=mean[i],scale=std[i])
                    det_outputs.append(normal.sample(seed=key))
                    key,subkey = jax.random.split(key)
                det_outputs = np.array([det_outputs]).T
                new_gp = build_gp_dict(*train_gp(inputs,det_outputs,int(gp_ms/2),noise=False))
                gp_list.append(new_gp) 
                f_best_list.append(np.max(det_outputs))

            util_args = (gp_list, f_best_list)

        aq = vmap(f_aq, in_axes=(0, None))

        if problem_data['plot'] == True:
            fig,ax = plt.subplots(2,1,figsize=(10,6),constrained_layout=True,sharex=True)

            n_plot = 400
            x_plot = np.linspace(x_bounds[0][0],x_bounds[0][1],n_plot)
            # y_plot = f.eval_vector(x_plot)
            # ax[0].plot(x_plot,y_plot,c='k',label='True Function',ls='dashed')
            for dat in data['data']:
                ax[0].scatter(dat['inputs'],dat['objective'],c='k')

            if problem_data['acquisition_function'] == 'LETHAM':
                for gp in gp_list:
                    x_plot_gp = np.linspace(bounds[0][0],bounds[0][1],n_plot)
                    gp_m,gp_s = inference(gp,x_plot_gp)
                    gp_m = gp_m * std_outputs + mean_outputs
                    gp_s = gp_s * std_outputs
                    ax[0].plot(x_plot,gp_m,c='k')
                    x_data = gp['D'].X
                    y_data = gp['D'].y
                    # x_data = (x_data - mean_inputs) / std_inputs 
                    # y_data = (y_data - mean_outputs) / std_outputs
                    x_data = x_data * std_inputs + mean_inputs
                    y_data = y_data * std_outputs + mean_outputs
                    ax[0].scatter(x_data,y_data,c='k',marker='x')
                    ax[0].fill_between(x_plot,gp_m - 2*gp_s,gp_m + 2*gp_s,color='k',lw=0,alpha=0.1)

                v_EI = vmap(EI, in_axes=(0, None))
                for gp in gp_list:
                    aq_plot = v_EI(x_plot_gp,(gp,f_best))
                    ax[1].plot(x_plot[:len(aq_plot)],-np.array(aq_plot),c='k',alpha=0.5)

                aq_plot = aq(x_plot_gp,util_args)

                ax[1].plot(x_plot[:len(aq_plot)],np.array(aq_plot),c='k',ls='dashed')
                # ax[1].set_yscale('symlog')
                # set scale to normal
                ax[1].set_yscale('linear')
                ax[1].set_yscale('linear')


            else:
                x_plot_gp = np.linspace(bounds[0][0],bounds[0][1],n_plot)
                gp_m,gp_s = inference(gp,x_plot_gp)
                gp_m = gp_m * std_outputs + mean_outputs
                gp_s = gp_s * std_outputs
                ax[0].plot(x_plot,gp_m,c='k')
                ax[0].fill_between(x_plot,gp_m - 2*gp_s,gp_m + 2*gp_s,color='k',lw=0,alpha=0.2)
                aq_plot = aq(x_plot_gp,util_args)
                ax[1].plot(x_plot[:len(aq_plot)],-np.array(aq_plot),c='k',alpha=0.5)



            ax[0].legend(frameon=False)


            # fig.savefig('true_function.png',dpi=200)
            fig.savefig(path + '/plot_' + str(iteration) + '.png',dpi=200)

        # optimising the aquisition of inputs, disregarding fidelity
        print("Optimising utility function...")
        upper_bounds_single = jnp.array([b[1] for b in bounds])
        lower_bounds_single = jnp.array([b[0] for b in bounds])

        opt_bounds = (lower_bounds_single, upper_bounds_single)
        s_init = jnp.array(sample_bounds(bounds, 256))
        
        iteration += 1
        solver = bounded_solver(
            method="l-bfgs-b",
            jit=True,
            fun=f_aq,
            tol=1e-10,
        )

        def optimise_aq(s):
            res = solver.run(init_params=s, bounds=opt_bounds, args=util_args)
            aq_val = res.state.fun_val
            print('Iterating utility took: ', res.state.iter_num, ' iterations with value of ',aq_val)
            x = res.params
            return aq_val, x

        aq_vals = []
        xs = []
        for s in s_init:
            aq_val, x = optimise_aq(s)
            aq_vals.append(aq_val)
            xs.append(x)

        x_opt_aq = xs[jnp.argmin(jnp.array(aq_vals))]

        x_opt = x_opt_aq
        x_opt = list((np.array(x_opt) * std_inputs) + mean_inputs)
            
        print("Optimal Solution: ", x_opt)
        x_opt = [float(x_opt[i]) for i in range(len(x_opt))]

        f_eval =  f(x_opt)
        run_info = {
            "inputs": list(x_opt),
        }

        run_info["objective"] = f_eval + np.random.normal(0,problem_data['noise'])
        run_info["id"] = str(uuid.uuid4())

        regret = min((f.f_opt - f_eval),data['data'][-1]['regret'])
        if regret.__class__ != float:
            regret = regret.item()
        run_info["regret"] = regret

        data["data"].append(run_info)


        save_json(data, data_path)

