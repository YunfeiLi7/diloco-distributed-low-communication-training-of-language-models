"""
DiLoCo: Distributed Low-Communication Training of Language Models

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - init_model_params
def init_model_params(input_dim, hidden_dim, output_dim, seed=0):
    # TODO: return dict with W1 (in,hid), b1 (hid,), W2 (hid,out), b2 (out,) as float64
    rng=np.random.default_rng(seed)
    W1=rng.normal(size=(input_dim,hidden_dim))* np.sqrt(2.0 / input_dim)
    W2=rng.normal(size=(hidden_dim,output_dim))* np.sqrt(2.0 / hidden_dim)
    b1=np.zeros((hidden_dim,)).astype(np.float64)
    b2=np.zeros((output_dim,)).astype(np.float64)
    return {
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2
    }
    pass

# Step 2 - relu
import numpy as np

def relu(x):
    # TODO: return an array of the same shape as x with negatives clipped to 0.
    return np.maximum(0.0,x)

# Step 3 - model_forward
import numpy as np

def model_forward(params, x):
    """Run the 2-layer MLP forward pass and stash intermediates for backprop."""
    # TODO: compute z1, h1 = relu(z1), logits, and return (logits, cache).
    z=x @ params['W1'] + params['b1']
    hidden = np.maximum(0.0,z)
    logits=hidden @ params['W2'] + params['b2']
    cache= {
        'h1': hidden,
        'logits': logits,
        'x':x,
        'z1': z
    }

    return logits,cache
    pass

# Step 4 - softmax
import numpy as np

def softmax(logits):
    # TODO: return a row-wise, numerically-stable softmax of logits with shape (N, C).
    M=np.max(logits,axis=-1,keepdims=True)
    shifted=logits-M 
    soft_logits=np.exp(shifted)/np.sum(np.exp(shifted),axis=-1,keepdims=True)
    return soft_logits
    pass

# Step 5 - cross_entropy_loss
def cross_entropy_loss(logits, labels):
    # TODO: Compute the mean cross-entropy loss between logits (N,C) and integer labels (N,).
    softmax_logits=softmax(logits)
    prob=softmax_logits[np.arange(len(labels)),labels]
    loss =  -np.sum(np.log(prob))/len(labels)
    return loss

# Step 6 - model_backward
def model_backward(params, cache, labels):
    x = cache["x"]
    z1 = cache["z1"]
    h1 = cache["h1"]
    logits = cache["logits"]

    N = labels.shape[0]

    # 1. softmax + cross entropy 对 logits 的梯度
    probs = softmax(logits)

    dlogits = probs.copy()
    dlogits[np.arange(N), labels] -= 1.0
    dlogits /= N

    # 2. 输出层: logits = h1 @ W2 + b2
    dW2 = h1.T @ dlogits
    db2 = np.sum(dlogits, axis=0)

    # 3. 反传到隐藏层
    dh1 = dlogits @ params["W2"].T

    # 4. 通过 ReLU
    dz1 = dh1 * (z1 > 0)

    # 5. 输入层: z1 = x @ W1 + b1
    dW1 = x.T @ dz1
    db1 = np.sum(dz1, axis=0)

    return {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2,
    }

# Step 7 - init_adamw_state
def init_adamw_state(params):
    # TODO: Build the AdamW state dict with zeroed first/second moments and t=0.
    m = {}
    v = {}
    for key , value in params.items():
        m[key] = np.zeros_like(value)
        v[key] = np.zeros_like(value)
    
    return {
        'm':m,
        'v':v,
        't':0
    }
    pass

# Step 8 - update_adam_moments
def update_adam_moments(state, grads, beta1, beta2):
    # TODO: increment state['t'] and update first/second moment EMAs for each param key.
    m=state['m']
    v=state['v']
    t=state['t']
    t=t+1
    new_m={}
    new_v={}
    for key , value in m.items():
        new_m[key]=beta1*m[key] + (1-beta1)*grads[key]

    for key ,value in v.items():
        new_v[key]=beta2*v[key]+ (1-beta2)*grads[key]**2

    return {
        'm':new_m,
        'v':new_v,
        't':t
    }
    pass

# Step 9 - bias_correct_moments
def bias_correct_moments(state, beta1, beta2):
    # TODO: return (m_hat, v_hat) dicts with Adam bias-corrected moments at step state['t'].
    m_hat={}
    v_hat={}
    m=state['m']
    v=state['v']

    for key,value in m.items():
        m_hat[key]=m[key]/(1-beta1**state['t'])
    
    for key,value in v.items():
        v_hat[key]=v[key]/(1-beta2**state['t'])
    
    return m_hat,v_hat

    
    pass

# Step 10 - adam_param_step
def adam_param_step(params, m_hat, v_hat, lr, eps):
    # TODO: return new params updated by p - lr * m_hat / (sqrt(v_hat) + eps) elementwise.
    new_params={}
    for key in params:
        new_params[key]=(
            params[key]-lr*m_hat[key]/(np.sqrt(v_hat[key])+eps)
        )
    
    return new_params
    pass

# Step 11 - decoupled_weight_decay
import numpy as np

def decoupled_weight_decay(params, lr, weight_decay):
    # TODO: return a new params dict where each tensor is shrunk by AdamW's decoupled weight decay factor.
    new_param={}
    for key in params:
        new_param[key]=params[key]*(1-lr*weight_decay)
    
    return new_param

    pass

# Step 12 - clone_params
def clone_params(params):
    # TODO: return a new dict whose values are independent copies of the input arrays.
    new_params={}
    for key in params:
        new_params[key]=params[key]*1
    return new_params
    pass

# Step 13 - scale_params
def scale_params(params, scalar):
    # TODO: return a new dict where every array in params is multiplied by scalar.
    new_params={}
    for key in params:
        new_params[key]=params[key]*scalar
    return new_params

# Step 14 - subtract_params
def subtract_params(params_a, params_b):
    # TODO: return a new dict with params_a[k] - params_b[k] for each key.
    new_params={}
    for key in params_a:
        new_params[key]=params_a[key]-params_b[key]
    
    return new_params

# Step 15 - average_params
def average_params(params_list):
    # TODO: return a new dict whose value at each key is the element-wise mean across the input dicts.
    new_params={}
    for key in params_list[0]:
        new_params[key]=np.zeros_like(params_list[0][key])
        for params in params_list:
            new_params[key]+=params[key]

        new_params[key]=new_params[key]/len(params_list)
    
    return new_params

# Step 16 - iid_shard_dataset
def iid_shard_dataset(x, y, num_workers, seed=0):
    # TODO: partition (x, y) uniformly at random into num_workers disjoint IID shards.
    N=len(y)
    rng =np.random.default_rng(seed)
    indices=rng.permutation(N)
    shards=[]
    base_num=N//num_workers
    rest=N%num_workers

    pointer=0
    for i in range(num_workers):
        size=base_num + (1 if i<rest else 0)
        idx=indices[pointer:pointer+size]
        x_shard=x[idx]
        y_shard=y[idx]
        shards.append((x_shard,y_shard))
        pointer+=size

    return shards

# Step 17 - noniid_shard_dataset
def noniid_shard_dataset(x, y, num_workers, num_classes, seed=0):
    # TODO: partition (x, y) across workers so each worker owns a distinct subset of classes.
    rng=np.random.default_rng(seed)
    work_indices=[[]for _ in range(num_workers)]
    for i in range(len(y)):
        c=y[i]
        idx=c % num_workers
        work_indices[idx].append(i)

    shards=[]

    for i in range(num_workers):
        idx=np.array(work_indices[i])
        idx=rng.permutation(idx)
        x_=x[idx]
        y_=y[idx]
        shards.append((x_,y_))
    

    return shards

# Step 18 - sample_worker_batch
def sample_worker_batch(x_shard, y_shard, batch_size, rng):
    # TODO: Sample batch_size examples from (x_shard, y_shard) using rng and return (x_batch, y_batch).
    n=len(y_shard)
    replace=batch_size > n 


    idx= rng.choice(n,size=batch_size,replace=replace)
    
    x_=x_shard[idx]
    y_=y_shard[idx]

    return (x_,y_)

# Step 19 - local_train_step
def local_train_step(params, adam_state, x_batch, y_batch, lr, beta1, beta2, eps, weight_decay):
    # TODO: one AdamW update: forward, loss, backward, moment update, param step, weight decay.
    logits,cache=model_forward(params=params,x=x_batch)
    loss=cross_entropy_loss(logits=logits, labels=y_batch)
    grads=model_backward(params=params, cache=cache, labels=y_batch)
    adam_now_state=update_adam_moments(state=adam_state, grads=grads, beta1=beta1, beta2=beta2)
    m_hat,v_hat=bias_correct_moments(state=adam_now_state, beta1=beta1, beta2=beta2)
    new_params=adam_param_step(params=params, m_hat=m_hat, v_hat=v_hat, lr=lr, eps=eps)
    decay_params=decoupled_weight_decay(params=new_params, lr=lr, weight_decay=weight_decay)
    return decay_params , adam_now_state ,loss

# Step 20 - inner_train_worker
def inner_train_worker(params, x_shard, y_shard, num_inner_steps, batch_size, lr, beta1, beta2, eps, weight_decay, seed):
    # TODO: run num_inner_steps AdamW updates on this worker's shard from a copy of params
    rng=np.random.default_rng(seed)
    worker_params=clone_params(params=params)
    adam_state=init_adamw_state(params=worker_params)
    mean_loss=0.0
    for i in range(num_inner_steps):
        x_batch,y_batch=sample_worker_batch(x_shard=x_shard, y_shard=y_shard, batch_size=batch_size, rng=rng)
        worker_params , adam_state ,loss=local_train_step(params=worker_params, adam_state=adam_state, x_batch=x_batch, y_batch=y_batch, lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay)
        mean_loss+=loss
    mean_loss/=num_inner_steps
    return worker_params , mean_loss

# Step 21 - init_outer_optimizer
def init_outer_optimizer(params):
    # TODO: Build outer optimizer state with a zero momentum buffer matching params shapes.
    momentum={}
    for key,value in params.items():
        momentum[key]=np.zeros_like(params[key])

    return {
        "momentum":momentum
    }

# Step 22 - update_outer_momentum
import numpy as np

def update_outer_momentum(outer_state, outer_grad, momentum_coef):
    """Update Nesterov momentum buffer: m <- momentum_coef * m + outer_grad."""
    # TODO: for each key in outer_state['momentum'], set m[k] = momentum_coef * m[k] + outer_grad[k]
    new_momentum={}
    old_momentum=outer_state['momentum']
    for key in old_momentum:
        new_momentum[key]=momentum_coef*old_momentum[key]+outer_grad[key]

    return {
        "momentum":new_momentum
    }

# Step 23 - nesterov_param_update
def nesterov_param_update(params, outer_state, outer_grad, outer_lr, momentum_coef):
    # TODO: apply the Nesterov look-ahead step using the (already-updated) momentum buffer and outer_grad.
    new_params={}
    
    for key in params:
        direction=outer_grad[key]+momentum_coef*outer_state[key]
        new_params[key]=params[key]-outer_lr*direction

    return new_params

# Step 24 - compute_outer_gradient
def compute_outer_gradient(global_params, worker_params_list):
    # TODO: return global_params minus the elementwise mean of worker_params_list, key by key.
    avg_worker_params = average_params(worker_params_list)
    outer_grad = subtract_params(global_params, avg_worker_params)

    return outer_grad

# Step 25 - run_diloco_round
def run_diloco_round(global_params, outer_state, worker_shards, num_inner_steps, batch_size, inner_hparams, outer_lr, momentum_coef, seed):
    worker_param_list=[]
    worker_losses=[]

    for i , (x_shard,y_shard) in enumerate(worker_shards):
        worker_params, mean_loss = inner_train_worker(
            params=global_params,
            x_shard=x_shard,
            y_shard=y_shard,
            num_inner_steps=num_inner_steps,
            batch_size=batch_size,
            lr=inner_hparams["lr"],
            beta1=inner_hparams["beta1"],
            beta2=inner_hparams["beta2"],
            eps=inner_hparams["eps"],
            weight_decay=inner_hparams["weight_decay"],
            seed=seed + i
        )
        worker_param_list.append(worker_params)
        worker_losses.append(mean_loss)

    
    outer_grad=compute_outer_gradient(global_params, worker_param_list)
    
    outer_state = update_outer_momentum(
        outer_state,
        outer_grad,
        momentum_coef
    )

    # 4. Nesterov 更新全局模型
    new_global_params = nesterov_param_update(
        global_params,
        outer_state["momentum"],
        outer_grad,
        outer_lr,
        momentum_coef
    )
    return new_global_params, outer_state, worker_losses

# Step 26 - train_diloco
def train_diloco(init_params, worker_shards, num_rounds, num_inner_steps, batch_size, inner_hparams, outer_lr, momentum_coef, seed=0):
    global_params=clone_params(init_params)
    outer_state=init_outer_optimizer(global_params)
    history={
        'round_losses':[]
    }
    for run_idx in range(num_rounds):
        global_params, outer_state, worker_losses=run_diloco_round(global_params, outer_state, worker_shards, num_inner_steps, batch_size, inner_hparams, outer_lr, momentum_coef, seed)
        round_losses=float(np.mean(worker_losses))
        history['round_losses'].append(round_losses)

    return global_params , history

# Step 27 - train_synchronous_baseline (not yet solved)
# TODO: implement

# Step 28 - evaluate_loss
def evaluate_loss(params, x, y):
    # TODO: return the mean cross-entropy loss of the model on the held-out data (x, y).
    logits, _ =model_forward(params,x)
    loss = cross_entropy_loss(logits, y)
    return float(loss)

# Step 29 - classification_accuracy
def classification_accuracy(params, x, y):
    # TODO: return top-1 accuracy of the 2-layer MLP on (x, y) as a float in [0, 1].
    logits , _ =model_forward(params,x)
    y_max=np.argmax(logits,axis=-1)
    accuracy=np.sum(y_max==y)/len(y)
    return float(accuracy)

# Step 30 - communication_savings
def communication_savings(num_rounds, num_inner_steps, num_workers, param_count):
    # TODO: count scalars transmitted under DiLoCo vs a fully synchronous baseline and return the ratio.
    per_step=2*num_workers*param_count
    diloco_scalars=per_step*num_rounds
    sync_scalars=diloco_scalars*num_inner_steps
    ratio=float(1.0/num_inner_steps)
    savings_factor=num_inner_steps
    return {
        'diloco_scalars':diloco_scalars,
        'sync_scalars':sync_scalars,
        'ratio':ratio,
        'savings_factor':savings_factor
    }

