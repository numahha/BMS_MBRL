import numpy as np

epsilon=0.1

p1=0.9
p2=0.5 # if p2 = 1-p1, then no modelling error exists



transition_matrix = np.array([[[1.-p1,p1,0.], # (s1,u1,*)
                               [1-p2,0.0,p2] # (s1,u2,*)
                              ],
                             [[0.,1.-p1,p1], # (s2,u1,*)
                              [p1,1.-p1,0.0] # (s2,u2,*)
                              ],
                             [[0.,1.,0.], # (s3,u1,*)
                              [0.,1.,0.] # (s3,u2,*)
                              ]])
#policy_matrix_grand = np.array([ [1.,0.], # (s1,*)
#                           [1.,0.], # (s2,*)
#                           [1.,0.]  # (s3,*)
#                          ])
policy_matrix_grand = np.array([ [1.,0.], # (s1,*)
                           [1.,0.], # (s2,*)
                           [1.,0.]  # (s3,*)
                          ])


random_policy = np.ones((transition_matrix.shape[0],transition_matrix.shape[1]))*1./transition_matrix.shape[1]



from scipy.special import betaln, gammaln

def log_multinomial_marginal_likelihood(n,a):
    temp_p = - gammaln(n.sum()+a.sum()) + gammaln(a.sum())
    for i in range(n.shape[0]):
        temp_p += gammaln(a[i]+n[i]) - gammaln(a[i])
    return temp_p


def log_flexible_model_marginal_likelihood(sampling_matrix):
    temp_p=0.
    #for i in range(transition_matrix.shape[0]):
    for i in range(transition_matrix.shape[0]):
        for k in range(transition_matrix.shape[1]):
            temp_n = np.array([sampling_matrix[i,k,0],
                               sampling_matrix[i,k,1],
                               sampling_matrix[i,k,2]])
            temp_p += log_multinomial_marginal_likelihood(temp_n,np.array([1.,1.,1.]))
    return temp_p

def log_simple_model_marginal_likelihood(sampling_matrix):
    success_sum_for_simple_model = sampling_matrix[0,0,1] + sampling_matrix[0,1,0] + sampling_matrix[1,0,2] + sampling_matrix[1,1,0]
    fail_sum_for_simple_model = sampling_matrix[0,0,0] + sampling_matrix[0,1,2] + sampling_matrix[1,0,1] + sampling_matrix[1,1,1]
    temp_n = np.array([success_sum_for_simple_model, fail_sum_for_simple_model])
    return log_multinomial_marginal_likelihood(temp_n,np.array([1.,1.]))






def sim_action_selection(eps_func, iter_num=50000000):
    si=0
    sampling_matrix = np.zeros((transition_matrix.shape[0],transition_matrix.shape[1],transition_matrix.shape[0]))
    diff_ml=[]
    sample_num_list=[]

    for t in range(iter_num):

        #epsilon = ((1.+t)**(-1./4))
        epsilon = eps_func(t)
        policy_matrix = (1.-epsilon)*policy_matrix_grand + epsilon*random_policy

        uk=0
        if np.random.rand() > policy_matrix[si,0]:
            uk=1

        p = np.random.rand()
        sj=0
        if p > transition_matrix[si,uk,0]:
            sj=1
            if p > (transition_matrix[si,uk,0]+transition_matrix[si,uk,1]):
                sj=2

        sampling_matrix[si,uk,sj] += 1
        si=sj


        if 0==(t%(iter_num/(10))) and t>0:
            print(t*100/iter_num,"%")
        if 0==(t%(iter_num/(1000))) and t>0:
            #print(epsilon,t)
            diff_ml.append(-((log_simple_model_marginal_likelihood(sampling_matrix) - log_flexible_model_marginal_likelihood(sampling_matrix))))
            sample_num_list.append(t+1)
    return sample_num_list, diff_ml

'''
si=0
sampling_matrix = np.zeros((transition_matrix.shape[0],transition_matrix.shape[1],transition_matrix.shape[0]))
diff_ml2=[]

for t in range(iter_num):

    epsilon = ((1.+t)**(-1./2))
    policy_matrix = (1.-epsilon)*policy_matrix_grand + epsilon*random_policy

    uk=0
    if np.random.rand() > policy_matrix[si,0]:
        uk=1

    p = np.random.rand()
    sj=0
    if p > transition_matrix[si,uk,0]:
        sj=1
        if p > (transition_matrix[si,uk,0]+transition_matrix[si,uk,1]):
            sj=2

    sampling_matrix[si,uk,sj] += 1
    si=sj

    if 0==(t%(iter_num/(1000))) and t>0:
        print(epsilon,t)
        diff_ml2.append(((log_simple_model_marginal_likelihood(sampling_matrix) - log_flexible_model_marginal_likelihood(sampling_matrix))))
'''

def epsilon_decay_success(t):
    return ((1.+t)**(-1./4))
def epsilon_decay_fail(t):
    return ((1.+t)**(-1.))


trial_num=10
trial_1 =[]
trial_2 =[]

for i in range(trial_num):
    print(i)
    sample_num_list, diff_ml = sim_action_selection(epsilon_decay_success)
    trial_1.append(diff_ml)
    sample_num_list, diff_ml = sim_action_selection(epsilon_decay_fail)
    trial_2.append(diff_ml)


from matplotlib import pyplot as plt
plt.plot(sample_num_list, trial_1[0], color="C0", label="$\epsilon=t_s^{-1/4}$")
plt.plot(sample_num_list, trial_2[0], color="C1",label="$\epsilon=t_s^{-1}$")
for i in range(trial_num-1):
    plt.plot(sample_num_list, trial_1[i+1], color="C0")
    plt.plot(sample_num_list, trial_2[i+1], color="C1")
plt.xlabel("$t_s$",fontsize=18)
plt.ylabel("ln BF",fontsize=18, labelpad=-5)
plt.legend()
plt.savefig("fig_decay_eps.pdf")
plt.close()


plt.plot(np.log(np.array(sample_num_list)), trial_2[0], color="C1",label="$\epsilon=t_s^{-1}$")
for i in range(trial_num-1):
    plt.plot(np.log(np.array(sample_num_list)), trial_2[i+1], color="C1")
plt.xlabel("$\ln t_s$",fontsize=18)
plt.ylabel("ln BF",fontsize=18, labelpad=-5)
plt.legend()
plt.savefig("fig_log.pdf")
plt.close()


plt.plot(np.sqrt(np.array(sample_num_list)), trial_1[0], color="C0",label="$\epsilon=t_s^{-1/4}$")
for i in range(trial_num-1):
    plt.plot(np.sqrt(np.array(sample_num_list)), trial_1[i+1], color="C0")
plt.xlabel("$\sqrt{t_s}$",fontsize=18, labelpad=-2)
plt.ylabel("ln BF",fontsize=18, labelpad=-5)
plt.legend()
plt.savefig("fig_sqrt.pdf")
plt.close()

