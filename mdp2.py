import numpy as np

#epsilon=.1

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

epsilon_list=[]
freq_list_00=[]
freq_list_01=[]
freq_list_10=[]
freq_list_11=[]
freq_list_20=[]
freq_list_21=[]
for epsilon in [1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10]:
    policy_matrix = (1.-epsilon)*policy_matrix_grand + epsilon*random_policy
    p_matrix = np.ones((transition_matrix.shape[0],transition_matrix.shape[0]))
    for i in range(transition_matrix.shape[0]):
        for j in range(transition_matrix.shape[0]):
            temp=0.
            for k in range(transition_matrix.shape[1]):
                temp+=transition_matrix[i,k,j]*policy_matrix[i,k]
            p_matrix[i,j] = temp

    eig_val, eig_vec = np.linalg.eig(p_matrix.T)

    eig_index=-1
    find_count=0
    for i in range(transition_matrix.shape[0]):
        if abs(1. - eig_val[i])<1.0e-12:
            eig_index=i
            find_count+=1
    distribution = eig_vec.T[eig_index].real/eig_vec.T[eig_index].real.sum()
    #print(distribution)

    #''' # debug of stationary distribution
    temp_mat=np.ones(transition_matrix.shape[0])
    for i in range(transition_matrix.shape[0]):
        temp=0.
        for j in range(transition_matrix.shape[0]):
            temp+=distribution[j]*p_matrix[j,i]
        temp_mat[i]=temp

    siuk_freq = (distribution*policy_matrix.T).T
    print(siuk_freq)

    sampling_matrix = np.ones((transition_matrix.shape[0],transition_matrix.shape[1],transition_matrix.shape[0]))
    for i in range(transition_matrix.shape[0]):
        for k in range(transition_matrix.shape[1]):
            for j in range(transition_matrix.shape[0]):
                sampling_matrix[i,k,j] = siuk_freq[i,k]*transition_matrix[i,k,j]

    success_sum_for_simple_model = sampling_matrix[0,0,1] + sampling_matrix[0,1,0] + sampling_matrix[1,0,2] + sampling_matrix[1,1,0]
    fail_sum_for_simple_model = sampling_matrix[0,0,0] + sampling_matrix[0,1,2] + sampling_matrix[1,0,1] + sampling_matrix[1,1,1]
    #success_sum_for_simple_model = sampling_matrix[0,0,1] + sampling_matrix[0,1,0] + sampling_matrix[1,0,2] + sampling_matrix[1,1,1]
    #fail_sum_for_simple_model = sampling_matrix[0,0,0] + sampling_matrix[0,1,2] + sampling_matrix[1,0,1] + sampling_matrix[1,1,0]

    epsilon_list.append(epsilon)
    freq_list_00.append(siuk_freq[0,0])
    freq_list_01.append(siuk_freq[0,1])
    freq_list_10.append(siuk_freq[1,0])
    freq_list_11.append(siuk_freq[1,1])
    freq_list_20.append(siuk_freq[2,0])
    freq_list_21.append(siuk_freq[2,1])

    #freq_list.append(distribution[1])


from matplotlib import pyplot as plt
plt.plot(epsilon_list, freq_list_00, marker='x', label="$(s_1,u_1)$")
plt.plot(epsilon_list, freq_list_01, marker='o', label="$(s_1,u_2)$")
plt.plot(epsilon_list, freq_list_10, marker='^', label="$(s_2,u_1)$")
plt.plot(epsilon_list, freq_list_11, marker='x', label="$(s_2,u_2)$")
plt.plot(epsilon_list, freq_list_20, marker='^', label="$(s_3,u_1)$")
plt.plot(epsilon_list, freq_list_21, marker='x', label="$(s_3,u_2)$")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("$\epsilon$",fontsize=18)
plt.ylabel("probability of visiting $(s_i,u_k)$",fontsize=18, labelpad=-2)

plt.legend()
plt.savefig("fig.pdf")
plt.show()
#print(log_simple_model_marginal_likelihood(2))
#print(log_flexible_model_marginal_likelihood(2))

