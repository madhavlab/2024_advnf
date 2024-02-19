import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt


def periodic_padding(x, padding=1):
    p = padding
    top_left = x[:, -p:, -p:, :] # top left
    top_center = x[:, -p:, :, :] # top center
    top_right = x[:, -p:, :p, :] # top right
    middle_left = x[:, :, -p:,:] # middle left
    middle_center = x # middle center
    middle_right = x[:, :, :p, :] # middle right
    bottom_left = x[:, :p, -p:, :] # bottom left
    bottom_center = x[:, :p, :, :] # bottom center
    bottom_right = x[:, :p, :p, :] # bottom right
    top = tf.concat([top_left, top_center, top_right], axis=2)
    middle = tf.concat([middle_left, middle_center, middle_right], axis=2)
    bottom = tf.concat([bottom_left, bottom_center, bottom_right], axis=2)
    padded_x = tf.concat([top, middle, bottom], axis=1)
    return padded_x

def calculate_energy(lattices,J=1,K=1):  
    x  = lattices.shape[1]
    y  = lattices.shape[2]
    
    L = np.reshape(lattices,(-1,x,y))
    H  = np.zeros(L.shape)
    
    for i in range(x):
        for j in range(y):
            H[:,i,j] -= 0.50*J*np.cos(2 * np.pi * ( L[:,i,j] -  L[:,i,(j+1)%y]))
            H[:,i,j] -= 0.50*J*np.cos(2 * np.pi * ( L[:,i,j] -  L[:,i,(j-1)%y]))
            H[:,i,j] -= 0.50*J*np.cos(2 * np.pi * ( L[:,i,j] -  L[:,(i+1)%x,j]))
            H[:,i,j] -= 0.50*J*np.cos(2 * np.pi * ( L[:,i,j] -  L[:,(i-1)%x,j]))
            H[:,i,j] -= 0.25*K*np.cos(2 * np.pi * ( L[:,i,j] +  L[:,(i-1)%x,(j-1)%y] -  L[:,i,(j-1)%y] -  L[:,(i-1)%x,j]))
            H[:,i,j] -= 0.25*K*np.cos(2 * np.pi * ( L[:,i,j] +  L[:,(i-1)%x,(j+1)%y] -  L[:,i,(j+1)%y] -  L[:,(i-1)%x,j]))
            H[:,i,j] -= 0.25*K*np.cos(2 * np.pi * ( L[:,i,j] +  L[:,(i+1)%x,(j-1)%y] -  L[:,i,(j-1)%y] -  L[:,(i+1)%x,j]))
            H[:,i,j] -= 0.25*K*np.cos(2 * np.pi * ( L[:,i,j] +  L[:,(i+1)%x,(j+1)%y] -  L[:,i,(j+1)%y] -  L[:,(i+1)%x,j]))
        
    return H       
    
def get_energy(lattices,J,K):
    
    H = calculate_energy(lattices,J=J,K=K) 
    energy      = (tf.reduce_mean(H,axis=[1,2]))
    mean_energy = tf.reduce_mean(energy)
    var_energy  = tf.math.reduce_std(energy)
    
    return [energy,mean_energy,var_energy]
    
def calculate_magnetization(lattices):
   
    [n,x,y,channels] = lattices.shape
    if channels==1:
        mag_cos = tf.reduce_mean(tf.reshape(tf.math.cos(2*np.pi*lattices),
                                            shape=[n,x,y]),axis=[1,2])
        mag_sin = tf.reduce_mean(tf.reshape(tf.math.sin(2*np.pi*lattices),
                                            shape=[n,x,y]),axis=[1,2])
        
    else:
        mag_cos = tf.reduce_mean(lattices[:,:,:,0],axis=[1,2])
        mag_sin = tf.reduce_mean(lattices[:,:,:,1],axis=[1,2])
        
    mag = tf.math.sqrt(mag_cos**2 + mag_sin**2)
    
    mean_mag_cos = tf.reduce_mean(mag_cos)                    
    mean_mag_sin = tf.reduce_mean(mag_sin)
    mean_mag     = tf.math.sqrt(mean_mag_cos**2 + mean_mag_sin**2)
    mean_mag1    = tf.reduce_mean(mag)
    
    mag_susceptibility = tf.math.reduce_std(mag)
    return [mag,mean_mag,mean_mag1,mag_susceptibility]
    
def calculate_specific_heat(lattices,T,J,K):
    
    [energy,mean_energy,var_energy] = get_energy(lattices,J,K)
    specific_heat = var_energy**2/((T**2)*64)
    
    return specific_heat
                                      
                                      
def observable_plot(lattices,samples,min_t,max_t,temp_len,J=1,K=1,name= 'xy_ring'):
    index_set = np.arange(temp_len)
    T_vals = np.linspace(min_t,max_t,temp_len)

    #Empty arrays for holding lattices1 and lattices2 energy , specific heat, magnetization,
    #Magnetization Variance, Mean Energy and Varaiance
    lattices_energy  = []
    lattices_sp_heat = []
    lattices_mag     = []
    lattices_mean_mag = []
    lattices_mean_energy = []
    lattices_var_energy  = []
    lattices_mag_susceptibility = []
    

    for i in index_set:
        [energy,mean_energy,var_energy] = get_energy(lattices[samples*i:samples*(i+1)],J,K)      
        specific_heat  = calculate_specific_heat(lattices[samples*i:samples*(i+1)],T_vals[i],J,K)        
        [mag,mean_mag,mean_new,mag_susceptibility] = calculate_magnetization(lattices[samples*i:samples*(i+1)])
        lattices_energy.append(energy)        
        lattices_mean_energy.append(mean_energy)        
        lattices_var_energy.append(var_energy)      
                
        lattices_sp_heat.append(specific_heat)  
        lattices_mag.append(mag) 
        lattices_mag_susceptibility.append(mag_susceptibility)
        
        lattices_mean_mag.append(mean_new)
     
    lower_limit_energy = list(np.array(lattices_mean_energy) - np.array(lattices_var_energy))
    upper_limit_energy = list(np.array(lattices_mean_energy) + np.array(lattices_var_energy))
    
    lower_limit_mag = list(np.array(lattices_mean_mag) - np.array(lattices_mag_susceptibility))
    upper_limit_mag = list(np.array(lattices_mean_mag) + np.array(lattices_mag_susceptibility))
    
    fig, axs = plt.subplots(3,figsize=(7,15))


    #Plot of Specific Heat
    
    
    axs[0].plot(T_vals,lattices_sp_heat,'r',marker='o')
    axs[0].set_xlabel('Temperature')
    axs[0].set_ylabel('Specific Heat')
        
    #Plot of Mean Energy
    axs[1].plot(T_vals,lattices_mean_energy,'r',marker='o')
    axs[1].fill_between(T_vals,lower_limit_energy,upper_limit_energy,color='red', alpha=0.1)
    axs[1].set_xlabel('Temperature')
    axs[1].set_ylabel('Mean Energy')
   

    #Plot of Mean Magnetization.
    axs[2].plot(T_vals,lattices_mean_mag,'r',marker='o')
    axs[2].fill_between(T_vals,lower_limit_mag,upper_limit_mag,color='red', alpha=0.1)
    
    axs[2].set_xlabel('Temperature')
    axs[2].set_ylabel('Mean Magnetization')
    fig.savefig(name +'.png')                                 
    
def comparison_plot(lattices1,lattices2,samples,min_t,max_t,temp_len,J=1,K=1,name='comparison_plot'):
    
    index_set = np.arange(temp_len)
    T_vals = np.linspace(min_t,max_t,temp_len)

    #Empty arrays for holding lattices1 and lattices2 energy , specific heat, magnetization,
    #Magnetization Variance, Mean Energy and Varaiance
    lattices1_energy  = []
    lattices2_energy  = []
    lattices1_sp_heat = []
    lattices2_sp_heat = []
    lattices1_mag     = []
    lattices2_mag     = []
    lattices1_mean_mag = []
    lattices2_mean_mag = []
    lattices1_mean_energy = []
    lattices2_mean_energy = []
    lattices1_var_energy  = []
    lattices2_var_energy  = []
    lattices1_mag_susceptibility = []
    lattices2_mag_susceptibility = []

    for i in index_set:
        [energy1,mean_energy1,var_energy1] = get_energy(lattices1[samples*i:samples*(i+1)],J,K)
        [energy2,mean_energy2,var_energy2] = get_energy(lattices2[samples*i:samples*(i+1)],J,K)
        
        specific_heat1 = calculate_specific_heat(lattices1[samples*i:samples*(i+1)],T_vals[i],J,K)
        specific_heat2 = calculate_specific_heat(lattices2[samples*i:samples*(i+1)],T_vals[i],J,K)
        
        [mag1,mean_mag1,mean_new1,mag_susceptibility1] = calculate_magnetization(lattices1[samples*i:samples*(i+1)])
        [mag2,mean_mag2,mean_new2,mag_susceptibility2] = calculate_magnetization(lattices2[samples*i:samples*(i+1)])
        
        lattices1_energy.append(energy1)
        lattices2_energy.append(energy2)
        
        lattices1_mean_energy.append(mean_energy1)
        lattices2_mean_energy.append(mean_energy2)
        
        lattices1_var_energy.append(var_energy1)
        lattices2_var_energy.append(var_energy2)
        
                
        lattices1_sp_heat.append(specific_heat1)
        lattices2_sp_heat.append(specific_heat2)
        
        
        lattices1_mag.append(mag1) 
        lattices2_mag.append(mag2)
        
        lattices1_mag_susceptibility.append(mag_susceptibility1)
        lattices2_mag_susceptibility.append(mag_susceptibility2)
        
        lattices1_mean_mag.append(mean_new1)
        lattices2_mean_mag.append(mean_new2)
     
    lower_limit1_energy = list(np.array(lattices1_mean_energy)-np.array(lattices1_var_energy))
    lower_limit2_energy = list(np.array(lattices2_mean_energy)-np.array(lattices2_var_energy))
    upper_limit1_energy = list(np.array(lattices1_mean_energy)+np.array(lattices1_var_energy))
    upper_limit2_energy = list(np.array(lattices2_mean_energy)+np.array(lattices2_var_energy))
    
    lower_limit1_mag = list(np.array(lattices1_mean_mag)-np.array(lattices1_mag_susceptibility))
    lower_limit2_mag = list(np.array(lattices2_mean_mag)-np.array(lattices2_mag_susceptibility))
    upper_limit1_mag = list(np.array(lattices1_mean_mag)+np.array(lattices1_mag_susceptibility))
    upper_limit2_mag = list(np.array(lattices2_mean_mag)+np.array(lattices2_mag_susceptibility))
    
    fig, axs = plt.subplots(3,figsize=(7,15))


    #Plot of Specific Heat
    
    
    axs[0].plot(T_vals,lattices1_sp_heat,'r',marker='o')
    axs[0].plot(T_vals,lattices2_sp_heat,'g',marker='^')
    axs[0].set_xlabel('Temperature')
    axs[0].set_ylabel('Specific Heat')
    axs[0].legend(['MCMC Samples','Generated Samples'])
    #plt.savefig('./cgan_samples/sp_heat_epoch_%s.png'%(epoch))
    #plt.show()
    
    #Plot of Mean Energy
    axs[1].plot(T_vals,lattices1_mean_energy,'r',marker='o')
    axs[1].fill_between(T_vals,lower_limit1_energy,upper_limit1_energy,color='red', alpha=0.1)
    axs[1].plot(T_vals,lattices2_mean_energy,'g',marker='o')
    axs[1].fill_between(T_vals,lower_limit2_energy,upper_limit2_energy,color='green', alpha=0.1)
    axs[1].set_xlabel('Temperature')
    axs[1].set_ylabel('Mean Energy')
    axs[1].legend(['MCMC Samples','Generated  Samples'])
    #plt.savefig('./cgan_samples/mean_energy_epoch_%s.png'%(epoch))
    #plt.show()

    #Plot of Mean Magnetization.
    axs[2].plot(T_vals,lattices1_mean_mag,'r',marker='o')
    axs[2].fill_between(T_vals,lower_limit1_mag,upper_limit1_mag,color='red', alpha=0.1)
    axs[2].plot(T_vals,lattices2_mean_mag,'g',marker='o')
    axs[2].fill_between(T_vals,lower_limit2_mag,upper_limit2_mag,color='green', alpha=0.1)
    axs[2].set_xlabel('Temperature')
    axs[2].set_ylabel('Mean Magnetization')
    axs[2].legend(['MCMC Samples','Generated Samples'])
    fig.savefig(name +'.png')
    #plt.show()

    
def earth_mover_distance(A,B):
    n = len(A)
    dist = np.zeros(n)    
    for i in range(1,n):
        dist[i] = A[i-1]-B[i-1]+dist[i-1]
    return np.sum(abs(dist))/(n*1000)

def percent_overlap(hist_1, hist_2):# For evaluation : Calculates the % Overlap between two Histograms
    overlap_each_bin = np.minimum(hist_1, hist_2)
    total_overlap    = np.true_divide(np.sum(overlap_each_bin), np.sum(hist_2))
    return total_overlap*100

def evaluation_metrics(lattices1,lattices2,J=1,K=1):
    nsamples = 1000
    index_set = np.arange(32)
    T_vals = np.linspace(0.05,2.05,32)

    #Empty arrays for holding lattices1 and lattices2 energy 
    lattices1_energy  = []
    lattices2_energy  = []
    lattices1_mag     = []
    lattices2_mag     = []
    overlap_energy    = []
    overlap_mag       = []
    EMD_energy        = []
    EMD_mag           = []
    
    for i in index_set:
        [energy1,mean_energy1,var_energy1] = get_energy(lattices1[1000*i:1000*(i+1)],J,K)
        [energy2,mean_energy2,var_energy2] = get_energy(lattices2[1000*i:1000*(i+1)],J,K)
        
        hist1_energy,bin1_energy = np.histogram(np.array(energy1),bins =80,range=[-2, 0])
        hist2_energy,bin2_energy = np.histogram(np.array(energy2),bins =80,range=[-2, 0])
            
        [mag1,mean_mag1,mean_new1,mag_susceptibility1] = calculate_magnetization(lattices1[1000*i:1000*(i+1)])
        [mag2,mean_mag2,mean_new2,mag_susceptibility2] = calculate_magnetization(lattices2[1000*i:1000*(i+1)])
        
        hist1_mag,bin1_mag = np.histogram(mag1,bins =40,range=[0, 1])
        hist2_mag,bin2_mag = np.histogram(mag2,bins =40,range=[0, 1])
        
        overlap_energy.append(percent_overlap(hist1_energy,hist2_energy))
        overlap_mag.append(percent_overlap(hist1_mag,hist2_mag))
        
        EMD_energy.append(earth_mover_distance(hist2_energy,hist1_energy))
        EMD_mag.append(earth_mover_distance(hist2_mag,hist1_mag))
    
    Energy_overlap         = [np.mean(np.array(overlap_energy)),np.std(np.array(overlap_energy))]
    Magnetization_overlap  = [np.mean(np.array(overlap_mag)),np.std(np.array(overlap_mag))]
    Energy_wass_dist       = [np.mean(np.array(EMD_energy)),np.std(np.array(EMD_energy))]
    Magnetization_wass_dist= [np.mean(np.array(EMD_mag)),np.std(np.array(EMD_mag))]
   
    return [Energy_overlap,Magnetization_overlap,Energy_wass_dist,Magnetization_wass_dist]

def multi_lattice_comparison_plot(lattices1,lattices2,lattices3,samples,min_t,max_t,temp_len,J=1,K=1,name = 'multi_lattice_plot'):
    index_set = np.arange(temp_len)
    T_vals = np.linspace(min_t,max_t,temp_len)

    #Empty arrays for holding lattices1 and lattices2 energy , specific heat, magnetization,
    #Magnetization Variance, Mean Energy and Varaiance
    lattices1_energy  = []
    lattices2_energy  = []
    lattices3_energy  = []
    
    lattices1_sp_heat = []
    lattices2_sp_heat = []
    lattices3_sp_heat = []
    
    lattices1_mag     = []
    lattices2_mag     = []
    lattices3_mag     = []
    
    lattices1_mean_mag = []
    lattices2_mean_mag = []
    lattices3_mean_mag = []
    
    lattices1_mean_energy = []
    lattices2_mean_energy = []
    lattices3_mean_energy = []
    
    lattices1_var_energy  = []
    lattices2_var_energy  = []
    lattices3_var_energy  = []
    
    lattices1_mag_susceptibility = []
    lattices2_mag_susceptibility = []
    lattices3_mag_susceptibility = []

    for i in index_set:
        [energy1,mean_energy1,var_energy1] = get_energy(lattices1[samples*i:samples*(i+1)],J,K)
        [energy2,mean_energy2,var_energy2] = get_energy(lattices2[samples*i:samples*(i+1)],J,K)
        [energy3,mean_energy3,var_energy3] = get_energy(lattices3[samples*i:samples*(i+1)],J,K)
        
        specific_heat1 = calculate_specific_heat(lattices1[samples*i:samples*(i+1)],T_vals[i],J,K)
        specific_heat2 = calculate_specific_heat(lattices2[samples*i:samples*(i+1)],T_vals[i],J,K)
        specific_heat3 = calculate_specific_heat(lattices3[samples*i:samples*(i+1)],T_vals[i],J,K)
        
        [mag1,mean_mag1,mean_new1,mag_susceptibility1] = calculate_magnetization(lattices1[samples*i:samples*(i+1)])
        [mag2,mean_mag2,mean_new2,mag_susceptibility2] = calculate_magnetization(lattices2[samples*i:samples*(i+1)])
        [mag3,mean_mag3,mean_new3,mag_susceptibility3] = calculate_magnetization(lattices3[samples*i:samples*(i+1)])
        
        lattices1_energy.append(energy1)
        lattices2_energy.append(energy2)
        lattices3_energy.append(energy3)
        
        lattices1_mean_energy.append(mean_energy1)
        lattices2_mean_energy.append(mean_energy2)
        lattices3_mean_energy.append(mean_energy3)
        
        lattices1_var_energy.append(var_energy1)
        lattices2_var_energy.append(var_energy2)
        lattices3_var_energy.append(var_energy3)
        
                
        lattices1_sp_heat.append(specific_heat1)
        lattices2_sp_heat.append(specific_heat2)
        lattices3_sp_heat.append(specific_heat3)
        
        lattices1_mag.append(mag1) 
        lattices2_mag.append(mag2)
        lattices3_mag.append(mag3)
        
        lattices1_mag_susceptibility.append(mag_susceptibility1)
        lattices2_mag_susceptibility.append(mag_susceptibility2)
        lattices3_mag_susceptibility.append(mag_susceptibility3)
        
        lattices1_mean_mag.append(mean_new1)
        lattices2_mean_mag.append(mean_new2)
        lattices3_mean_mag.append(mean_new3)
     
    lower_limit1_energy = list(np.array(lattices1_mean_energy)-np.array(lattices1_var_energy))
    lower_limit2_energy = list(np.array(lattices2_mean_energy)-np.array(lattices2_var_energy))
    lower_limit3_energy = list(np.array(lattices3_mean_energy)-np.array(lattices3_var_energy))
    
    upper_limit1_energy = list(np.array(lattices1_mean_energy)+np.array(lattices1_var_energy))
    upper_limit2_energy = list(np.array(lattices2_mean_energy)+np.array(lattices2_var_energy))
    upper_limit3_energy = list(np.array(lattices3_mean_energy)+np.array(lattices3_var_energy))
    
    lower_limit1_mag = list(np.array(lattices1_mean_mag)-np.array(lattices1_mag_susceptibility))
    lower_limit2_mag = list(np.array(lattices2_mean_mag)-np.array(lattices2_mag_susceptibility))
    lower_limit3_mag = list(np.array(lattices3_mean_mag)-np.array(lattices3_mag_susceptibility))
    
    upper_limit1_mag = list(np.array(lattices1_mean_mag)+np.array(lattices1_mag_susceptibility))
    upper_limit2_mag = list(np.array(lattices2_mean_mag)+np.array(lattices2_mag_susceptibility))
    upper_limit3_mag = list(np.array(lattices3_mean_mag)+np.array(lattices3_mag_susceptibility))
    
    fig, axs = plt.subplots(3,figsize=(7,15))


    #Plot of Specific Heat
    
    
    axs[0].plot(T_vals,lattices1_sp_heat,'r',marker='o')
    axs[0].plot(T_vals,lattices2_sp_heat,'g',marker='^')
    axs[0].plot(T_vals,lattices3_sp_heat,'b',marker='*')
    axs[0].set_xlabel('Temperature')
    axs[0].set_ylabel('Specific Heat')
    axs[0].legend(['MCMC Samples','Generated Samples','MH-Model'])
        
    #Plot of Mean Energy
    axs[1].plot(T_vals,lattices1_mean_energy,'r',marker='o')
    axs[1].fill_between(T_vals,lower_limit1_energy,upper_limit1_energy,color='red', alpha=0.1)
    axs[1].plot(T_vals,lattices2_mean_energy,'g',marker='o')
    axs[1].fill_between(T_vals,lower_limit2_energy,upper_limit2_energy,color='green', alpha=0.1)
    axs[1].plot(T_vals,lattices3_mean_energy,'b',marker='o')
    axs[1].fill_between(T_vals,lower_limit3_energy,upper_limit3_energy,color='blue', alpha=0.1)
    axs[1].set_xlabel('Temperature')
    axs[1].set_ylabel('Mean Energy')
    axs[1].legend(['MCMC Samples','Generated  Samples','MH-Model'])
    

    #Plot of Mean Magnetization.
    axs[2].plot(T_vals,lattices1_mean_mag,'r',marker='o')
    axs[2].fill_between(T_vals,lower_limit1_mag,upper_limit1_mag,color='red', alpha=0.1)
    axs[2].plot(T_vals,lattices2_mean_mag,'g',marker='o')
    axs[2].fill_between(T_vals,lower_limit2_mag,upper_limit2_mag,color='green', alpha=0.1)
    axs[2].plot(T_vals,lattices3_mean_mag,'b',marker='o')
    axs[2].fill_between(T_vals,lower_limit3_mag,upper_limit3_mag,color='blue', alpha=0.1)
    axs[2].set_xlabel('Temperature')
    axs[2].set_ylabel('Mean Magnetization')
    axs[2].legend(['MCMC Samples','Generated Samples','MH-Model'])
    fig.savefig(name +'.png')
    #plt.show()

    
    
    